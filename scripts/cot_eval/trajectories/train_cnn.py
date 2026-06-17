import argparse
import logging
import os
import sys
import tempfile
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as L
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from tqdm import tqdm

# Add parent directory to path to import stats_utils and src.utils
sys.path.append(str(Path(__file__).parent.parent.absolute()))
sys.path.append(str(Path(__file__).parent.parent.parent.parent.absolute()))

from stats_utils import bootstrap_metrics
from src.utils import load_predictions

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Data Loading Utilities ---

def parse_harmfulness(pred_str: str) -> int:
    if not isinstance(pred_str, str):
        return -1
    lower_pred = pred_str.lower()
    if "harmful response: yes" in lower_pred:
        return 1
    elif "harmful response: no" in lower_pred:
        return 0
    if lower_pred.strip() == "yes":
        return 1
    if lower_pred.strip() == "no":
        return 0
    return -1

def detect_dataset(task_name: str, wg_results_dir: Path) -> str:
    if not wg_results_dir.exists():
        return None
    datasets = sorted([d.name for d in wg_results_dir.iterdir() if d.is_dir()], key=len, reverse=True)
    for ds in datasets:
        if task_name.startswith(ds):
            return ds
    return None

def load_trajectories(args):
    if args.mode == "harmfulness":
        base_dir = Path("results/harmfulness/mil_probe_eval")
        wg_results_dir = Path("results/harmfulness/eval_wildguard")
    else:
        base_dir = Path("results/math/mil_probe_eval")
        wg_results_dir = None

    json_files = list(base_dir.rglob(f"{args.model}/*_pool_over_prompt/cot_mil_probe_preds_cummax.json"))
    parquet_files = list(base_dir.rglob(f"{args.model}/*_pool_over_prompt/cot_mil_probe_preds_cummax.parquet"))
    
    files_dict = {}
    for f in json_files + parquet_files:
        if args.experiment_id and f.parent.name not in args.experiment_id:
            continue
        base = f.with_suffix("")
        if base not in files_dict or f.suffix == ".parquet":
            files_dict[base] = f
    
    all_files = sorted(list(files_dict.values()))
    if not all_files:
        raise FileNotFoundError(f"No trajectory files found for model {args.model}")

    all_data = []
    wg_cache = {}

    for file_path in tqdm(all_files, desc="Loading trajectories"):
        experiment_id = file_path.parent.name
        task_name = experiment_id.replace("_pool_over_prompt", "")
        
        cot_labels = []
        resp_labels = []

        if args.mode == "harmfulness":
            dataset_name = detect_dataset(task_name, wg_results_dir)
            if dataset_name:
                cache_key = (args.model, dataset_name)
                if cache_key not in wg_cache:
                    wg_model_dir = wg_results_dir / dataset_name / args.model
                    try:
                        cot_data = load_predictions(wg_model_dir / "preds_cot.json")
                        cot_labels = [parse_harmfulness(p) for p in cot_data]
                        resp_data = load_predictions(wg_model_dir / "preds_response.json")
                        resp_labels = [parse_harmfulness(p) for p in resp_data]
                        wg_cache[cache_key] = (cot_labels, resp_labels)
                    except Exception as e:
                        logger.warning(f"Could not load WG labels for {dataset_name}: {e}")
                        wg_cache[cache_key] = (None, None)
                cot_labels, resp_labels = wg_cache[cache_key]

        data = load_predictions(file_path)

        for i, sample in enumerate(data):
            prompt_preds = sample.get("prompt_token_predictions", [])
            cot_preds = sample.get("token_predictions", [])
            
            prompt_probs = [t.get("harmful_prob", 0.0) for t in prompt_preds]
            cot_probs = [t.get("harmful_prob", 0.0) for t in cot_preds]
            
            trajectory = np.array(prompt_probs + cot_probs, dtype=np.float32)
            
            c_label = -1
            r_label = -1
            if args.mode == "harmfulness":
                c_label = cot_labels[i] if cot_labels and i < len(cot_labels) else -1
                r_label = resp_labels[i] if resp_labels and i < len(resp_labels) else -1
                if c_label == -1 or r_label == -1:
                    continue
                label = r_label
            else:
                label = 1 - sample.get("label", 1) 
                r_label = label

            all_data.append({
                "trajectory": trajectory,
                "label": label,
                "cot_label": c_label,
                "resp_label": r_label,
                "prompt_len": len(prompt_probs),
                "num_cot_tokens": len(cot_probs),
                "sample_id": sample.get("sample_id"),
                "task": task_name,
                "experiment_id": experiment_id,
                "baseline_prob": cot_probs[-1] if cot_probs else 0.0,
            })

    return pd.DataFrame(all_data)

# --- PyTorch Components ---

class TrajectoryDataset(Dataset):
    def __init__(self, trajectories, prompt_lens, labels, target_len=512):
        self.trajectories = trajectories
        self.prompt_lens = prompt_lens
        self.labels = labels
        self.target_len = target_len

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj_raw = self.trajectories[idx]
        prompt_len = self.prompt_lens[idx]
        label = self.labels[idx]
        
        traj = torch.tensor(traj_raw, dtype=torch.float32)
        boundary = torch.zeros_like(traj)
        boundary[prompt_len:] = 1.0
        
        traj_interp = F.interpolate(traj.view(1,1,-1), size=self.target_len, mode='linear', 
                             align_corners=False).squeeze(0)
        boundary_interp = F.interpolate(boundary.view(1,1,-1), size=self.target_len, 
                                 mode='nearest').squeeze(0)
        
        x = torch.cat([traj_interp, boundary_interp], dim=0)
        return x, torch.tensor(label, dtype=torch.long)

class CNNClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv_local  = nn.Conv1d(2, 32, kernel_size=5,  padding=2)
        self.conv_mid    = nn.Conv1d(2, 32, kernel_size=21, padding=10)
        self.conv_global = nn.Conv1d(2, 32, kernel_size=51, padding=25)
        
        self.bn1 = nn.BatchNorm1d(96)
        self.conv2 = nn.Conv1d(96, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.4)
        
        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        x1 = F.gelu(self.conv_local(x))
        x2 = F.gelu(self.conv_mid(x))
        x3 = F.gelu(self.conv_global(x))
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.bn1(x)
        
        x = F.gelu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        
        x_avg = x.mean(dim=-1)
        x_max = x.amax(dim=-1)
        x = torch.cat([x_avg, x_max], dim=1)
        
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

class LitCNN(L.LightningModule):
    def __init__(self, model, lr=1e-3, weight_decay=0.05, pos_weight=None):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        if pos_weight is not None:
            self.register_buffer("weight", pos_weight)
        else:
            self.weight = None
        self.validation_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y, weight=self.weight)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y, weight=self.weight)
        probs = F.softmax(logits, dim=1)[:, 1]
        
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.validation_step_outputs.append({"probs": probs, "labels": y})
        return loss

    def on_validation_epoch_end(self):
        all_probs = torch.cat([x["probs"] for x in self.validation_step_outputs]).cpu().numpy()
        all_labels = torch.cat([x["labels"] for x in self.validation_step_outputs]).cpu().numpy()
        self.validation_step_outputs.clear()
        
        if len(np.unique(all_labels)) > 1:
            auc = roc_auc_score(all_labels, all_probs)
        else:
            auc = 0.5
            logger.warning("Single class in validation fold, AUC set to 0.5")
            
        self.log("val_auc", auc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

# --- Main Logic ---

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--experiment_id", type=str, nargs="+")
    parser.add_argument("--mode", type=str, choices=["harmfulness", "math"], required=True)
    parser.add_argument("--output_csv", type=str, default="results/cnn_performance_aggregation.csv")
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--target_len", type=int, default=512)
    parser.add_argument("--n_splits", type=int, default=3)
    return parser.parse_args()

def main():
    args = parse_args()
    logger.info(f"Training CNN for model {args.model} in mode {args.mode}")

    df = load_trajectories(args)
    
    initial_len = len(df)
    df = df[df.num_cot_tokens > 0].copy()
    if len(df) < initial_len:
        logger.info(f"Dropped {initial_len - len(df)} samples with 0 CoT tokens.")

    logger.info(f"Final dataset size: {len(df)} trajectories")

    trajectories = df["trajectory"].values
    prompt_lens = df["prompt_len"].values
    labels = df["label"].values
    
    cv = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    all_probs = np.zeros(len(df))
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(trajectories, labels)):
        logger.info(f"Fold {fold+1}/{args.n_splits}")
        L.seed_everything(42 + fold)
        
        train_ds = TrajectoryDataset(trajectories[train_idx], prompt_lens[train_idx], labels[train_idx], target_len=args.target_len)
        val_ds = TrajectoryDataset(trajectories[val_idx], prompt_lens[val_idx], labels[val_idx], target_len=args.target_len)
        
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size)
        
        num_pos = np.sum(labels[train_idx] == 1)
        num_neg = np.sum(labels[train_idx] == 0)
        if num_pos > 0 and num_neg > 0:
            pos_weight_val = num_neg / num_pos
            pos_weight = torch.tensor([1.0, pos_weight_val], dtype=torch.float32)
        else:
            pos_weight = None
        
        cnn = CNNClassifier()
        model = LitCNN(cnn, pos_weight=pos_weight)
        
        temp_dir = tempfile.mkdtemp()
        checkpoint = ModelCheckpoint(
            monitor="val_auc",
            mode="max",
            save_top_k=1,
            dirpath=temp_dir
        )
        early_stop = EarlyStopping(monitor="val_auc", mode="max", patience=10)
        
        trainer = L.Trainer(
            max_epochs=args.epochs,
            accelerator="auto",
            devices=1,
            logger=False,
            callbacks=[early_stop, checkpoint],
            enable_checkpointing=True
        )
        
        trainer.fit(model, train_loader, val_loader)
        
        # Load best model for inference
        if checkpoint.best_model_path:
            logger.info(f"Loading best model from {checkpoint.best_model_path}")
            best_model = LitCNN.load_from_checkpoint(
                checkpoint.best_model_path,
                model=CNNClassifier(),
                pos_weight=pos_weight
            )
        else:
            best_model = model
            
        best_model.eval()
        best_model.to(trainer.strategy.root_device)
        fold_probs = []
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(best_model.device)
                logits = best_model(x)
                probs = F.softmax(logits, dim=1)[:, 1]
                fold_probs.extend(probs.cpu().numpy())
        
        all_probs[val_idx] = fold_probs
        shutil.rmtree(temp_dir)

    # Evaluation
    metrics_dict = {
        "auc": lambda y_t, y_p: roc_auc_score(y_t, y_p),
        "ap": lambda y_t, y_p: average_precision_score(y_t, y_p),
        "f1": lambda y_t, y_p: f1_score(y_t, (y_p > 0.5).astype(int)),
    }
    
    cnn_results = bootstrap_metrics(labels, all_probs, metrics_dict)
    baseline_results = bootstrap_metrics(labels, df["baseline_prob"].values, metrics_dict)
    
    logger.info(f"CNN AUC: {cnn_results['auc']:.4f}")
    logger.info(f"Baseline AUC: {baseline_results['auc']:.4f}")

    if args.mode == "harmfulness":
        unfaithful_mask = df["cot_label"] != df["resp_label"]
        if unfaithful_mask.any():
            try:
                auc_unfaithful = roc_auc_score(labels[unfaithful_mask], all_probs[unfaithful_mask])
                logger.info(f"CNN AUC on unfaithful CoT subset ({unfaithful_mask.sum()} samples): {auc_unfaithful:.4f}")
                cnn_results["auc_unfaithful"] = auc_unfaithful
            except Exception as e:
                logger.warning(f"Could not calculate unfaithful AUC: {e}")

    exp_str = "+".join(args.experiment_id) if args.experiment_id else "all"
    row = {
        "model": args.model,
        "experiment_id": exp_str,
        "mode": args.mode,
        "num_samples": len(df),
    }
    for k, v in cnn_results.items():
        row[f"cnn_{k}"] = v
    for k, v in baseline_results.items():
        row[f"baseline_{k}"] = v
        
    df_res = pd.DataFrame([row])
    if not os.path.exists(args.output_csv):
        df_res.to_csv(args.output_csv, index=False)
    else:
        df_res.to_csv(args.output_csv, mode="a", header=False, index=False)

    if args.save_predictions:
        output_dir = Path(args.output_csv).parent / "predictions"
        output_dir.mkdir(parents=True, exist_ok=True)
        df["cnn_proba"] = all_probs
        pred_file = output_dir / f"{args.model}_{exp_str.replace('+', '_')}_{args.mode}_cnn_preds.parquet"
        df.to_parquet(pred_file, index=False)
        logger.info(f"Saved predictions to {pred_file}")

if __name__ == "__main__":
    main()

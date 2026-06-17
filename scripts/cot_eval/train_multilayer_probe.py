import argparse
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from tqdm import tqdm

sys.path.append(".")

from src.evaluation.utils import calculate_scores
from src.prototype import (
    calculate_means_and_inv_cov,
    get_gda_params,
    get_gda_pred,
    get_nmc_pred,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _extract_token_id(file_name: str) -> Optional[int]:
    match = re.match(r"layer_(.+?)_token_(\d+)_hidden_states\.parquet$", file_name)
    if match:
        return int(match.group(2))
    return None


def _extract_layer_key(file_name: str) -> Optional[str]:
    token_match = re.match(
        r"layer_(.+?)_token_\d+_hidden_states\.parquet$",
        file_name,
    )
    if token_match:
        return token_match.group(1)
    mean_match = re.match(r"layer_(.+?)_mean_hidden_states\.parquet$", file_name)
    if mean_match:
        return mean_match.group(1)
    return None


def _discover_layer_files(
    train_hidden_states_folder: Path,
    use_mean_representation: bool,
    token_id: Optional[int],
    exclude_embedding_layer: bool,
) -> list[Path]:
    files = [x for x in train_hidden_states_folder.iterdir() if x.is_file()]
    selected = []
    for file_path in files:
        file_name = file_path.name
        is_mean = "_mean_hidden_states.parquet" in file_name
        if use_mean_representation and not is_mean:
            continue
        if not use_mean_representation and is_mean:
            continue

        cur_token_id = _extract_token_id(file_name)
        if token_id is not None and cur_token_id != token_id:
            continue

        layer_key = _extract_layer_key(file_name)
        if layer_key is None:
            continue
        if exclude_embedding_layer and layer_key == "0":
            continue
        selected.append(file_path)
    return sorted(selected, key=lambda p: p.name)


def load_data(
    datapath: Path,
    num_samples_per_class: Optional[int] = None,
    random_state: Optional[int] = None,
    num_frac_data: Optional[float] = None,
):
    data = pd.read_parquet(datapath)
    if num_frac_data is not None:
        data = data.sample(frac=num_frac_data, random_state=random_state)
    if num_samples_per_class is not None:
        sampled = []
        for label in data["labels"].unique():
            sampled.append(
                data[data["labels"] == label].sample(
                    num_samples_per_class,
                    random_state=random_state,
                )
            )
        data = pd.concat(sampled).reset_index(drop=True)

    hidden_states = torch.tensor(np.array(data["hidden_state"].tolist())).to(DEVICE)
    labels = torch.tensor(np.array(data["labels"].tolist())).to(DEVICE)
    return hidden_states, labels


def _serialize_layer_params(params: dict, use_nmc: bool) -> dict:
    serialized = {}
    for layer_name, layer_params in params.items():
        if use_nmc:
            serialized[layer_name] = {
                "means": layer_params.detach().cpu().numpy(),
            }
        else:
            serialized[layer_name] = {
                "W": layer_params["W"].detach().cpu().numpy(),
                "b": layer_params["b"].detach().cpu().numpy(),
            }
    return serialized


def main(
    train_hidden_states_folder: Path,
    save_folder: Path,
    num_samples_per_class: Optional[int] = None,
    num_frac_data: Optional[float] = None,
    random_state: Optional[int] = 42,
    use_nmc: bool = False,
    use_distance: bool = False,
    use_mean_representation: bool = False,
    token_id: Optional[int] = None,
    exclude_embedding_layer: bool = True,
    verbose: bool = False,
):
    layer_files = _discover_layer_files(
        train_hidden_states_folder=train_hidden_states_folder,
        use_mean_representation=use_mean_representation,
        token_id=token_id,
        exclude_embedding_layer=exclude_embedding_layer,
    )
    if not layer_files:
        raise ValueError(
            "No hidden-state parquet files matched requested filters in "
            f"{train_hidden_states_folder}."
        )

    all_gda_params = {}
    all_train_preds = {}
    reference_labels = None

    for layer_file in tqdm(layer_files, desc="Training per-layer probes"):
        train_hidden_states, train_labels = load_data(
            layer_file,
            num_samples_per_class=num_samples_per_class,
            random_state=random_state,
            num_frac_data=num_frac_data,
        )

        if reference_labels is None:
            reference_labels = train_labels
        elif train_labels.shape != reference_labels.shape or not torch.equal(
            train_labels, reference_labels
        ):
            raise ValueError(
                "Label mismatch across layer files. Ensure all layer parquet files "
                "were generated from exactly the same dataset order."
            )

        means, inv, _ = calculate_means_and_inv_cov(
            train_hidden_states,
            labels=train_labels,
            calculate_per_class_cov=True,
            scale_covariances=True,
        )
        gda_params = get_gda_params(means, inv)
        all_gda_params[layer_file.name] = gda_params if not use_nmc else means

        if use_nmc:
            cur_pred = (
                get_nmc_pred(
                    train_hidden_states,
                    means,
                    return_probs=not use_distance,
                    return_logits=use_distance,
                )
                .detach()
                .cpu()
                .numpy()
            )
        else:
            cur_pred = (
                get_gda_pred(
                    train_hidden_states,
                    return_probs=not use_distance,
                    return_logits=use_distance,
                    **gda_params,
                )
                .detach()
                .cpu()
                .numpy()
            )
        if use_distance:
            cur_pred = cur_pred[:, 1] - cur_pred[:, 0]
        else:
            cur_pred = cur_pred[:, 1]
        all_train_preds[layer_file.name] = cur_pred

    meta_df = pd.DataFrame(all_train_preds)
    meta_model = make_pipeline(
        LogisticRegressionCV(
            penalty="l1",
            solver="saga",
            scoring="f1",
            max_iter=100000,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
    )
    y_train = reference_labels.cpu().numpy()
    meta_model.fit(meta_df, y_train)
    train_probs = meta_model.predict_proba(meta_df)[:, 1]
    train_scores = calculate_scores(train_probs, y_train)

    if verbose:
        print(f"Best C: {meta_model[-1].C_[0]}")
        print(f"Train scores: {train_scores}")

    save_folder.mkdir(parents=True, exist_ok=True)

    with open(save_folder / "meta_model_coefficients.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                layer_id: coef
                for layer_id, coef in zip(
                    meta_df.columns, meta_model[-1].coef_[0].tolist()
                )
            },
            f,
            indent=4,
            ensure_ascii=True,
        )

    probe_artifact = {
        "use_nmc": use_nmc,
        "use_distance": use_distance,
        "use_mean_representation": use_mean_representation,
        "feature_names": list(meta_df.columns),
        "layer_params": _serialize_layer_params(all_gda_params, use_nmc=use_nmc),
        "meta_model": meta_model,
        "train_scores": train_scores,
        "random_state": random_state,
    }
    with open(save_folder / "multilayer_probe.pkl", "wb") as f:
        pickle.dump(probe_artifact, f)

    with open(save_folder / "probe_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_layers": len(meta_df.columns),
                "feature_names": list(meta_df.columns),
                "use_nmc": use_nmc,
                "use_distance": use_distance,
                "use_mean_representation": use_mean_representation,
                "train_scores": train_scores,
                "random_state": random_state,
            },
            f,
            indent=4,
            ensure_ascii=True,
        )

    print(f"Saved probe to: {save_folder / 'multilayer_probe.pkl'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and save multilayer probe from hidden-state parquet files."
    )
    parser.add_argument(
        "--train_hidden_states_folder",
        type=Path,
        required=True,
        help="Folder with training hidden-state parquet files.",
    )
    parser.add_argument(
        "--save_folder",
        type=Path,
        default=Path("results/cot_eval/probe"),
        help="Folder to save probe artifact and metadata.",
    )
    parser.add_argument("--num_samples_per_class", type=int, default=None)
    parser.add_argument("--num_frac_data", type=float, default=None)
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random state for reproducibility.",
    )
    parser.add_argument("--use_nmc", action="store_true", help="Use NMC per layer.")
    parser.add_argument(
        "--use_distance",
        action="store_true",
        help="Use class-distance/logit difference instead of class probability.",
    )
    parser.add_argument(
        "--use_mean_representation",
        action="store_true",
        help="Train on *_mean_hidden_states.parquet files.",
    )
    parser.add_argument(
        "--token_id",
        type=int,
        default=None,
        help="Only use layer files with this token id (for *_token_{id}_hidden_states.parquet).",
    )
    parser.add_argument(
        "--include_embedding_layer",
        action="store_true",
        help="Include layer_0 files (excluded by default).",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    main(
        train_hidden_states_folder=args.train_hidden_states_folder,
        save_folder=args.save_folder,
        num_samples_per_class=args.num_samples_per_class,
        num_frac_data=args.num_frac_data,
        random_state=args.random_state,
        use_nmc=args.use_nmc,
        use_distance=args.use_distance,
        use_mean_representation=args.use_mean_representation,
        token_id=args.token_id,
        exclude_embedding_layer=not args.include_embedding_layer,
        verbose=args.verbose,
    )

from dataclasses import dataclass
from typing import Optional

import torch
from lightning import LightningModule
from torch.nn import BCEWithLogitsLoss
from torchmetrics import AUROC, Accuracy, F1Score, MetricCollection
from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from src.cot_probe.mil_probe import MILMultiLayerProbe
from src.utils import load_model_for_pred


class CaptureHook:
    """Callable class used as a forward hook to be picklable."""

    def __init__(self, target_dict: dict, name: str):
        self.target_dict = target_dict
        self.name = name

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            self.target_dict[self.name] = output[0].detach()
        else:
            self.target_dict[self.name] = output.detach()


@dataclass
class OptimizerArgs:
    weight_decay: float = 0.01


@dataclass
class LrSchedulerArgs:
    name: str = "linear"
    num_warmup_steps: float = 0.1


class CotProbeModel(LightningModule):
    """Lightning module for training a MIL multi-layer probe on top of a frozen LLM.

    Registers forward hooks on the base model's transformer layers to capture
    hidden states, then passes them through the MILMultiLayerProbe.
    """

    def __init__(
        self,
        base_model_name: str,
        probe_hidden_dims: list[int] = [256, 128],
        activation: str = "gelu",
        layer_indices: list[int] | None = None,
        optimizer_args: OptimizerArgs = OptimizerArgs(),
        learning_rate: float = 1e-3,
        lr_scheduler_args: Optional[LrSchedulerArgs] = None,
        pos_weight: float | None = None,
        d_model: int | None = None,
        num_layers: int | None = None,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["model"])
        self.base_model_name = base_model_name
        self.lr = learning_rate
        self.optimizer_args = optimizer_args
        self.lr_scheduler_args = lr_scheduler_args
        self.layer_indices = layer_indices

        self.layer_names = []
        self.captured_hidden_states = {}
        self.d_model = d_model
        self.num_layers = num_layers
        self.probe_hidden_dims = probe_hidden_dims
        self.activation = activation
        self.pos_weight = pos_weight

        # Loss
        if pos_weight is not None:
            self.loss_fn = BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        else:
            self.loss_fn = BCEWithLogitsLoss()

        # Metrics
        metrics = MetricCollection(
            {
                "Accuracy": Accuracy("binary"),
                "AUROC": AUROC("binary", thresholds=100),
                "F1": F1Score("binary"),
            }
        )
        self.train_metrics = metrics.clone("train/")
        self.val_metrics = metrics.clone("val/")

        # Only save probe weights
        self.strict_loading = False

    def configure_model(self):
        self.model, _ = load_model_for_pred(
            self.base_model_name,
            None,
            merge_and_unload=True,
            device_map="cpu",
            use_fast_tokenizer=False,
            attn_implementation=None,
        )
        for param in self.model.parameters():
            param.requires_grad = False
        all_layer_names = self._get_transformer_layer_names(self.model)
        if self.layer_indices is not None:
            self.layer_names = [all_layer_names[i] for i in self.layer_indices]
        else:
            self.layer_names = all_layer_names
        print(
            f"Using {len(self.layer_names)} layers: {[n.split('.')[-1] for n in self.layer_names]}"
        )

        self.captured_hidden_states: dict[str, torch.Tensor] = {}
        self.hooks = []
        for name in self.layer_names:
            module = dict(self.model.named_modules())[name]
            self.hooks.append(module.register_forward_hook(self._make_hook(name)))

        # Determine d_model from the model config
        if self.d_model is None:
            self.d_model = self._get_d_model(self.model)

        probe_num_layers = len(self.layer_names)
        # Build the probe
        self.probe = MILMultiLayerProbe(
            d_model=self.d_model,
            num_layers=probe_num_layers,
            hidden_dims=self.probe_hidden_dims,
            activation=self.activation,
        )

    @staticmethod
    def _get_transformer_layer_names(model: torch.nn.Module) -> list[str]:
        """Discover transformer layer names by looking for common patterns."""
        layer_names = []
        for name, _ in model.named_modules():
            # Match common transformer layer patterns:
            # model.layers.0, model.model.layers.0, etc.
            parts = name.split(".")
            if len(parts) >= 2 and parts[-2] == "layers" and parts[-1].isdigit():
                layer_names.append(name)
        return sorted(layer_names, key=lambda n: int(n.split(".")[-1]))

    @staticmethod
    def _get_d_model(model: torch.nn.Module) -> int:
        """Extract d_model from the model config."""
        config = getattr(model, "config", None)
        if config is None:
            raise ValueError("Model does not have a config attribute.")
        for attr in ("hidden_size", "d_model", "n_embd"):
            if hasattr(config, attr):
                return getattr(config, attr)
        raise ValueError(
            "Cannot infer d_model from model config. "
            f"Available attrs: {list(vars(config).keys())}"
        )

    def _make_hook(self, name: str):
        return CaptureHook(self.captured_hidden_states, name)

    def _get_hidden_states(
        self, batch: dict
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Run the frozen model and collect hidden states from hooks."""
        self.captured_hidden_states.clear()

        with torch.no_grad():
            self.model(**batch)

        hidden_states_per_layer = [
            self.captured_hidden_states[name] for name in self.layer_names
        ]
        return hidden_states_per_layer

    def _create_response_mask(
        self,
        attention_mask: torch.Tensor,
        response_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Create a mask that is 1 only for the response tokens.

        Args:
            attention_mask: (batch, seq) — 1 for non-padding
            response_lengths: (batch,) — number of response tokens per example

        Returns:
            response_mask: (batch, seq) — 1 for response tokens only
        """
        batch_size, seq_len = attention_mask.shape
        # For each example, valid (non-pad) tokens are at positions where attention_mask=1.
        # The response tokens are the last `response_length` valid tokens.
        valid_lengths = attention_mask.sum(dim=1)  # (batch,)
        response_starts = valid_lengths - response_lengths  # (batch,)

        indices = torch.arange(seq_len, device=attention_mask.device).unsqueeze(
            0
        )  # (1, seq)
        response_mask = (
            indices >= response_starts.unsqueeze(1)
        ) & attention_mask.bool()
        return response_mask.float()

    def forward(self, batch: dict, labels: torch.Tensor):
        response_lengths = batch.pop("response_length", None)

        hidden_states_per_layer = self._get_hidden_states(batch)

        # Build the mask for max-pooling: only response tokens
        if response_lengths is not None:
            pool_mask = self._create_response_mask(
                batch["attention_mask"], response_lengths
            )
        else:
            pool_mask = batch["attention_mask"]

        final_logits, per_layer_logits = self.probe(
            hidden_states_per_layer, attention_mask=pool_mask
        )

        return final_logits.squeeze(-1), per_layer_logits

    def training_step(self, batch, **kwargs):
        labels = batch.pop("labels").float()

        logits, _ = self.forward(batch, labels)
        loss = self.loss_fn(logits, labels)

        preds = torch.sigmoid(logits)
        self.train_metrics(preds, labels.int())
        self.log("train/loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        self.log_dict(self.train_metrics, on_epoch=True, on_step=True)

        return loss

    def validation_step(self, batch, **kwargs):
        labels = batch.pop("labels").float()

        logits, _ = self.forward(batch, labels)
        loss = self.loss_fn(logits, labels)

        preds = torch.sigmoid(logits)
        self.val_metrics(preds, labels.int())
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def on_validation_epoch_end(self):
        val_metrics = self.val_metrics.compute()
        self.log_dict(val_metrics, on_step=False, on_epoch=True, sync_dist=True)
        self.val_metrics.reset()

    def on_train_epoch_end(self):
        self.train_metrics.reset()

    def configure_optimizers(self):
        # Only optimize the probe parameters
        optimizer = torch.optim.AdamW(
            params=self.probe.parameters(),
            lr=self.lr,
            **self.optimizer_args.__dict__,
        )
        if self.lr_scheduler_args is not None:
            num_training_steps = self.trainer.estimated_stepping_batches
            num_warmup_steps = int(
                self.lr_scheduler_args.num_warmup_steps * num_training_steps
            )
            if self.lr_scheduler_args.name == "linear":
                lr_scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=num_training_steps,
                )
            elif self.lr_scheduler_args.name == "cosine":
                lr_scheduler = get_cosine_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=num_training_steps,
                )
            else:
                raise ValueError(
                    f"Scheduler {self.lr_scheduler_args.name} is not supported"
                )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "interval": "step",
                },
            }
        return optimizer

    def on_save_checkpoint(self, checkpoint):
        # Only keep probe weights in checkpoint
        checkpoint["state_dict"] = {
            k: v for k, v in checkpoint["state_dict"].items() if "probe" in k
        }

    def predict_from_hidden_states(
        self,
        hidden_states_per_layer: list[torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        use_cummax: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the probe on pre-captured hidden states (no model forward needed).

        Args:
            hidden_states_per_layer: list of tensors, each (batch, seq, d_model)
                or (batch, d_model).
            attention_mask: (batch, seq) — only needed for 3D inputs.
            use_cummax: if True, return per-position predictions using cumulative max.

        Returns:
            final_logits: (batch,) or (batch, seq) if use_cummax
            per_layer_logits: (batch, num_layers) or (batch, seq, num_layers)
        """
        final_logits, per_layer_logits = self.probe(
            hidden_states_per_layer,
            attention_mask=attention_mask,
            use_cummax=use_cummax,
        )
        return final_logits.squeeze(-1), per_layer_logits

    @classmethod
    def load_probe(
        cls,
        checkpoint_path: str,
        map_location: str | torch.device = "cpu",
        **kwargs,
    ) -> "CotProbeModel":
        """Load a trained probe from a Lightning checkpoint without a base model.

        Uses saved hyperparameters when available (from save_hyperparameters),
        falls back to inferring architecture from saved weights for older
        checkpoints.

        Args:
            checkpoint_path: path to the .ckpt file
            map_location: device to load weights onto
            **kwargs: overrides for CotProbeModel.__init__

        Returns:
            CotProbeModel with loaded probe weights (model=None)
        """
        checkpoint = torch.load(
            checkpoint_path, map_location=map_location, weights_only=False
        )
        state_dict = checkpoint.get("state_dict", checkpoint)
        hparams = checkpoint.get("hyper_parameters", {})

        inferred_d_model = None
        inferred_num_layers = 0
        for key in state_dict:
            if "layer_probes" in key and "transform.0.weight" in key:
                inferred_num_layers += 1
                if inferred_d_model is None:
                    inferred_d_model = state_dict[key].shape[1]

        if hparams:
            # Use saved hyperparameters
            init_kwargs = {**hparams, **kwargs}
            init_kwargs.pop("model", None)
            
            # Fallback for older probes that don't have base_model_name saved
            if "base_model_name" not in init_kwargs:
                init_kwargs["base_model_name"] = "unknown"

            if init_kwargs.get("d_model") is None:
                init_kwargs["d_model"] = inferred_d_model

            if (
                init_kwargs.get("num_layers") is None
                and init_kwargs.get("layer_indices") is None
            ):
                init_kwargs["num_layers"] = inferred_num_layers

            instance = cls(**init_kwargs)
        else:
            # Fallback: infer architecture from saved weights
            if inferred_d_model is None or inferred_num_layers == 0:
                raise ValueError(
                    f"Could not infer probe architecture from checkpoint keys: "
                    f"{list(state_dict.keys())}"
                )

            probe_hidden_dims = kwargs.pop("probe_hidden_dims", None)
            if probe_hidden_dims is None:
                probe_hidden_dims = []
                i = 0
                while True:
                    weight_key = f"probe.layer_probes.0.transform.{i * 2}.weight"
                    if weight_key in state_dict:
                        probe_hidden_dims.append(state_dict[weight_key].shape[0])
                        i += 1
                    else:
                        break
                if not probe_hidden_dims:
                    raise ValueError(
                        "Could not infer probe_hidden_dims from checkpoint."
                    )

            instance = cls(
                base_model_name=kwargs.pop("base_model_name", "unknown"),
                d_model=inferred_d_model,
                num_layers=inferred_num_layers,
                probe_hidden_dims=probe_hidden_dims,
                **kwargs,
            )

        # Instantiate the probe before loading weights, since configure_model
        # is not called when just loading the LightningModule for inference.
        probe_num_layers = instance.num_layers if instance.num_layers else inferred_num_layers
        instance.probe = MILMultiLayerProbe(
            d_model=instance.d_model,
            num_layers=probe_num_layers,
            hidden_dims=instance.probe_hidden_dims,
            activation=instance.activation,
        )

        instance.load_state_dict(state_dict, strict=False)
        instance.eval()
        return instance

    @staticmethod
    def attach_hooks(
        model: torch.nn.Module,
        layer_indices: list[int] | None = None,
    ) -> tuple[
        list[str], dict[str, torch.Tensor], list[torch.utils.hooks.RemovableHandle]
    ]:
        """Register forward hooks on transformer layers (same as during training).

        Args:
            model: the base LLM
            layer_indices: which layer indices to hook (None = all)

        Returns:
            layer_names: list of hooked layer names
            captured_states: dict that will be populated on each forward pass
            hooks: list of hook handles (call .remove() to detach)
        """
        all_layer_names = CotProbeModel._get_transformer_layer_names(model)
        if layer_indices is not None:
            layer_names = [all_layer_names[i] for i in layer_indices]
        else:
            layer_names = all_layer_names

        captured_states: dict[str, torch.Tensor] = {}
        hooks = []

        named_modules = dict(model.named_modules())
        for name in layer_names:
            module = named_modules[name]
            hooks.append(
                module.register_forward_hook(CaptureHook(captured_states, name))
            )

        return layer_names, captured_states, hooks

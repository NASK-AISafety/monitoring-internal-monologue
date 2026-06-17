import os
import shutil

import hydra
from lightning.pytorch.plugins.environments import LightningEnvironment
from lightning.pytorch.strategies import FSDPStrategy
from lightning import Trainer
from omegaconf import DictConfig, OmegaConf

from src.cot_probe.data import CotProbeData
from src.cot_probe.trainer import CotProbeModel, LrSchedulerArgs, OptimizerArgs
from src.evaluation.templates import BASE_CHAT_TEMPLATE
from src.utils import load_model_for_pred
from transformers import AutoTokenizer
import functools
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy


@hydra.main(
    config_path="configs/cot_probe_config", config_name="config", version_base=None
)
def main(uninitialized_cfg: DictConfig):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "22690"
    os.environ["NCCL_SOCKET_IFNAME"] = "lo"
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    os.environ["NCCL_IGNORE_DISABLED_IPV6"] = "1"
    cfg = hydra.utils.instantiate(uninitialized_cfg)

    # Load the base model (frozen, used only for hidden state extraction)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.base_model,
        use_fast=False,
        padding_side="left",
        truncation_side="right",
        trust_remote_code=True,
    )
    if (
        tokenizer.chat_template
        and "{% if '</think>' in content %}" in tokenizer.chat_template
    ):
        tokenizer.chat_template = tokenizer.chat_template.replace(
            "{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}",
            "",
        )
    if tokenizer.chat_template is None:
        tokenizer.chat_template = BASE_CHAT_TEMPLATE

    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.pad_token_id = tokenizer.unk_token_id
        else:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

    # Build Lightning module
    lightning_model = CotProbeModel(
        base_model_name=cfg.model.base_model,
        probe_hidden_dims=(
            list(cfg.probe.hidden_dims) if cfg.probe.hidden_dims is not None else None
        ),
        layer_indices=(
            list(cfg.probe.layer_indices)
            if cfg.probe.layer_indices is not None
            else None
        ),
        activation=cfg.probe.activation,
        learning_rate=cfg.trainer.optimizer.lr,
        optimizer_args=OptimizerArgs(**cfg.trainer.optimizer.optimizer_args),
        lr_scheduler_args=(
            None
            if cfg.trainer.optimizer.lr_scheduler_args is None
            else LrSchedulerArgs(**cfg.trainer.optimizer.lr_scheduler_args)
        ),
        pos_weight=cfg.probe.pos_weight,
    )
    # Build data module
    data = CotProbeData(
        tokenizer=tokenizer,
        dataset_path=cfg.data.dataset_path,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        max_length=cfg.data.max_length,
        add_generation_prompt=cfg.data.add_generation_prompt,
        test_size=cfg.data.test_size,
        messages_only=cfg.data.messages_only,
        messages_column=cfg.data.messages_column,
    )

    callbacks = [cfg.trainer.save_callback]
    if cfg.trainer.callbacks is not None:
        callbacks += cfg.trainer.callbacks
    trainer_kwargs = dict(cfg.trainer.trainer_args)

    # Check if FSDP is requested
    if trainer_kwargs.get("strategy") == "fsdp":
        print("Initializing custom FSDP Strategy with auto_wrap_policy...")

        # Tell FSDP to wrap the model into smaller chunks (e.g., > 10M params)
        # This prevents the massive 28GB flattened tensor spike!
        my_auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy, min_num_params=10_000_000
        )

        custom_fsdp = FSDPStrategy(
            sync_module_states=True,
            auto_wrap_policy=my_auto_wrap_policy,
            # use_orig_params is highly recommended when freezing base model weights
            # and only training a probe/adapter on top!
            use_orig_params=True,
        )
        trainer_kwargs["strategy"] = custom_fsdp

    trainer = Trainer(
        callbacks=callbacks,
        logger=cfg.trainer.logger,
        plugins=[LightningEnvironment()],
        **trainer_kwargs,
    )

    if trainer.logger is not None:
        trainer.logger.log_hyperparams(
            OmegaConf.to_container(uninitialized_cfg, resolve=True)
        )

    trainer.fit(lightning_model, datamodule=data)

    # Save the best checkpoint as "best.ckpt" for easy downstream access
    ckpt_callback = None
    for cb in trainer.callbacks:
        if hasattr(cb, "best_model_path"):
            ckpt_callback = cb
            break

    if ckpt_callback and ckpt_callback.best_model_path:
        best_src = ckpt_callback.best_model_path
        best_dst = os.path.join(os.path.dirname(best_src), "best.ckpt")
        shutil.copy2(best_src, best_dst)
        print(f"Saved best checkpoint to {best_dst}")
    else:
        print("Warning: no best checkpoint found to save")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter

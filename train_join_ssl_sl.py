import pickle
import os

old_unpickler = pickle.Unpickler
import lightning
import logging

pickle.Unpickler = old_unpickler

from collections import OrderedDict
import copy

import hydra
import torch
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)

# Flags are absorbed by Hydra.
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch_optimizer import Lamb

from mojo_kirby.data import Dataset, collate
from mojo_kirby.data.sampler import (
    RandomFixedWindowSampler,
    SequentialFixedWindowSampler,
    SequentialSessionSampler,
)
from mojo_kirby.taxonomy import decoder_registry, Decoder
from mojo_kirby.transforms import Compose
from mojo_kirby.utils import train_wrapper
from mojo_kirby.models.po_latent_join_ssl import JoinSSLTokenizer
from lightning.pytorch import seed_everything

wandb_project = os.environ.get("WANDB_PROJECT")

decoder_registry = {
    str(Decoder.CURSORVELOCITY2D): decoder_registry[str(Decoder.CURSORVELOCITY2D)]
}


def run_training(cfg: DictConfig):
    # Fix random seed, skipped if cfg.seed is None
    seed_everything(cfg.seed, workers=True)

    # Higher speed on machines with tensor cores.
    torch.set_float32_matmul_precision("medium")

    log = logging.getLogger(__name__)

    # Device setup is managed by PyTorch Lightning.

    # make model
    model = hydra.utils.instantiate(
        cfg.model,
        task_specs=decoder_registry,
        backend_config=cfg.backend_config,
        bin_width=cfg.bin_width,
        _convert_="object",
    )

    # The transform list is defined in the config file.
    sequence_length = cfg.sequence_length
    transforms = hydra.utils.instantiate(
        cfg.train_transforms, sequence_length=sequence_length
    )

    # build tokenizer
    tokenizer_style = "poyo" if cfg.model.backbone == "transformer" else "possm"
    tokenizer = JoinSSLTokenizer(
        model.unit_emb.tokenizer,
        model.session_emb.tokenizer,
        decoder_registry=decoder_registry,
        latent_step=cfg.bin_width,
        num_latents_per_step=cfg.model.num_latents,
        batch_type=model.batch_type,
        query_max_unit_counts=cfg.get("ssl_query_counts", 10),
        keep_unit_ratio=cfg.get("keep_unit_ratio", True),
        include_decoding_outputs=True,
        bin_width=cfg.bin_width,
        bin_step=cfg.step_size,
        tokenizer_style=tokenizer_style,
    )

    if cfg.get("use_unit_dropout", False):
        transform = Compose([*transforms, tokenizer])
    else:
        transform = Compose([tokenizer])

    log.info("Data root: {}".format(cfg.data_root))
    train_dataset = Dataset(
        cfg.data_root,
        "train",
        include=OmegaConf.to_container(cfg.dataset),
        transform=transform,
    )
    val_tokenizer = copy.copy(tokenizer)
    val_tokenizer.eval = True
    if cfg.model.backbone == "transformer":
        val_tokenizer.eval_step_size = cfg.get("eval_step_size", sequence_length / 2)
    val_dataset = Dataset(
        cfg.data_root,
        "valid",
        include=OmegaConf.to_container(cfg.dataset),
        transform=val_tokenizer,
    )

    test_tokenizer = copy.copy(tokenizer)
    test_tokenizer.eval = True
    if cfg.model.backbone == "transformer":
        test_tokenizer.eval_step_size = cfg.get("eval_step_size", sequence_length / 2)
    test_dataset = Dataset(
        cfg.data_root,
        "test",
        include=OmegaConf.to_container(cfg.dataset),
        transform=test_tokenizer,
    )

    if not cfg.finetune:
        # Register units and sessions
        model.unit_emb.initialize_vocab(train_dataset.unit_ids)
        model.session_emb.initialize_vocab(train_dataset.session_ids)
    else:
        assert cfg.ckpt_path is not None, (
            "Missing `ckpt_path`. Checkpoint is required finetuning."
        )
        ckpt = torch.load(cfg.ckpt_path, map_location="cpu")
        state_dict = ckpt["state_dict"]

        # Remove 'model.' prefix at the front of the state dict keys
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.removeprefix("model.")
            new_state_dict[new_key] = state_dict[key]

        model.load_state_dict(new_state_dict)
        log.info(f"Loaded model state dict from {cfg.ckpt_path}")

        # Optionally freeze parameters for Unit Identification
        if cfg.freeze_perceiver_until_epoch != 0:
            model.freeze_middle()
            log.info(f"Froze perceiver")

        # Register new units and sessions, and delete old ones
        model.unit_emb.extend_vocab(train_dataset.unit_ids, exist_ok=False)
        model.unit_emb.subset_vocab(train_dataset.unit_ids)

        model.session_emb.extend_vocab(train_dataset.session_ids, exist_ok=False)
        model.session_emb.subset_vocab(train_dataset.session_ids)

    # sampler and dataloader
    train_sampler = RandomFixedWindowSampler(
        interval_dict=train_dataset.get_sampling_intervals(),
        window_length=sequence_length,
        generator=None,
    )

    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        collate_fn=collate,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        drop_last=True,
        pin_memory=True,
        # For debugging. we allow the user to set num_workers to 0.
        persistent_workers=True if cfg.num_workers > 0 else False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )

    log.info(f"Training on {len(train_sampler)} samples")
    log.info(f"Training on {len(train_dataset.unit_ids)} units")
    log.info(f"Training on {len(train_dataset.session_ids)} sessions")

    if cfg.model.backbone == "transformer":
        val_sampler = SequentialFixedWindowSampler(
            interval_dict=val_dataset.get_sampling_intervals(),
            window_length=sequence_length,
            step=cfg.get("eval_step_size", sequence_length / 2),
        )
    elif "ssm" in cfg.model.backbone:
        val_sampler = SequentialSessionSampler(
            interval_dict=val_dataset.get_sampling_intervals(),
            min_length=sequence_length,
            drop_short=True,
        )

    eval_batch_size = cfg.get("eval_batch_size", cfg.batch_size)
    eval_num_workers = cfg.get("eval_num_workers", cfg.num_workers)
    val_loader = DataLoader(
        val_dataset,
        sampler=val_sampler,
        collate_fn=collate,
        batch_size=eval_batch_size,
        num_workers=eval_num_workers,
    )

    log.info(f"Validating on {len(val_sampler)} samples")
    log.info(f"Validating on {len(val_dataset.unit_ids)} units")
    log.info(f"Validating on {len(val_dataset.session_ids)} sessions")

    if cfg.model.backbone == "transformer":
        test_sampler = SequentialFixedWindowSampler(
            interval_dict=test_dataset.get_sampling_intervals(),
            window_length=sequence_length,
            step=cfg.get("eval_step_size", sequence_length / 2),
        )
    elif "ssm" in cfg.model.backbone:
        test_sampler = SequentialSessionSampler(
            interval_dict=test_dataset.get_sampling_intervals(),
            min_length=sequence_length,
            drop_short=True,
        )

    test_loader = DataLoader(
        test_dataset,
        sampler=test_sampler,
        collate_fn=collate,
        batch_size=eval_batch_size,
        num_workers=eval_num_workers,
    )

    log.info(f"Testing on {len(test_sampler)} samples")
    log.info(f"Testing on {len(test_dataset.unit_ids)} units")
    log.info(f"Testing on {len(test_dataset.session_ids)} sessions")

    # No need to explicitly use DDP with the model, lightning does this for us.
    max_lr = cfg.base_lr * cfg.batch_size

    if cfg.epochs > 0 and cfg.steps == 0:
        epochs = cfg.epochs
    elif cfg.steps > 0 and cfg.epochs == 0:
        epochs = cfg.steps // len(train_loader) + 1
    else:
        raise ValueError("Must specify either epochs or steps")

    log.info(f"Epochs: {epochs}")

    if cfg.optimizer == "Lamb":
        mul_unit_lr = int(cfg.get("unit_lr_multiplier", 10))
        unit_emb_params = list(model.unit_emb.parameters())
        unit_lr = max_lr * mul_unit_lr
        remaining_params = [
            p for n, p in model.named_parameters() if "unit_emb" not in n
        ]
        optimizer = Lamb(
            [
                {"params": unit_emb_params, "lr": unit_lr},
                {"params": remaining_params},
            ],
            lr=max_lr,
            weight_decay=cfg.weight_decay,
        )
        max_lr_scheduler = [unit_lr, max_lr]
    elif cfg.optimizer == "Adam":
        optimizer = torch.optim.AdamW(
            model.parameters(),  # filter(lambda p: p.requires_grad, model.parameters()),
            lr=max_lr,
            weight_decay=cfg.weight_decay,
        )
        max_lr_scheduler = max_lr

    lr_div_factor = cfg.get("lr_div_factor", 1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr_scheduler,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=cfg.pct_start,
        anneal_strategy="cos",
        div_factor=lr_div_factor,
    )

    wrapper = train_wrapper.TrainWrapper_Join(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    tb = lightning.pytorch.loggers.tensorboard.TensorBoardLogger(
        save_dir=cfg.log_dir,
    )

    wandb = lightning.pytorch.loggers.WandbLogger(
        name=cfg.name,
        project=wandb_project,
        entity=cfg.get("wandb_entity", None),
        log_model=cfg.get("wandb_log_model", False),
        save_dir=cfg.log_dir,
    )

    model_ckpt_callback = ModelCheckpoint(
        dirpath=os.path.join(cfg.log_dir, f"lightning_logs/{wandb.version}"),
        save_top_k=5,
        monitor=cfg.get("ckpt_monitor_metric", "val_loss"),
        mode=cfg.get("ckpt_monitor_mode", "min"),
        save_last=True,
        save_on_train_epoch_end=True,
        every_n_epochs=cfg.eval_epochs,
    )
    callbacks = [
        ModelSummary(max_depth=2),  # Displays the number of parameters in the model.
        model_ckpt_callback,
        train_wrapper.JoinSSLValidator(val_loader),
        train_wrapper.JoinSSLValidator(test_loader, on_test=True),
        LearningRateMonitor(
            logging_interval="step"
        ),  # Create a callback to log the learning rate.
    ]

    if cfg.finetune:
        if cfg.freeze_perceiver_until_epoch > 0:
            callbacks.append(
                train_wrapper.UnfreezeAtEpoch(cfg.freeze_perceiver_until_epoch)
            )

    trainer = lightning.Trainer(
        logger=[tb, wandb],
        default_root_dir=cfg.log_dir,
        check_val_every_n_epoch=cfg.eval_epochs,
        max_epochs=epochs,
        log_every_n_steps=1,
        strategy=(
            "ddp_find_unused_parameters_true" if torch.cuda.is_available() else "auto"
        ),
        callbacks=callbacks,
        num_sanity_val_steps=-1,
        precision=cfg.precision,
        reload_dataloaders_every_n_epochs=50,
        accelerator="gpu",
        devices=cfg.gpus,
        num_nodes=cfg.nodes,
    )

    log.info(
        f"Local rank/node rank/world size/num nodes: {trainer.local_rank}/{trainer.node_rank}/{trainer.world_size}/{trainer.num_nodes}"
    )

    for logger in trainer.loggers:
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    trainer.fit(
        wrapper,
        train_loader,
        [0],
        ckpt_path=cfg.ckpt_path if cfg.load_resumed else None,
    )

    log.info("Beginning Testing")

    if cfg.test_best:
        # Load the best model's parameters
        test_ckpt = model_ckpt_callback.best_model_path
        assert len(test_ckpt) > 0, (
            "No best model has been checkpointed yet. "
            "Probably because the validator has not been run."
        )

        model = load_model_from_ckpt(model, test_ckpt)
        log.info(f"Loaded model state dict from {test_ckpt}")
    else:
        # take the last model
        pass
    trainer.test(wrapper, [0])


def load_model_from_ckpt(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"]

    # Remove 'model.' prefix from the state dict keys
    new_state_dict = {}
    for key in state_dict.keys():
        new_key = key.removeprefix("model.")
        new_state_dict[new_key] = state_dict[key]

    model.load_state_dict(new_state_dict)
    return model


@hydra.main(
    version_base="1.3", config_path="./configs", config_name="train_join_ssl.yaml"
)
def main(cfg: DictConfig):
    run_training(cfg)


if __name__ == "__main__":
    main()

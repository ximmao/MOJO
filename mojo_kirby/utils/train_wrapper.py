import time
import subprocess
import logging
from typing import Optional

import torch
import torch.nn as nn
from lightning import LightningModule
import lightning.pytorch.loggers as pl_loggers
from lightning.pytorch.callbacks import Callback

from mojo_kirby.utils.validation_wrapper import (
    CustomValidator,
    JoinSSLValidator,
)

log = logging.getLogger(__name__)

class UnfreezeAtEpoch(Callback):
    def __init__(self, unfreeze_at_epoch=10):
        super().__init__()
        self._unfreeze_at_epoch = unfreeze_at_epoch

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch == self._unfreeze_at_epoch:
            log.info(f"Reached epoch {trainer.current_epoch}, unfreezing entire model")
            for module in pl_module.model.children():
                for param in module.parameters():
                    param.requires_grad = True


class TrainWrapper_Join(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["optimizer", "scheduler"])

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.tb = None
        self.wandb = None

    def configure_optimizers(self):
        return [self.optimizer], [{"scheduler": self.scheduler, "interval": "step"}]

    def setup(self, stage=None):
        # Make specific loggers available.
        for logger in self.loggers:
            if isinstance(logger, pl_loggers.WandbLogger):
                self.wandb = logger.experiment
            elif isinstance(logger, pl_loggers.TensorBoardLogger):
                self.tb = logger.experiment

    def on_train_start(self):
        # Log the output of `cat /proc/meminfo` using a shell script.
        try:
            # Execute the command and capture its output
            result = subprocess.run(
                ["cat", "/proc/meminfo"],
                capture_output=True,
                text=True,
                check=True,
            )
            result = result.stdout
        except subprocess.CalledProcessError as e:
            print(f"Command failed with error: {e}")
            result = ""

        # Log the output
        logging.info(f"Memory info: \n{result}")

    def on_train_epoch_start(self):
        self.epoch_time = time.time()

    def training_step(self, data, data_idx):
        loss, results_ssl, results_dec = self.model(**data)

        self.log("tot_train_loss", loss, prog_bar=True)

        if results_ssl is not None:
            output_ssl, loss_ssl, taskwise_loss_ssl, taskwise_metric_ssl = results_ssl

            self.log("ssl_train_loss", loss_ssl, prog_bar=True)
            self.log_dict(
                {f"ssl_train_losses/{k}": v for k, v in taskwise_loss_ssl.items()}
            )
            self.log_dict(
                {
                    f"ssl_train_metrics/{k}": v
                    for k, v in taskwise_metric_ssl.items()
                    if (not ("SPIKE_RATE_PERBATCH_R2_ALL_") in k)
                }
            )

        if results_dec is not None:
            output_dec, loss_dec, taskwise_loss_dec = results_dec

            for name in data["output_values"].keys():
                if len(data["output_values"][name]) > 0:
                    continue

            self.log("train_loss", loss_dec, prog_bar=True)
            self.log_dict(
                {f"train_losses/{k}": v for k, v in taskwise_loss_dec.items()}
            )

        return {"loss": loss}

    def on_train_epoch_end(self):
        for tag, value in self.model.named_parameters():
            self.log(f"weights/mean_{tag}", value.cpu().mean(), sync_dist=True)
            self.log(f"weights/std_{tag}", value.cpu().std(), sync_dist=True)
            if value.grad is not None:
                self.log(
                    f"grads/mean_{tag}",
                    value.grad.cpu().mean(),
                    sync_dist=True,
                )

        self.log("epoch_time", time.time() - self.epoch_time)

        self.log("ssl_loss_coef", self.model.ssl_loss_coef)
        self.log("sl_loss_coef", self.model.sl_loss_coef)
        curr_coef = self.model.ssl_loss_coef
        if self.model.loss_coef_scheduler:
            if self.current_epoch >= 100:
                self.model.ssl_loss_coef = 0.1 + (curr_coef - 0.1) * (
                    1
                    + torch.cos(
                        torch.pi * (torch.tensor(self.current_epoch) - 100 + 1) / 400
                    )
                ) / (
                    1
                    + torch.cos(
                        torch.pi * (torch.tensor(self.current_epoch) - 100) / 400
                    )
                )
            else:
                self.model.ssl_loss_coef = curr_coef

    def validation_step(self, data, data_idx):
        # Necessary to trick PyTorch Lightning into running the custom validator.
        pass

    def test_step(self, data, data_idx):
        # Necessary to trick PyTorch Lightning into running the custom validator.
        pass

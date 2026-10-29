import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import torch
import torch.distributed as dist
import wandb
from lightning.pytorch.callbacks import Callback
import logging

from mojo_kirby.data.sampler import DistributedSamplerWrapper
from mojo_kirby.nn import compute_loss_or_metric
from mojo_kirby.taxonomy import Decoder, OutputType, Task
from rich import print as rprint


def move_to_gpu(d, pl_module):
    for k, v in d.items():
        if isinstance(v, dict):
            move_to_gpu(v, pl_module)
        elif isinstance(v, torch.Tensor):
            d[k] = v.to(pl_module.device)
        elif isinstance(v, list):
            if isinstance(v[0], torch.Tensor):
                d[k] = [item.to(pl_module.device) for item in v]


class CustomValidator(Callback):
    def __init__(
        self,
        loader,
        on_test=False,  # True if we are testing, False if we are validating
        prefix=None,  # Prefix text for the metrics
    ):
        super().__init__()
        self.loader = loader

        self.on_test = on_test
        if prefix is None:
            self.prefix = "test" if on_test else "val"
        else:
            self.prefix = prefix

    def run(self, trainer, pl_module):
        session_timestamp = {}
        session_subtask_index = {}
        session_gt_output = {}
        session_pred_output = {}

        if isinstance(self.loader.sampler, DistributedSamplerWrapper):
            self.loader.sampler.set_params(trainer.world_size, trainer.local_rank)

        for batch in tqdm(
            self.loader,
            desc=f"{self.prefix} @ Epoch {trainer.current_epoch}",
            disable=(trainer.local_rank != 0),
        ):
            absolute_starts = batch.pop("absolute_start")  # (B,)
            session_ids = batch.pop("session_id")  # (B,)
            output_subtask_index = batch.pop("output_subtask_index")

            batch_format = None
            if "input_mask" in batch:
                batch_format = "padded"
            elif "input_seqlen" in batch:
                batch_format = "chained"
            else:
                raise ValueError("Invalid batch format.")

            # move to gpu dict of dicts
            move_to_gpu(batch, pl_module)

            def get_autocast_args(trainer):
                if trainer.precision.startswith("bf16"):
                    return torch.bfloat16, True
                elif trainer.precision.startswith("16"):
                    return torch.float16, True
                else:
                    return None, False

            dtype, enabled = get_autocast_args(trainer)
            # forward pass
            with torch.cuda.amp.autocast(enabled=enabled, dtype=dtype):
                with torch.inference_mode():
                    pred_output, loss, losses_taskwise = pl_module.model(**batch)

            # log the val_loss
            pl_module.log_dict({f"{self.prefix}_loss": loss})

            batch_size = len(pred_output)
            timestamps = [{} for _ in range(batch_size)]
            subtask_index = [{} for _ in range(batch_size)]
            gt_output = [{} for _ in range(batch_size)]

            # collect ground truth
            for taskname, spec in pl_module.model.readout.decoder_specs.items():
                taskid = Decoder.from_string(taskname).value

                # get the mask of tokens that belong to this task
                mask = batch["output_decoder_index"] == taskid

                if not torch.any(mask):
                    # there is not a single token for this task, so we skip
                    continue

                if batch_format == "padded":
                    token_batch = torch.where(mask)[0]
                elif batch_format == "chained":
                    token_batch = batch["output_batch_index"][mask]

                batch_i, token_batch = torch.unique(token_batch, return_inverse=True)
                for i in range(len(batch_i)):
                    timestamps[batch_i[i]][taskname] = (
                        batch["output_timestamps"][mask][token_batch == i]
                        + absolute_starts[batch_i[i]]
                    )
                    subtask_index[batch_i[i]][taskname] = output_subtask_index[
                        taskname
                    ][(token_batch == i).detach().cpu()]
                    gt_output[batch_i[i]][taskname] = batch["output_values"][taskname][
                        token_batch == i
                    ]

            # register all of the data
            for i in range(batch_size):
                session_id = session_ids[i]

                if session_id not in session_pred_output:
                    session_pred_output[session_id] = {}
                    session_gt_output[session_id] = {}
                    session_timestamp[session_id] = {}
                    session_subtask_index[session_id] = {}

                for taskname, pred_values in pred_output[i].items():
                    if taskname not in session_pred_output[session_id]:
                        session_pred_output[session_id][taskname] = (
                            pred_values.detach().cpu()
                        )
                        session_gt_output[session_id][taskname] = (
                            gt_output[i][taskname].detach().cpu()
                        )
                        session_timestamp[session_id][taskname] = (
                            timestamps[i][taskname].detach().cpu()
                        )
                        session_subtask_index[session_id][taskname] = (
                            subtask_index[i][taskname].detach().cpu()
                        )
                    else:
                        session_pred_output[session_id][taskname] = torch.cat(
                            (
                                session_pred_output[session_id][taskname],
                                pred_values.detach().cpu(),
                            )
                        )
                        session_gt_output[session_id][taskname] = torch.cat(
                            (
                                session_gt_output[session_id][taskname],
                                gt_output[i][taskname].detach().cpu(),
                            )
                        )
                        session_timestamp[session_id][taskname] = torch.cat(
                            (
                                session_timestamp[session_id][taskname],
                                timestamps[i][taskname].detach().cpu(),
                            )
                        )
                        session_subtask_index[session_id][taskname] = torch.cat(
                            (
                                session_subtask_index[session_id][taskname],
                                subtask_index[i][taskname].detach().cpu(),
                            )
                        )

        def gather_concat_dict(obj):
            """All-gather and concatenate dictionary-of-dictionary-of-tensors objects"""
            gathered_objlist = [None] * trainer.world_size
            dist.all_gather_object(gathered_objlist, obj)

            # Concatenate all tensors in the dictionaries
            gathered_obj = defaultdict(lambda: defaultdict(list))
            for objlist in gathered_objlist:
                for outer_key, inner_dict in objlist.items():
                    for inner_key, tensor in inner_dict.items():
                        gathered_obj[outer_key][inner_key].append(tensor)

            # now actually concatenate the tensors in the innermost lists
            for outer_key, inner_dict in gathered_obj.items():
                for inner_key, tensor_list in inner_dict.items():
                    gathered_obj[outer_key][inner_key] = torch.cat(tensor_list, dim=0)

            dist.barrier()
            return gathered_obj

        # Gather
        if trainer.world_size > 1:
            session_timestamp = gather_concat_dict(session_timestamp)
            session_gt_output = gather_concat_dict(session_gt_output)
            session_pred_output = gather_concat_dict(session_pred_output)
            session_subtask_index = gather_concat_dict(session_subtask_index)

        metrics = dict()
        for session_id in tqdm(
            session_gt_output,
            desc=f"Compiling metrics @ Epoch {trainer.current_epoch}",
            disable=(trainer.local_rank != 0),
        ):
            for taskname in session_gt_output[session_id]:
                decoders = self.loader.dataset.session_info_dict[session_id]["config"][
                    "multitask_readout"
                ]

                decoder = None
                for decoder_ in decoders:
                    if decoder_["decoder_id"] == taskname:
                        decoder = decoder_

                assert decoder is not None, f"Decoder not found for {taskname}"
                metrics_spec = decoder["metrics"]
                for metric in metrics_spec:
                    gt = session_gt_output[session_id][taskname]
                    pred = session_pred_output[session_id][taskname]
                    timestamps = session_timestamp[session_id][taskname]
                    subtask_index = session_subtask_index[session_id][taskname]

                    metric_subtask = metric.get("subtask", None)
                    if metric_subtask is not None:
                        select_subtask_index = Task.from_string(metric_subtask).value
                        mask = subtask_index == select_subtask_index
                        gt = gt[mask]
                        pred = pred[mask]
                        timestamps = timestamps[mask]

                    # pool
                    output_type = pl_module.model.readout.decoder_specs[taskname].type
                    if output_type == OutputType.CONTINUOUS:
                        pred = avg_pool(timestamps, pred)
                        gt = avg_pool(timestamps, gt)
                    elif output_type in [
                        OutputType.BINARY,
                        OutputType.MULTINOMIAL,
                        OutputType.MULTILABEL,
                    ]:
                        gt = gt_pool(timestamps, gt)
                        pred = avg_pool(timestamps, pred)

                    # Resolve the appropriate loss function.
                    metrics[
                        f"{self.prefix}_{session_id}_{str(taskname.lower())}_{metric['metric']}"
                    ] = compute_loss_or_metric(
                        metric["metric"], output_type, pred, gt, 1.0
                    ).item()

        metrics[f"average_{self.prefix}_metric"] = np.array(
            list(metrics.values())
        ).mean()

        pl_module.log_dict(metrics)
        logging.info(f"Logged {len(metrics)} {self.prefix} metrics.")

        metrics_data = []
        for metric_name, metric_value in metrics.items():
            metrics_data.append({"metric": metric_name, "value": metric_value})

        metrics_df = pd.DataFrame(metrics_data)
        if trainer.local_rank == 0:
            if pl_module.tb is not None:
                pl_module.tb.add_text(
                    f"{self.prefix}_metrics", metrics_df.to_markdown()
                )
            if pl_module.wandb is not None:
                pl_module.wandb.log(
                    {f"{self.prefix}_metrics": wandb.Table(dataframe=metrics_df)}
                )

        rprint(metrics_df)

        return metrics_df

    def on_validation_epoch_start(self, trainer, pl_module):
        if not self.on_test:
            return self.run(trainer, pl_module)

    def on_test_epoch_start(self, trainer, pl_module):
        if self.on_test:
            return self.run(trainer, pl_module)

class JoinSSLValidator(CustomValidator):
    def __init__(
        self,
        loader,
        on_test=False,  # True if we are testing, False if we are validating
        prefix=None,  # Prefix text for the metrics
        repeated_time=False,
        cls_repeated_task=None,
    ):
        super().__init__(loader, on_test, prefix)
        self.repeated_time = repeated_time
        assert not repeated_time, "disable repeated time"
        self.cls_repeated_task = cls_repeated_task

    def run(self, trainer, pl_module):
        session_timestamp = {}
        session_subtask_index = {}
        session_gt_output = {}
        session_pred_output = {}
        session_ssl_metric = {}

        if isinstance(self.loader.sampler, DistributedSamplerWrapper):
            self.loader.sampler.set_params(trainer.world_size, trainer.local_rank)

        for batch in tqdm(
            self.loader,
            desc=f"{self.prefix} @ Epoch {trainer.current_epoch}",
            disable=(trainer.local_rank != 0),
        ):
            absolute_starts = batch.pop("absolute_start")  # (B,)
            session_ids = batch.pop("session_id")  # (B,)
            output_subtask_index = batch.pop("output_subtask_index")

            # move to gpu dict of dicts
            move_to_gpu(batch, pl_module)

            def get_autocast_args(trainer):
                if trainer.precision.startswith("bf16"):
                    return torch.bfloat16, True
                elif trainer.precision.startswith("16"):
                    return torch.float16, True
                else:
                    return None, False

            dtype, enabled = get_autocast_args(trainer)
            # forward pass
            with torch.cuda.amp.autocast(enabled=enabled, dtype=dtype):
                with torch.inference_mode():
                    loss, results_ssl, results_dec = pl_module.model(**batch)
                    if results_ssl is not None:
                        output_ssl, loss_ssl, taskwise_loss_ssl, taskwise_metric_ssl = (
                            results_ssl
                        )
                    if results_dec is not None:
                        output_dec, loss_dec, taskwise_loss_dec = results_dec

            # log the val_loss
            pl_module.log_dict({f"tot_{self.prefix}_loss": loss})
            if results_ssl is not None:
                batch_size = len(output_ssl)
                self.log_dict({f"ssl_{self.prefix}_losses": loss_ssl})
                self.log_dict(
                    {
                        f"ssl_{self.prefix}_losses/{k}": v
                        for k, v in taskwise_loss_ssl.items()
                    }
                )
                self.log_dict(
                    {
                        f"ssl_{self.prefix}_metrics/{k}": v
                        for k, v in taskwise_metric_ssl.items()
                        if (not ("SPIKE_RATE_PERBATCH_R2_ALL_") in k)
                    }
                )

            if results_dec is not None:
                batch_size = len(output_dec)
                self.log_dict({f"{self.prefix}_loss": loss_dec})
            else:
                self.log_dict(
                    {
                        f"{self.prefix}_loss": torch.tensor(
                            10.0, device=loss.device, dtype=torch.float32
                        )
                    }
                )

            if results_ssl is not None:
                # session_ssl_metric
                dataset_ids = [ses_id.split("/")[0] for ses_id in session_ids]
                for k, v in taskwise_metric_ssl.items():
                    if ("SPIKE_RATE_PERBATCH_R2_ALL_") in k:
                        for i in range(batch_size):
                            if dataset_ids[i] not in session_ssl_metric:
                                session_ssl_metric[dataset_ids[i]] = {}
                                session_ssl_metric[dataset_ids[i]]["SPIKECOUNT"] = []
                            session_ssl_metric[dataset_ids[i]]["SPIKECOUNT"].append(
                                v[i].detach().cpu()
                            )

            if results_dec is not None:
                # get gt_output and timestamps to be in the same format as pred_output
                timestamps = [{} for _ in range(batch_size)]
                subtask_index = [{} for _ in range(batch_size)]
                gt_output = [{} for _ in range(batch_size)]

                # collect ground truth
                for taskname, spec in pl_module.model.readout.decoder_specs.items():
                    if taskname not in batch["output_values"]:
                        continue

                    mask = (
                        batch["output_decoder_index"]
                        == Decoder.from_string(taskname).value
                    )

                    token_batch = torch.where(mask)[0]

                    batch_i, token_batch = torch.unique(
                        token_batch, return_inverse=True
                    )
                    for i in range(len(batch_i)):
                        batch_output_timestamps = batch["output_timestamps"][mask][
                            token_batch == i
                        ]
                        if self.repeated_time and (taskname in self.cls_repeated_task):
                            batch_output_timestamps = (
                                torch.zeros_like(batch_output_timestamps)
                                + batch_output_timestamps[0]
                            )
                            batch_output_timestamps = batch_output_timestamps.to(
                                batch["output_timestamps"]
                            )
                        timestamps[batch_i[i]][taskname] = (
                            batch_output_timestamps + absolute_starts[batch_i[i]]
                        )

                        subtask_index[batch_i[i]][taskname] = output_subtask_index[
                            taskname
                        ][(token_batch == i).detach().cpu()]
                        gt_output[batch_i[i]][taskname] = batch["output_values"][
                            taskname
                        ][token_batch == i]

                # register all of the data
                for i in range(batch_size):
                    session_id = session_ids[i]

                    if session_id not in session_pred_output:
                        session_pred_output[session_id] = {}
                        session_gt_output[session_id] = {}
                        session_timestamp[session_id] = {}
                        session_subtask_index[session_id] = {}

                    for taskname, pred_values in output_dec[i].items():
                        if taskname not in session_pred_output[session_id]:
                            session_pred_output[session_id][taskname] = (
                                pred_values.detach().cpu()
                            )
                            session_gt_output[session_id][taskname] = (
                                gt_output[i][taskname].detach().cpu()
                            )
                            session_timestamp[session_id][taskname] = (
                                timestamps[i][taskname].detach().cpu()
                            )
                            session_subtask_index[session_id][taskname] = (
                                subtask_index[i][taskname].detach().cpu()
                            )
                        else:
                            session_pred_output[session_id][taskname] = torch.cat(
                                (
                                    session_pred_output[session_id][taskname],
                                    pred_values.detach().cpu(),
                                )
                            )
                            session_gt_output[session_id][taskname] = torch.cat(
                                (
                                    session_gt_output[session_id][taskname],
                                    gt_output[i][taskname].detach().cpu(),
                                )
                            )
                            session_timestamp[session_id][taskname] = torch.cat(
                                (
                                    session_timestamp[session_id][taskname],
                                    timestamps[i][taskname].detach().cpu(),
                                )
                            )
                            session_subtask_index[session_id][taskname] = torch.cat(
                                (
                                    session_subtask_index[session_id][taskname],
                                    subtask_index[i][taskname].detach().cpu(),
                                )
                            )
        if len(session_ssl_metric) > 0:
            for dataset_id in session_ssl_metric:
                session_ssl_metric[dataset_id]["SPIKECOUNT"] = torch.stack(
                    session_ssl_metric[dataset_id]["SPIKECOUNT"]
                )

        def gather_concat_dict(obj):
            """All-gather and concatenate dictionary-of-dictionary-of-tensors objects"""
            gathered_objlist = [None] * trainer.world_size
            dist.all_gather_object(gathered_objlist, obj)

            # Concatenate all tensors in the dictionaries
            gathered_obj = defaultdict(lambda: defaultdict(list))
            for objlist in gathered_objlist:
                for outer_key, inner_dict in objlist.items():
                    for inner_key, tensor in inner_dict.items():
                        gathered_obj[outer_key][inner_key].append(tensor)

            # now actually concatenate the tensors in the innermost lists
            for outer_key, inner_dict in gathered_obj.items():
                for inner_key, tensor_list in inner_dict.items():
                    gathered_obj[outer_key][inner_key] = torch.cat(tensor_list, dim=0)

            dist.barrier()
            return gathered_obj

        # Gather
        if trainer.world_size > 1:
            if len(session_ssl_metric) > 0:
                session_ssl_metric = gather_concat_dict(session_ssl_metric)
            if len(session_pred_output) > 0:
                session_timestamp = gather_concat_dict(session_timestamp)
                session_gt_output = gather_concat_dict(session_gt_output)
                session_pred_output = gather_concat_dict(session_pred_output)
                session_subtask_index = gather_concat_dict(session_subtask_index)

        if len(session_ssl_metric) > 0:
            for dataset_id in session_ssl_metric:
                for taskname in session_ssl_metric[dataset_id]:
                    session_ssl_metric[dataset_id][taskname] = torch.mean(
                        session_ssl_metric[dataset_id][taskname]
                    ).item()
            self.log_dict(
                {
                    f"ssl_{self.prefix}_metrics/{k}": v["SPIKECOUNT"]
                    for k, v in session_ssl_metric.items()
                }
            )

        if len(session_pred_output) > 0:
            metrics = dict()
            for session_id in tqdm(
                session_gt_output,
                desc=f"Compiling metrics @ Epoch {trainer.current_epoch}",
                disable=(trainer.local_rank != 0),
            ):
                for taskname in session_gt_output[session_id]:
                    decoders = self.loader.dataset.session_info_dict[session_id][
                        "config"
                    ]["multitask_readout"]

                    decoder = None
                    for decoder_ in decoders:
                        if decoder_["decoder_id"] == taskname:
                            decoder = decoder_

                    assert decoder is not None, f"Decoder not found for {taskname}"
                    metrics_spec = decoder["metrics"]
                    for metric in metrics_spec:
                        gt = session_gt_output[session_id][taskname]
                        pred = session_pred_output[session_id][taskname]
                        timestamps = session_timestamp[session_id][taskname]
                        subtask_index = session_subtask_index[session_id][taskname]

                        metric_subtask = metric.get("subtask", None)
                        if metric_subtask is not None:
                            select_subtask_index = Task.from_string(
                                metric_subtask
                            ).value
                            mask = subtask_index == select_subtask_index
                            gt = gt[mask]
                            pred = pred[mask]
                            timestamps = timestamps[mask]

                        # pool
                        output_type = pl_module.model.readout.decoder_specs[
                            taskname
                        ].type
                        if output_type == OutputType.CONTINUOUS:
                            pred = avg_pool(timestamps, pred)
                            gt = avg_pool(timestamps, gt)
                        elif output_type in [
                            OutputType.BINARY,
                            OutputType.MULTINOMIAL,
                            OutputType.MULTILABEL,
                        ]:
                            gt = gt_pool(timestamps, gt)
                            pred = avg_pool(timestamps, pred)

                        # Resolve the appropriate loss function.
                        metrics[
                            f"{self.prefix}_{session_id}_{str(taskname.lower())}_{metric['metric']}"
                        ] = compute_loss_or_metric(
                            metric["metric"], output_type, pred, gt, 1.0
                        ).item()

            metrics[f"average_{self.prefix}_metric"] = np.array(
                list(metrics.values())
            ).mean()

            pl_module.log_dict(metrics)
            logging.info(f"Logged {len(metrics)} {self.prefix} metrics.")

            metrics_data = []
            for metric_name, metric_value in metrics.items():
                metrics_data.append({"metric": metric_name, "value": metric_value})

            metrics_df = pd.DataFrame(metrics_data)
            if trainer.local_rank == 0:
                if pl_module.tb is not None:
                    pl_module.tb.add_text(
                        f"{self.prefix}_metrics", metrics_df.to_markdown()
                    )
                if pl_module.wandb is not None:
                    pl_module.wandb.log(
                        {f"{self.prefix}_metrics": wandb.Table(dataframe=metrics_df)}
                    )

            rprint(metrics_df)

        return 0

def avg_pool(timestamps: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    # Find unique timestamps and their inverse indices
    unique_timestamps, indices = torch.unique(
        timestamps, return_inverse=True, sorted=True
    )

    # Prepare a tensor for summing values for each unique timestamp
    pooled_sum = torch.zeros(
        (len(unique_timestamps), *values.shape[1:]),
        device=values.device,
        dtype=values.dtype,
    )

    # Use mode for integers
    if values.dtype == torch.long:
        # NOT IDEAL, IT IS FASTER TO AVERAGE THE LOGITS THAN TO PERFORM A VOTE
        mode_values = torch.zeros_like(pooled_sum)
        for i, timestamp in enumerate(unique_timestamps):
            mask = timestamps == timestamp
            group_values = values[mask]
            mode, _ = torch.mode(group_values, dim=0)
            mode_values[i] = mode
        return mode_values

    # Count occurrences of each unique timestamp
    counts = torch.zeros(
        len(unique_timestamps), device=timestamps.device, dtype=values.dtype
    )
    counts = counts.scatter_add_(
        0, indices, torch.ones_like(indices, dtype=values.dtype)
    )
    # Accumulate values for each unique timestamp
    indices_expanded = indices.unsqueeze(-1).expand_as(values)
    pooled_sum.scatter_add_(0, indices_expanded, values)
    # Calculate the average
    epsilon = 1e-8
    averages = torch.div(pooled_sum, counts.unsqueeze(-1) + epsilon)

    return averages


def gt_pool(timestamps: torch.Tensor, values: torch.Tensor) -> torch.Tensor:

    return (
        torch.round(avg_pool(timestamps, values.float().view(-1, 1))).long().squeeze()
    )


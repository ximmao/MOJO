import torch
import torch.nn.functional as F

from torchmetrics import R2Score, Accuracy

from mojo_kirby.taxonomy import OutputType


def compute_loss_or_metric(
    loss_or_metric: str,
    output_type: OutputType,
    output: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    if loss_or_metric in ["mse", "bce", "mallows_distance", "pnll"]:
        if weights.sum() == 0.0:
            return torch.tensor(0, device=output.device, dtype=torch.float32)

    if output_type == OutputType.CONTINUOUS:
        if loss_or_metric == "mse":
            # MSE loss
            loss_noreduce = F.mse_loss(output, target, reduction="none").mean(dim=1)
            return (weights * loss_noreduce).sum() / weights.sum()
        elif loss_or_metric == "r2":
            r2score = R2Score(num_outputs=target.shape[1])
            return r2score(output, target)
        elif loss_or_metric == "frame_diff_acc":
            normalized_window = 30 / 450
            differences = torch.abs(output - target)
            correct_predictions = differences <= normalized_window
            accuracy = (
                correct_predictions.float().mean()
            )  # Convert boolean tensor to float and calculate mean
            return accuracy
        else:
            raise NotImplementedError(
                f"Loss/Metric {loss_or_metric} not implemented for continuous output"
            )

    if output_type in [
        OutputType.BINARY,
        OutputType.MULTINOMIAL,
        OutputType.MULTILABEL,
    ]:
        if loss_or_metric == "bce":
            target = target.to(torch.long)
            if not (len(target.size()) == 1 and target.size(0) == 1):
                target = target.squeeze()
            loss_noreduce = F.cross_entropy(output, target, reduction="none")
            if loss_noreduce.ndim > 1:
                loss_noreduce = loss_noreduce.mean(dim=1)
            return (weights * loss_noreduce).sum() / weights.sum()
        elif loss_or_metric == "mallows_distance":
            num_classes = output.size(-1)
            output = torch.softmax(output, dim=-1).view(-1, num_classes)
            target = target.view(-1, 1)
            weights = weights.view(-1)
            # Mallow distance
            target = torch.zeros_like(output).scatter_(1, target, 1.0)
            # we compute the mallow distance as the sum of the squared differences
            loss = torch.mean(
                torch.square(
                    torch.cumsum(target, dim=-1) - torch.cumsum(output, dim=-1)
                ),
                dim=-1,
            )
            loss = (weights * loss).sum() / weights.sum()
            return loss
        elif loss_or_metric == "accuracy":
            pred_class = torch.argmax(output, dim=1)
            return (pred_class == target.squeeze()).sum() / len(target)
        elif loss_or_metric == "balanced_acc":
            num_classes = output.size(-1)
            pred_class = torch.argmax(output, dim=1)
            balaccuracy = Accuracy(
                task="multiclass", average="macro", num_classes=num_classes
            )
            return balaccuracy(pred_class, target.squeeze())
        elif loss_or_metric == "frame_diff_acc":
            pred_class = torch.argmax(output, dim=1)
            difference = torch.abs(pred_class - target.squeeze())
            correct_predictions = difference <= 30
            return correct_predictions.float().mean()
        else:
            raise NotImplementedError(
                f"Loss/Metric {loss_or_metric} not implemented for binary/multilabel "
                "output"
            )

    if output_type == OutputType.POISSON:
        raise NotImplementedError(
            f"Loss/Metric {loss_or_metric} not implemented for poisson output"
        )

    raise NotImplementedError(
        "I don't know how to handle this task type. Implement plis"
    )

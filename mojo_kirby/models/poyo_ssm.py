import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm.models.mixer_seq_simple import create_block
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
    print("Failed to import Triton LayerNorm / RMSNorm kernels")

try:
    from xlstm import (
        xLSTMBlockStack,
        xLSTMBlockStackConfig,
        mLSTMBlockConfig,
        mLSTMLayerConfig,
        sLSTMBlockConfig,
        sLSTMLayerConfig,
        FeedForwardConfig,
    )
except ImportError:
    xLSTMBlockStackConfig = None
    print("Failed to import xlstm")

# Dropout broke in PyTorch 1.11
if tuple(map(int, torch.__version__.split(".")[:2])) == (1, 11):
    print("WARNING: Dropout is bugged in PyTorch 1.11. Results may be worse.")
    dropout_fn = nn.Dropout
if tuple(map(int, torch.__version__.split(".")[:2])) >= (1, 12):
    dropout_fn = nn.Dropout1d
else:
    dropout_fn = nn.Dropout2d

BACKEND_CONFIGS = {
    "cpu": (("stacked", "stacked", "stacked"), ("math", "math", "math")),
    "gpu_fp32": (
        ("stacked", "stacked", "stacked"),
        ("math", "math", "math"),
    ),
}

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class GRU(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        bias=True,
        batch_first=True,
        dropout=0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.model_type = "gru"

        self.net = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
        )

    def forward(self, x):
        out, _ = self.net(x)

        return out


class MambaNet(nn.Module):
    def __init__(
        self,
        model_dim,
        nlayers,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        fused_add_norm=False,
        residual_in_fp32=False,
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    model_dim,
                    0,  # 0 -> no MLP, i.e., Identity; previous value: model_dim
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                )
                for i in range(nlayers)
            ]
        )
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            model_dim, eps=norm_epsilon
        )

    def forward(self, hidden_states, inference_params=None):
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
        if not self.fused_add_norm:
            residual = (
                (hidden_states + residual) if residual is not None else hidden_states
            )
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = (
                rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            )
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        return hidden_states


class MambaModel(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        model_dim=128,
        nlayers=1,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        fused_add_norm=False,
        residual_in_fp32=False,
    ) -> None:
        super().__init__()
        self.model_type = "mamba"
        self.embed = nn.Linear(input_dim, model_dim)
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.mamba = MambaNet(
            model_dim=model_dim,
            nlayers=nlayers,
            ssm_cfg=ssm_cfg,
            norm_epsilon=norm_epsilon,
            rms_norm=rms_norm,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
        )

    def forward(self, x):
        hidden_states = self.embed(x)
        hidden_states = self.mamba(hidden_states)
        return hidden_states


class XLSTMModel(nn.Module):
    def __init__(
        self,
        input_dim,
        dim,
        nlayers=1,
        dropout=0.0,  # by default 0.
        include_slstm=False,
    ):
        super().__init__()

        slstm_at = [min(nlayers - 1, 1)] if include_slstm else []
        len_slstm = len(slstm_at)
        model_cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=4,
                    qkv_proj_blocksize=4,
                    num_heads=4,
                )
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend="cuda" if torch.cuda.is_available() else "vanilla",
                    num_heads=4,
                    conv1d_kernel_size=4,
                    bias_init="powerlaw_blockdependent",
                ),
                feedforward=FeedForwardConfig(
                    proj_factor=1.3,
                    act_fn="gelu",
                ),
            ),
            context_length=512,  # set a maximal value
            num_blocks=nlayers + len_slstm,
            embedding_dim=dim,
            slstm_at=slstm_at,
            dropout=dropout,
        )

        self.xlstm = xLSTMBlockStack(model_cfg)
        self.embed = nn.Linear(input_dim, dim)

    def forward(self, x):
        hidden_states = self.embed(x)
        hidden_states = self.xlstm(
            hidden_states
        )  # outputs has same dimension as inputs

        return hidden_states

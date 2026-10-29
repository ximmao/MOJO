from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType
from torchmetrics import R2Score
from einops import rearrange, repeat

from mojo_kirby.taxonomy import (
    DecoderSpec,
    Decoder,
    StringIntEnum,
)
from mojo_kirby.nn import (
    Embedding,
    InfiniteVocabEmbedding,
    MultitaskReadout,
    POSSMMultitaskReadout,
    prepare_for_multitask_readout,
    RotaryEmbedding,
    RotaryCrossAttention,
    RotarySelfAttention,
)
from mojo_kirby.nn.perceiver_rotary import FeedForward
from mojo_kirby.data import pad, pad_sequence, chain, track_mask, track_batch
from mojo_kirby.utils import (
    create_linspace_latent_tokens,
    TokenType,
)

class MaskingType(StringIntEnum):
    CAUSAL_MASKING = 45
    TEMPORAL_MASKING = 48


from .poyo_ssm import MambaModel, GRU, XLSTMModel

BACKEND_CONFIGS = {
    "cpu": (("stacked", "stacked", "stacked"), ("math", "math", "math")),
    "gpu_fp32": (
        ("stacked", "stacked", "stacked"),
        ("math", "math", "math"),
    ),
}


class BiGRU(nn.Module):
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

        self.model_type = "bigru"

        self.net = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=True,
        )

    def forward(self, x):
        out, _ = self.net(x)

        return out


class PerceiverRotaryEncoder(nn.Module):
    def __init__(
        self,
        *,
        dim=512,
        context_dim=None,
        dim_head=64,
        depth=0,
        cross_heads=1,
        self_heads=8,
        ffn_dropout=0.2,
        lin_dropout=0.4,
        atn_dropout=0.0,
        batch_type: Union[str, Tuple[str, str, str]] = "stacked",
        backend: Union[str, Tuple[str, str, str]] = "math",
        t_min=1e-4,
        t_max=4.0,
        output_latent_depth=0,
        num_latents=64,
    ):
        super().__init__()

        self.rotary_emb = RotaryEmbedding(dim_head, t_min, t_max)

        self.dropout = nn.Dropout(p=lin_dropout)

        # Encoding transformer (q-latent, kv-input spikes)
        self.enc_atn = RotaryCrossAttention(
            dim=dim,
            context_dim=context_dim,
            heads=cross_heads,
            dropout=atn_dropout,
            dim_head=dim_head,
            rotate_value=True,
            batch_type=batch_type,
            backend=backend,
        )
        self.enc_ffn = nn.Sequential(
            nn.LayerNorm(dim), FeedForward(dim=dim, dropout=ffn_dropout)
        )

        # Processing transfomers (qkv-latent)
        self.proc_layers = nn.ModuleList([])
        self.depth = depth

        for i in range(depth):
            self.proc_layers.append(
                nn.ModuleList(
                    [
                        RotarySelfAttention(
                            dim=dim,
                            heads=self_heads,
                            dropout=atn_dropout,
                            dim_head=dim_head,
                            rotate_value=True,
                            batch_type=batch_type,
                            backend=backend,
                        ),
                        nn.Sequential(
                            nn.LayerNorm(dim),
                            FeedForward(dim=dim, dropout=ffn_dropout),
                        ),
                    ]
                )
            )

        # self.dim = dim
        self.output_latent_depth = output_latent_depth
        assert output_latent_depth <= depth
        self.num_latents = num_latents

    def forward(
        self,
        *,
        inputs,  # (B, N_in, dim)
        latents,  # (B, N_latent, dim)
        input_timestamps,  # (B, N_in)
        latent_timestamps,  # (B, N_latent)
        input_mask=None,  # (B, N_in) or None
        sa_depthwise_att=True,
        sa_depthwise_att_bin=True,
    ) -> TensorType["batch", "*nqueries", "dim"]:

        # compute timestamp embeddings
        input_timestamp_emb = self.rotary_emb(input_timestamps)
        latent_timestamp_emb = self.rotary_emb(latent_timestamps)

        # encode
        latents = latents + self.enc_atn(
            latents,
            inputs,
            latent_timestamp_emb,
            input_timestamp_emb,
            context_mask=input_mask,  # used if default attention
        )
        latents = latents + self.enc_ffn(latents)

        # process
        if self.depth > 0 and sa_depthwise_att:
            B, NL, DL = latents.size()
            latent_steps = int(NL // self.num_latents)
            DIL = latent_timestamp_emb.size(-1)

            if sa_depthwise_att_bin:
                latents = latents.view(B, latent_steps, self.num_latents, DL)
                latent_timestamp_emb = latent_timestamp_emb.view(
                    B, latent_steps, self.num_latents, DIL
                )
                latents = latents.view(-1, self.num_latents, DL)
                latent_timestamp_emb = latent_timestamp_emb.view(
                    -1, self.num_latents, DIL
                )
            else:
                raise ValueError()

        if self.depth > 0:
            sa_cnt = 0
            for self_attn, self_ff in self.proc_layers:
                latents = latents + self.dropout(
                    self_attn(latents, latent_timestamp_emb)
                )
                latents = latents + self.dropout(self_ff(latents))
                sa_cnt += 1

            assert self.output_latent_depth == sa_cnt

        if self.depth > 0 and sa_depthwise_att:
            if sa_depthwise_att_bin:
                latents = latents.view(B, latent_steps, self.num_latents, DL)
                latents = latents.view(B, NL, DL)
            else:
                raise NotImplementedError()

        return latents


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        *,
        dim=512,
        dim_head=64,
        num_latents=64,
        depth=2,
        self_heads=8,
        ffn_dropout=0.2,
        lin_dropout=0.4,
        atn_dropout=0.0,
        emb_init_scale=0.02,
        batch_type: Union[str, Tuple[str, str, str]] = "stacked",
        backend: Union[str, Tuple[str, str, str]] = "math",
        t_min=1e-4,
        t_max=4.0,
        enable_masked_pretrain=True,
    ):
        super().__init__()

        self.rotary_emb = RotaryEmbedding(dim_head, t_min, t_max)
        self.position_emb = Embedding(num_latents, dim, init_scale=emb_init_scale)
        self.enable_masked_pretrain = enable_masked_pretrain
        if enable_masked_pretrain:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
            torch.nn.init.normal_(self.mask_token, mean=0, std=emb_init_scale)

        self.dropout = nn.Dropout(p=lin_dropout)

        self.proc_layers = nn.ModuleList([])
        for i in range(depth):
            self.proc_layers.append(
                nn.ModuleList(
                    [
                        RotarySelfAttention(
                            dim=dim,
                            heads=self_heads,
                            dropout=atn_dropout,
                            dim_head=dim_head,
                            rotate_value=True,
                            batch_type=batch_type,
                            backend=backend,
                        ),
                        nn.Sequential(
                            nn.LayerNorm(dim),
                            FeedForward(dim=dim, dropout=ffn_dropout),
                        ),
                    ]
                )
            )
        self.backend = backend

    def forward(
        self,
        *,
        unmasked_inputs,  # (B, N_latent, dim)
        task_mask,  # (B, N_latent) / target mask for the task <=> mask indicating masked token
        input_mask,  # (B, N_latent, N_latent) / mask for attention (potentially causal mask)
        input_timestamps,  # (B, N_latent)
        input_index,  # (B, N_latent)
        task_name,
    ):

        input_timestamp_emb = self.rotary_emb(input_timestamps)
        input_position_emb = self.position_emb(input_index)

        # replace the masked visual tokens by mask_token
        if task_name == "NoMask":
            input = unmasked_inputs
        else:
            task_mask = task_mask.unsqueeze(2).type_as(self.mask_token)
            input = unmasked_inputs * (1 - task_mask) + self.mask_token * task_mask
        latents = input + input_position_emb

        # process
        for self_attn, self_ff in self.proc_layers:
            latents = latents + self.dropout(
                self_attn(latents, input_timestamp_emb, x_mask=input_mask)
            )
            latents = latents + self.dropout(self_ff(latents))

        return latents


class PerceiverRotaryDecoder(nn.Module):
    def __init__(
        self,
        *,
        dim=512,
        context_dim=None,
        dim_head=64,
        depth=2,
        cross_heads=1,
        self_heads=8,
        ffn_dropout=0.2,
        lin_dropout=0.4,
        atn_dropout=0.0,
        batch_type: Union[str, Tuple[str, str, str]] = "stacked",
        backend: Union[str, Tuple[str, str, str]] = "math",
        t_min=1e-4,
        t_max=4.0,
    ):
        super().__init__()

        self.rotary_emb = RotaryEmbedding(dim_head, t_min, t_max)

        self.dropout = nn.Dropout(p=lin_dropout)

        self.depth = depth
        self.proc_layers = nn.ModuleList([])
        for i in range(depth):
            self.proc_layers.append(
                nn.ModuleList(
                    [
                        RotarySelfAttention(
                            dim=dim,
                            heads=self_heads,
                            dropout=atn_dropout,
                            dim_head=dim_head,
                            rotate_value=True,
                            batch_type=batch_type,
                            backend=backend,
                        ),
                        nn.Sequential(
                            nn.LayerNorm(dim),
                            FeedForward(dim=dim, dropout=ffn_dropout),
                        ),
                    ]
                )
            )

        self.dec_atn = RotaryCrossAttention(
            dim=context_dim,  # qo_dim
            context_dim=dim,  # kv_dim
            heads=cross_heads,
            dropout=atn_dropout,
            dim_head=dim_head,
            rotate_value=False,
            batch_type=batch_type,
            backend=backend,
        )
        self.dec_ffn = nn.Sequential(
            nn.LayerNorm(context_dim),
            FeedForward(dim=context_dim, dropout=ffn_dropout),
        )

        self.dim = dim

    def forward(
        self,
        *,
        latents,  # (B, N_latent, dim)
        output_queries,  # (B, N_out, dim)
        latent_timestamps,  # (B, N_latent)
        output_query_timestamps,  # (B, N_out)
        latent_mask=None,  # (B, N_latent) or None
    ) -> TensorType["batch", "*nqueries", "dim"]:

        # compute timestamp embeddings
        latent_timestamp_emb = self.rotary_emb(latent_timestamps)
        output_timestamp_emb = self.rotary_emb(output_query_timestamps)

        if self.depth > 0:
            for self_attn, self_ff in self.proc_layers:
                latents = latents + self.dropout(
                    self_attn(latents, latent_timestamp_emb)
                )
                latents = latents + self.dropout(self_ff(latents))

        # decode
        output_queries = output_queries + self.dec_atn(
            output_queries,
            latents,
            output_timestamp_emb,
            latent_timestamp_emb,
            context_mask=latent_mask,
        )
        output_queries = output_queries + self.dec_ffn(output_queries)

        return output_queries


class StateSpaceWrapper(nn.Module):
    """
    SSM wrapper to faciliate masking, take either GRU / Mamba / S4 as member
    """

    def __init__(
        self,
        *,
        recurrent_backbone="mamba",
        dim=128,
        num_latents=2,
        hidden_dim=256,
        num_layers=6,
        dropout=0.4,
        enable_masked_pretrain=True,
        emb_init_scale=0.02,
        # for swa
        self_heads=4,
        dim_head=64,
        atn_dropout=0.4,
        ffn_dropout=0.2,
        bin_width=0.02,
        sliding_window_size=20,
        rotary_t_max=10.0,
        batch_type="stacked",
        backend="math",
    ):
        super().__init__()

        self.enable_masked_pretrain = enable_masked_pretrain
        if enable_masked_pretrain:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, num_latents * dim))
            torch.nn.init.normal_(self.mask_token, mean=0, std=emb_init_scale)

        if recurrent_backbone == "mamba":
            self.rnn_module = MambaModel(
                input_dim=num_latents * dim,
                output_dim=2,
                model_dim=hidden_dim,
                nlayers=num_layers,
            )
        elif recurrent_backbone == "mamba2":
            self.rnn_module = MambaModel(
                input_dim=num_latents * dim,
                output_dim=2,
                model_dim=hidden_dim,
                nlayers=num_layers,
                ssm_cfg={"layer": "Mamba2"},
            )
        elif recurrent_backbone == "gru":
            self.rnn_module = GRU(
                input_size=num_latents * dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
            )
        elif recurrent_backbone == "bigru":
            self.rnn_module = BiGRU(
                input_size=num_latents * dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
            )
        elif recurrent_backbone == "xlstm":
            self.rnn_module = XLSTMModel(
                input_dim=num_latents * dim,
                dim=hidden_dim,
                nlayers=num_layers,
                dropout=0.0,
            )
        elif recurrent_backbone == "xlstm-mixed":
            self.rnn_module = XLSTMModel(
                input_dim=num_latents * dim,
                dim=hidden_dim,
                nlayers=num_layers,
                dropout=0.0,
                include_slstm=True,
            )
        else:
            raise NotImplementedError()
        self.backbone_name = recurrent_backbone

        self.num_latents = num_latents
        self.dim = dim

    def forward(
        self,
        *,
        unmasked_inputs,  # (B, N_latent, dim)
        task_mask=None,  # (B, T) / target mask for the task <=> mask indicating masked token
        task_name="NoMask",
        latent_timestamps=None,
        num_intervals=None,
    ):

        # RNN works with concated latent
        concat_inputs = unmasked_inputs.view(
            unmasked_inputs.size(0),
            int(unmasked_inputs.size(1) // self.num_latents),
            self.num_latents * self.dim,
        )

        # replace the masked visual tokens by mask_token
        if task_name == "NoMask":
            ssm_inputs = concat_inputs
        else:
            # for SSL, we force seq len as 1
            task_mask = task_mask.unsqueeze(2).type_as(self.mask_token)
            ssm_inputs = concat_inputs * (1 - task_mask) + self.mask_token * task_mask

        if self.backbone_name == "swa":
            latents = self.rnn_module(
                ssm_inputs,
                latent_timestamps=latent_timestamps,
                num_intervals=num_intervals,
            )
        else:
            latents = self.rnn_module(ssm_inputs)  # B x Latent_Steps x concated_dim

        return latents


class POPlusLatentJoinSSL(nn.Module):
    """
    SSL model for learning the poyo latents, original poyo model up until the predicted latents are kept.
    """

    def __init__(
        self,
        *,
        dim=128,
        dim_input=None,
        dim_head=64,
        num_latents=8,
        enc_depth=2,
        ae_depth=6,
        output_ssl_depth=0,
        output_dec_depth=0,
        cross_heads=4,
        dec_cross_heads=None,
        self_heads=8,
        dec_self_heads=None,
        ffn_dropout=0.2,
        lin_dropout=0.4,
        atn_dropout=0.2,
        emb_init_scale=0.02,
        unit_emb_init_scale=None,
        bin_width=0.02,
        rotary_t_max=4.0,
        use_value_emb=True,
        patch_size=5,
        backbone="transformer",
        dim_ssm=256,
        backend_config="gpu_fp32",
        masking_loss="none",  # placeholder
        decoding_head="join",
        loss_coef_scheduler=False,
        ssl_loss_coef_init=0.2,
        sl_loss_coef_init=1.0,
        decoding_head_model="ca",
        qry_cross_att_per_bin=True,
        enable_temporal_mask=True,  # placeholder
        temporal_mask_mode="random",
        mask_ratio=0.5,
        output_ca_restrict_context=True,
        output_ca_ctx_lim=3,
        separate_ssm_enc_path=False,
        separate_atn_enc_path=True,
        poyo_special_token=False,  # start / end vs 2 existence
        detach_sl_head=False,
        task_specs: Dict[str, DecoderSpec],
    ):
        super().__init__()

        dim_input = dim_input or dim
        dec_cross_heads = dec_cross_heads or cross_heads
        dec_self_heads = dec_self_heads or self_heads
        unit_emb_init_scale = unit_emb_init_scale or emb_init_scale
        self.unit_emb = InfiniteVocabEmbedding(
            dim_input, init_scale=unit_emb_init_scale
        )
        self.session_emb = InfiniteVocabEmbedding(dim, init_scale=emb_init_scale)
        self.spike_type_emb = Embedding(
            TokenType.max_value() + 1, dim_input, init_scale=emb_init_scale
        )
        self.latent_emb = Embedding(num_latents, dim, init_scale=emb_init_scale)
        self.num_latents = num_latents
        self.use_value_emb = use_value_emb
        context_dim_factor = 1 + sum([self.use_value_emb])
        context_dim = int(dim_input * context_dim_factor)
        if use_value_emb:
            self.value_emb = nn.Linear(patch_size, dim_input, bias=False)
        qry_ca_context_dim = int(dim_input)

        # needed to handle discrepancy in terms of bin intervals between spikes and latents
        self.spike_latent_bin_ratio = 1.0
        self.spike_latent_bin_tolerance = 0.1  # 0.1 as a small value, can be anything

        self.qry_cross_att_per_bin = qry_cross_att_per_bin
        if self.qry_cross_att_per_bin:
            assert output_ssl_depth == 0
        self.enable_temporal_mask = enable_temporal_mask
        assert temporal_mask_mode in ["random", "forward"]
        self.temporal_mask_mode = temporal_mask_mode
        # span temporal masking not used
        self.span_temporal_masking = False
        self.num_time_spans = 1
        self.mask_ratio = mask_ratio
        self.masking_loss = masking_loss  # loss function related with the masked tokens
        assert masking_loss == "none"
        self.decoding_head = decoding_head
        assert decoding_head in ["none", "ssl", "sl", "join"], decoding_head
        self.decoding_head_model = decoding_head_model
        assert decoding_head_model in ["ca"], decoding_head_model

        # determine backend
        assert backend_config == "gpu_fp32", backend_config
        self.batch_type = BACKEND_CONFIGS[backend_config][0]
        attention_backend = ("math", "math", "math")
        self.attention_backend = attention_backend

        self.loss_coef_scheduler = loss_coef_scheduler
        self.ssl_loss_coef = ssl_loss_coef_init
        self.sl_loss_coef = sl_loss_coef_init
        self.separate_ssm_enc_path = separate_ssm_enc_path
        self.separate_atn_enc_path = separate_atn_enc_path
        self.poyo_special_token = poyo_special_token
        self.detach_sl_head = detach_sl_head

        # self.input_ca_w_list = input_ca_w_list
        self.perceiver_encoder = PerceiverRotaryEncoder(
            dim=dim,
            context_dim=context_dim,
            dim_head=dim_head,
            depth=enc_depth,
            cross_heads=cross_heads,
            self_heads=self_heads,
            ffn_dropout=ffn_dropout,
            lin_dropout=lin_dropout,
            atn_dropout=atn_dropout,
            batch_type="stacked",
            backend=attention_backend[0],  #'math',
            output_latent_depth=enc_depth,  # taken output of first hidden layer (output of first CA module)
            num_latents=num_latents,
            t_max=rotary_t_max,
        )

        self.post_encoder_projector = False

        self.backbone = backbone
        assert self.backbone in [
            "transformer",
            "ssm_mamba",
            "ssm_gru",
            "ssm_s4",
            "ssm_swa",
            "ssm_xlstm",
            "ssm_mamba2",
            "ssm_xlstm-mixed",
            "ssm_bigru",
        ]
        if backbone == "transformer":
            # self.poyo_ssl_latent_steps = 50
            self.mask_autoencoder = TransformerEncoder(
                dim=dim,
                dim_head=dim_head,
                num_latents=num_latents,
                depth=ae_depth,
                self_heads=self_heads,
                ffn_dropout=ffn_dropout,
                lin_dropout=lin_dropout,
                atn_dropout=atn_dropout,
                emb_init_scale=emb_init_scale,
                batch_type="stacked",
                backend=attention_backend[1],  #'math',
                enable_masked_pretrain=(
                    (decoding_head == "ssl") or (decoding_head == "join")
                ),
                t_max=rotary_t_max,
            )
        else:
            assert "ssm" in backbone
            # variable latent_steps due to SequentialSessionSampler
            ssm_dropout = atn_dropout
            self.dim_ssm = dim_ssm
            self.mask_autoencoder = StateSpaceWrapper(
                recurrent_backbone=backbone.split("_")[-1],
                dim=dim,
                num_latents=num_latents,
                hidden_dim=dim_ssm,
                num_layers=ae_depth,
                dropout=ssm_dropout,
                enable_masked_pretrain=(
                    (decoding_head == "ssl") or (decoding_head == "join")
                ),
                self_heads=self_heads,
                dim_head=dim_head,
                sliding_window_size=10,
                atn_dropout=atn_dropout,
                ffn_dropout=ffn_dropout,
                bin_width=bin_width,
                rotary_t_max=rotary_t_max,
                batch_type="stacked",
                backend="math",
            )

        self.ssl_bin_width = bin_width
        self.possm_bin_width = bin_width
        # self.possm_bin_step = possm_bin_step
        self.dim = dim
        if decoding_head == "none":  # placeholder for future pur contrasive learning
            self.enable_pretrain = True
            self.enable_sl_ft = False
            assert masking_loss != "none", masking_loss
        elif decoding_head == "ssl":
            self.enable_pretrain = True
            self.enable_sl_ft = False
        elif decoding_head == "sl":
            self.enable_pretrain = False
            self.enable_sl_ft = True
        elif decoding_head == "join":
            self.enable_pretrain = True
            self.enable_sl_ft = True

        if self.enable_pretrain or self.enable_sl_ft:
            if backbone == "transformer":
                hidden_dim = dim
            elif "ssm" in backbone:
                if backbone == "ssm_bigru":
                    hidden_dim = dim_ssm * 2
                else:
                    hidden_dim = dim_ssm

            self.output_ca_restrict_context = output_ca_restrict_context
            self.output_ca_ctx_lim = output_ca_ctx_lim

        if self.enable_pretrain:
            self.projection_spike_rate = nn.Linear(qry_ca_context_dim, 1)
            if decoding_head_model == "ca":
                self.spike_decoder = PerceiverRotaryDecoder(
                    dim=hidden_dim,
                    context_dim=qry_ca_context_dim,
                    dim_head=dim_head,
                    depth=output_ssl_depth,
                    cross_heads=dec_cross_heads,
                    self_heads=dec_self_heads,
                    ffn_dropout=ffn_dropout,
                    lin_dropout=lin_dropout,
                    atn_dropout=atn_dropout,
                    batch_type="stacked",
                    backend=attention_backend[2],  #'math',
                    t_max=rotary_t_max,
                )
        # make sure mask token is initialized when enabling ssl pretraining mode
        assert self.enable_pretrain == self.mask_autoencoder.enable_masked_pretrain, (
            "enable_pretrain flag must be consistent"
        )

        if self.enable_sl_ft:
            self.task_emb = Embedding(
                Decoder.max_value() + 1, dim, init_scale=emb_init_scale
            )
            if backbone == "transformer":
                self.readout = MultitaskReadout(
                    latent_dim=dim,
                    decoder_specs=task_specs,
                    batch_type=self.batch_type[2],
                )
            elif "ssm" in backbone:
                self.readout = POSSMMultitaskReadout(
                    latent_dim=dim,
                    decoder_specs=task_specs,
                    use_output_ca=(decoding_head_model == "ca"),
                )

            if decoding_head_model == "ca":
                self.task_decoder = PerceiverRotaryDecoder(
                    dim=hidden_dim,
                    context_dim=dim,
                    dim_head=dim_head,
                    depth=output_dec_depth,
                    cross_heads=dec_cross_heads,
                    self_heads=dec_self_heads,
                    ffn_dropout=ffn_dropout,
                    lin_dropout=lin_dropout,
                    atn_dropout=atn_dropout,
                    batch_type="stacked",
                    backend=attention_backend[2],  #'math',
                    t_max=rotary_t_max,
                )

    def freeze_ssl_embedding(self) -> List[nn.Module]:
        # Freeze everything except the ssl module
        frozen_modules = []
        freeze_modules = [
            self.unit_emb,
            self.spike_type_emb,
            self.latent_emb,
        ]
        for module in self.children():
            if not (module in freeze_modules):
                continue
            for param in module.parameters():
                param.requires_grad = False
            frozen_modules.append(module)

        return frozen_modules

    def freeze_shared_embedding(self) -> List[nn.Module]:
        # Freeze everything except the ssl module
        frozen_modules = []
        freeze_modules = [
            self.spike_type_emb,
            self.latent_emb,
        ]
        if self.enable_sl_ft:
            freeze_modules.append(self.task_emb)
        for module in self.children():
            if not (module in freeze_modules):
                continue
            for param in module.parameters():
                param.requires_grad = False
            frozen_modules.append(module)

        return frozen_modules

    def freeze_encoder(self) -> List[nn.Module]:
        # Freeze everything except the ssl module
        frozen_modules = []
        freeze_modules = [
            self.perceiver_encoder,
        ]

        for module in self.children():
            if not (module in freeze_modules):
                continue
            for param in module.parameters():
                param.requires_grad = False
            frozen_modules.append(module)

        return frozen_modules

    def freeze_autoencoder(self) -> List[nn.Module]:
        # Freeze everything except the ssl module
        frozen_modules = []
        freeze_modules = [
            self.mask_autoencoder,
        ]
        for module in self.children():
            if not (module in freeze_modules):
                continue
            for param in module.parameters():
                param.requires_grad = False
            frozen_modules.append(module)

        return frozen_modules

    def freeze_decoder(self) -> List[nn.Module]:
        # Freeze everything except the ssl module
        frozen_modules = []
        if self.enable_pretrain:
            freeze_modules = [
                self.spike_decoder,
                self.projection_spike_rate,
            ]
        if self.enable_sl_ft:
            freeze_modules = [
                self.readout,
                self.task_decoder,
            ]

        for module in self.children():
            if not (module in freeze_modules):
                continue
            for param in module.parameters():
                param.requires_grad = False
            frozen_modules.append(module)

        return frozen_modules

    def freeze_middle(self) -> List[nn.Module]:
        # Freeze everything except unit embedding, and session embedding
        frozen_modules = []
        banned_modules = [
            self.unit_emb,
            self.session_emb,
        ]
        for module in self.children():
            if module in banned_modules:
                continue
            for param in module.parameters():
                param.requires_grad = False
            frozen_modules.append(module)

        return frozen_modules

    def freeze_middle_newTask(self) -> List[nn.Module]:
        # Freeze everything except unit embedding, and session embedding
        frozen_modules = []
        banned_modules = [
            self.unit_emb,
            self.session_emb,
        ]
        if self.enable_sl_ft:
            banned_modules += [
                self.task_emb,
                self.readout,
            ]
        for module in self.children():
            if module in banned_modules:
                continue
            for param in module.parameters():
                param.requires_grad = False
            frozen_modules.append(module)

        return frozen_modules

    def unfreeze_middle(self) -> None:
        for module in self.children():
            for param in module.parameters():
                param.requires_grad = True

    def retrieve_latents_sequence(
        self,
        *,
        # input sequence
        spike_unit_index,  # (B, T, N_spike)
        spike_timestamps,  # (B, T, N_spike)
        spike_type,  # (B, T, N_spike)
        spike_mask,  # (B, T, N_spike)
        special_token_unit_index=None,  # (B, N_Unit)
        special_token_mask=None,  # (B, N_Unit)
        input_value=None,  # for poyo plus tokenizer
        num_intervals=50,
        # latent sequence
        latent_index,  # (B, N_latent_per_bin)
        latent_timestamps,  # (B, N_latent_per_bin)
        sl_mode=True,
    ):

        # flatten spike sequence as poyo do full-seq attention
        batch_size = spike_unit_index.size(0)
        spike_unit_index = spike_unit_index.view(batch_size, -1)
        spike_timestamps = spike_timestamps.view(batch_size, -1)
        spike_type = spike_type.view(batch_size, -1)
        spike_mask = spike_mask.view(batch_size, -1)
        if self.use_value_emb:
            input_value = input_value.view(batch_size, -1, input_value.size(-1))

        # append special tokens
        poyo_bin_width = 1.0 / 8
        expected_seq_len = num_intervals * self.ssl_bin_width
        # assert torch.abs(expected_seq_len - 1.) < 1e-4

        if not self.use_value_emb:
            input_unit_index = torch.cat(
                (special_token_unit_index, special_token_unit_index, spike_unit_index),
                dim=-1,
            )
            input_timestamps = torch.cat(
                (
                    torch.zeros_like(special_token_unit_index).to(spike_timestamps),
                    torch.ones_like(special_token_unit_index).to(spike_timestamps),
                    spike_timestamps,
                ),
                dim=-1,
            )
            if sl_mode:
                input_token_type = torch.cat(
                    (
                        (
                            special_token_mask.long() * int(TokenType.START_OF_SEQUENCE)
                        ).to(
                            spike_type
                        ),  # doesnt want to train separately start & end token
                        (special_token_mask.long() * int(TokenType.END_OF_SEQUENCE)).to(
                            spike_type
                        ),
                        spike_type,
                    ),
                    dim=-1,
                )
            else:
                input_token_type = torch.cat(
                    (
                        (
                            special_token_mask.long() * int(TokenType.EXIST_IN_SEQUENCE)
                        ).to(
                            spike_type
                        ),  # doesnt want to train separately start & end token
                        (
                            special_token_mask.long() * int(TokenType.EXIST_IN_SEQUENCE)
                        ).to(spike_type),
                        spike_type,
                    ),
                    dim=-1,
                )
            input_token_mask = torch.cat(
                (special_token_mask, special_token_mask, spike_mask), dim=-1
            )

            inputs = self.unit_emb(input_unit_index) + self.spike_type_emb(
                input_token_type
            )
        else:
            input_unit_index = spike_unit_index
            input_timestamps = spike_timestamps
            input_token_type = spike_type
            input_token_mask = spike_mask
            inputs = []
            inputs.append(
                self.unit_emb(input_unit_index) + self.spike_type_emb(input_token_type)
            )
            inputs.append(self.value_emb(input_value))
            inputs = torch.cat(
                inputs,
                dim=-1,
            )

        # latents
        full_latent_timestamps = (
            torch.arange(poyo_bin_width, expected_seq_len + 1e-6, poyo_bin_width)
            - poyo_bin_width / 2
        )
        num_time_steps = len(full_latent_timestamps)
        # assert num_time_steps == 8
        full_latent_timestamps = full_latent_timestamps.to(latent_timestamps.device)
        full_latent_timestamps = (
            full_latent_timestamps.unsqueeze(0)
            .unsqueeze(2)
            .repeat(batch_size, 1, latent_index.size(-1))
            .view(batch_size, -1)
        )
        full_latent_index = latent_index.repeat(1, num_time_steps)
        latents = self.latent_emb(full_latent_index)

        output_latents = self.perceiver_encoder(
            inputs=inputs,
            latents=latents,
            input_timestamps=input_timestamps,
            latent_timestamps=full_latent_timestamps,
            input_mask=input_token_mask,
            sa_depthwise_att=False,
        )

        return output_latents

    def retrieve_latents_iteration(
        self,
        *,
        # input sequence
        spike_unit_index,  # (B, T, N_spike)
        spike_timestamps,  # (B, T, N_spike)
        spike_type,  # (B, T, N_spike)
        spike_mask,  # (B, T, N_spike)
        special_token_unit_index=None,  # (B, N_Unit)
        special_token_mask=None,  # (B, N_Unit)
        input_value=None,
        num_intervals,
        # latent sequence
        latent_index,  # (B, N_latent_per_bin)
        latent_timestamps,  # (B, N_latent_per_bin)
        # mode
        input_bin_width=0.02,
        sl_mode=True,
    ):

        outputs = []

        # NOTE: So we can iterate over bins
        spike_timestamps = spike_timestamps.permute(1, 0, 2)  # (T, B, N_spike)
        spike_unit_index = spike_unit_index.permute(1, 0, 2)
        spike_type = spike_type.permute(1, 0, 2)
        spike_mask = spike_mask.permute(1, 0, 2)
        if self.use_value_emb:
            input_value = input_value.permute(1, 0, 2, 3)

        if (not self.use_value_emb) and (not sl_mode):
            assert special_token_unit_index is not None

        for i in range(num_intervals):
            if not self.use_value_emb:
                if sl_mode:
                    input_unit_index = spike_unit_index[i]
                    input_timestamps = spike_timestamps[i]
                    input_token_type = spike_type[i]
                    input_token_mask = spike_mask[i]
                else:
                    input_unit_index = torch.cat(
                        (special_token_unit_index, spike_unit_index[i]), dim=-1
                    )
                    input_timestamps = torch.cat(
                        (
                            (
                                latent_timestamps[:, :1].repeat(
                                    1, special_token_unit_index.size(-1)
                                )
                                + input_bin_width * i
                            ).to(spike_timestamps),
                            spike_timestamps[i],
                        ),
                        dim=-1,
                    )
                    input_token_type = torch.cat(
                        (
                            (
                                special_token_mask.long()
                                * int(TokenType.EXIST_IN_SEQUENCE)
                            ).to(spike_type),
                            spike_type[i],
                        ),
                        dim=-1,
                    )
                    input_token_mask = torch.cat(
                        (special_token_mask, spike_mask[i]), dim=-1
                    )

                inputs = self.unit_emb(input_unit_index) + self.spike_type_emb(
                    input_token_type
                )
            else:
                input_unit_index = spike_unit_index[i]
                input_timestamps = spike_timestamps[i]
                input_token_type = spike_type[i]
                input_token_mask = spike_mask[i]
                input_bin_value = input_value[i]

                inputs = []
                inputs.append(
                    self.unit_emb(input_unit_index)
                    + self.spike_type_emb(input_token_type)
                )
                inputs.append(self.value_emb(input_bin_value))
                inputs = torch.cat(
                    inputs,
                    dim=-1,
                )

            assert latent_index.size(1) == self.num_latents
            bin_latents = self.latent_emb(latent_index)
            shift_input_ca_timestamps = False
            if shift_input_ca_timestamps:
                bin_spike_timestamps = input_timestamps
                bin_latent_timestamps = (latent_timestamps + input_bin_width * i).to(
                    latent_timestamps
                )
            else:
                bin_spike_timestamps = (input_timestamps - input_bin_width * i).to(
                    spike_timestamps
                )
                bin_latent_timestamps = latent_timestamps

            ca_output = self.perceiver_encoder(
                inputs=inputs,
                latents=bin_latents,
                input_timestamps=bin_spike_timestamps,
                latent_timestamps=bin_latent_timestamps,
                input_mask=input_token_mask,
                sa_depthwise_att=True,
            )

            outputs.append(ca_output)

        output_latents = torch.stack(outputs, 1)  # B x T x N_latent_per_bin x D
        output_latents = output_latents.view(
            output_latents.size(0),
            num_intervals.item() * self.num_latents,
            output_latents.size(3),
        )

        return output_latents

    def project_latents_poyo(
        self,
        *,
        latents,  # (B, N_latent, dim) or (N_all_latent, dim)
        latent_timestamps,  # (B, N_latent) or (N_all_latent,)
        session_index,
        output_query_timestamps,  # (B, N_out) or (N_all_out,)
        output_decoder_index,
    ):

        output_queries = self.task_emb(output_decoder_index) + self.session_emb(
            session_index
        )

        assert self.batch_type[1] == "stacked"
        causal_projector = False
        batch_size, kv_len, _ = latents.size()
        q_len = output_queries.size(1)
        if causal_projector:
            output_timestamps_seq_expanded = output_query_timestamps.unsqueeze(
                2
            ).expand(batch_size, q_len, kv_len)
            latent_timestamps_seq_expanded = latent_timestamps.unsqueeze(1).expand(
                batch_size, q_len, kv_len
            )
            latent_mask = (
                output_timestamps_seq_expanded - latent_timestamps_seq_expanded
            ) >= 0

            # since the latent and output query seqs are asynchronous, certain output steps can predate the first latent timestamps,
            # they will be given the first latent timestamps as workaround, so this is technically not exactly a causal mask
            batch_query_unmasked = torch.sum(latent_mask, dim=2)
            if not (
                batch_query_unmasked > 0
            ).all():  # if certain query have no unmasked latent
                minimal_causal_mask = torch.zeros(kv_len).to(latents.device)
                minimal_causal_mask[: self.num_latents] = 1.0
                minimal_causal_mask = minimal_causal_mask.bool()
                latent_mask[batch_query_unmasked == 0] = minimal_causal_mask.unsqueeze(
                    0
                ).expand((batch_query_unmasked == 0).sum(), -1)

            latent_mask = latent_mask.to(output_queries.device).bool()
        else:
            latent_mask = None

        # probably can use poyo decoder or concat context token
        if self.decoding_head_model == "ca":
            outputs = self.task_decoder(
                latents=latents,  # (B, N_latent, dim) or (N_all_latent, dim)
                output_queries=output_queries,  # (B, N_out, dim) or (N_all_out, dim)
                latent_timestamps=latent_timestamps,  # (B, N_latent) or (N_all_latent,)
                output_query_timestamps=output_query_timestamps,  # (B, N_out) or (N_all_out,)
                latent_mask=latent_mask,  # (B, N_latent, N_latent)
            )

        return outputs

    def project_latents_possm(
        self,
        *,
        concat_latents,  # (B, N_latent, dim) or (N_all_latent, dim)
        concat_latent_timestamps,  # (B, N_latent) or (N_all_latent,)
        session_index,
        output_query_timestamps,  # (B, N_out) or (N_all_out,)
        output_decoder_index,
        output_bin_index=None,  # needed for possm
    ):

        output_queries = self.task_emb(output_decoder_index) + self.session_emb(
            session_index
        )

        B, NL, DL = concat_latents.size()
        if self.backbone == "ssm_bigru":
            assert DL == self.dim_ssm * 2, concat_latents.size()
        else:
            assert DL == self.dim_ssm, concat_latents.size()

        if "ssm" in self.backbone:
            latent_mask = torch.tril(
                torch.ones(NL + 1, NL + 1, device=output_query_timestamps.device)
            )[:, 1:].to(torch.bool)
            latent_mask[0][0] = True
            if self.output_ca_restrict_context:
                for i in range(self.output_ca_ctx_lim + 1, NL + 1):
                    latent_mask[i][: i - self.output_ca_ctx_lim] = False
            latent_mask = latent_mask[output_bin_index.flatten()].reshape(
                output_bin_index.shape[0], output_bin_index.shape[1], -1
            )
        else:
            latent_mask = None

        # probably can use poyo decoder or concat context token
        if self.decoding_head_model == "ca":
            outputs = self.task_decoder(
                latents=concat_latents,  # (B, N_latent, dim) or (N_all_latent, dim)
                output_queries=output_queries,  # (B, N_out, dim) or (N_all_out, dim)
                latent_timestamps=concat_latent_timestamps,  # (B, N_latent) or (N_all_latent,)
                output_query_timestamps=output_query_timestamps,  # (B, N_out) or (N_all_out,)
                latent_mask=latent_mask,  # (B, N_latent, N_latent)
            )

        return outputs

    def predict_spike_rate_poyo(
        self,
        *,
        latents,  # (B, N_latent, dim)
        latent_timestamps,  # (B, N_latent)
        #
        query_unit_index,  # (B, N_out)
        query_timestamps,  # (B, N_out)
        query_spike_type,  # (B, N_out)
        query_unit_counts,  # (B, N_out)
        query_assignment=None,  # (B, N_out)
        query_mask=None,
        #
        task_mask,  # mask for current masking strategy: causal / unit / random (B, N_latent)
        task_name,  # task identifier, needed to construct loss_taskwise (str)
    ):

        B, NL, _ = latents.size()
        NS = int(NL // self.num_latents)

        output_queries = self.unit_emb(query_unit_index) + self.spike_type_emb(
            query_spike_type
        )

        batch_size, q_len = query_unit_index.size()
        kv_len = latent_timestamps.size(1)
        assert q_len == query_assignment.size(1)
        query_att_mask = (
            query_mask.clone().unsqueeze(2).expand(batch_size, q_len, kv_len)
        )

        if self.qry_cross_att_per_bin or (
            (task_name == str(MaskingType.TEMPORAL_MASKING))
            or (task_name == str(MaskingType.CAUSAL_MASKING))
        ):
            assert query_assignment.dtype == torch.int64
            assert query_assignment.min() >= 0 and query_assignment.max() < (NS + 1)

            bin_indices = (
                torch.arange(1, NS + 1)
                .to(query_assignment)
                .unsqueeze(0)
                .unsqueeze(1)
                .expand(batch_size, q_len, NS)
            )
            query_assignment_mask = (
                query_assignment.unsqueeze(2).expand(batch_size, q_len, NS)
                - bin_indices
            )
            query_assignment_mask = torch.where(
                query_assignment_mask.abs() < self.spike_latent_bin_tolerance,
                True,
                False,
            )
            query_assignment_mask = query_assignment_mask.repeat_interleave(
                self.num_latents, dim=2
            )
            query_att_mask = query_att_mask & query_assignment_mask

        rate_prediction_mask = query_att_mask.clone().view(
            B, q_len, NS, self.num_latents
        )
        if (task_name == str(MaskingType.TEMPORAL_MASKING)) or (
            task_name == str(MaskingType.CAUSAL_MASKING)
        ):
            rate_prediction_mask = (rate_prediction_mask.sum(3) > 0).bool()

            masked_bin_mask = (
                task_mask.view(B, NS, self.num_latents).sum(2) > 0
            ).bool()
            masked_bin_mask = masked_bin_mask.unsqueeze(1).expand(B, q_len, NS)
            rate_prediction_mask = (
                (rate_prediction_mask & masked_bin_mask).sum(2) > 0
            ).bool()
        else:
            raise NotImplementedError()
        assert rate_prediction_mask.dim() == 2

        if self.decoding_head_model == "ca":
            if self.qry_cross_att_per_bin:
                outputs = self.spike_decoder(
                    latents=latents,  # (B, N_latent, dim) or (N_all_latent, dim)
                    output_queries=output_queries,  # (B, N_out, dim) or (N_all_out, dim)
                    latent_timestamps=latent_timestamps,  # (B, N_latent) or (N_all_latent,)
                    output_query_timestamps=query_timestamps,  # (B, N_out) or (N_all_out,)
                    latent_mask=query_att_mask,  # (B, N_latent, N_latent)
                )
            else:
                assert self.spike_decoder.dec_atn.batch_type == "stacked"
                outputs = self.spike_decoder(
                    latents=latents,  # (B, N_latent, dim) or (N_all_latent, dim)
                    output_queries=output_queries,  # (B, N_out, dim) or (N_all_out, dim)
                    latent_timestamps=latent_timestamps,  # (B, N_latent) or (N_all_latent,)
                    output_query_timestamps=query_timestamps,  # (B, N_out) or (N_all_out,)
                    latent_mask=None,  # (B, N_latent, N_latent)
                )

        assert outputs.dim() == 3
        log_output_rate = self.projection_spike_rate(outputs).squeeze(2)
        taskwise_outputs = [
            {
                "SPIKE_RATE_PREDICTION_" + task_name: log_output_rate[i][
                    rate_prediction_mask[i]
                ]
            }
            for i in range(B)
        ]
        taskwise_loss = {}
        taskwise_metric = {}
        loss = torch.tensor(0, device=outputs.device, dtype=torch.float32)

        taskwise_loss["SPIKE_RATE_PREDICTION_" + task_name] = F.poisson_nll_loss(
            log_output_rate[rate_prediction_mask],
            query_unit_counts[rate_prediction_mask],
            log_input=True,
            reduction="mean",
        )

        with torch.no_grad():
            output_rate = torch.exp(log_output_rate)

            r2score = R2Score().to(output_rate.device)
            taskwise_metric["SPIKE_RATE_INBATCH_R2_" + task_name] = r2score(
                output_rate[rate_prediction_mask],
                query_unit_counts[rate_prediction_mask],
            )

            batch_rate_prediction_mask = rate_prediction_mask.sum(-1) > 0
            r2_batch = []
            for i in range(B):
                if batch_rate_prediction_mask[i]:
                    r2_batch.append(
                        r2score(
                            output_rate[i][rate_prediction_mask[i]],
                            query_unit_counts[i][rate_prediction_mask[i]],
                        )
                    )
            taskwise_metric["SPIKE_RATE_PERBATCH_R2_" + task_name] = torch.mean(
                torch.stack(r2_batch)
            )
            taskwise_metric["SPIKE_RATE_PERBATCH_R2_ALL_" + task_name] = torch.stack(
                r2_batch
            )

        loss = taskwise_loss["SPIKE_RATE_PREDICTION_" + task_name]

        return taskwise_outputs, loss, taskwise_loss, taskwise_metric

    def predict_spike_rate_possm(
        self,
        *,
        concat_latents,  # (B, N_latent, dim)
        concat_latent_timestamps,  # (B, N_latent)
        #
        query_unit_index,  # (B, N_out)
        query_timestamps,  # (B, N_out)
        query_spike_type,  # (B, N_out)
        query_unit_counts,  # (B, N_out)
        query_assignment=None,  # (B, N_out)
        query_mask=None,
        # rnn implicitly equivalent to causal masking, mask is trianglar,
        task_mask,  # this is to mask out certain time points (eariler) on loss function
        task_name,  # task identifier, needed to construct loss_taskwise (str)
    ):

        B, NL, DL = concat_latents.size()
        assert DL == self.dim_ssm, concat_latents.size()

        output_queries = self.unit_emb(query_unit_index) + self.spike_type_emb(
            query_spike_type
        )

        batch_size, q_len = query_unit_index.size()
        kv_len = concat_latent_timestamps.size(1)
        assert kv_len == NL
        assert NL == task_mask.size(1)
        assert q_len == query_assignment.size(1)
        query_att_mask = (
            query_mask.clone().unsqueeze(2).expand(batch_size, q_len, kv_len)
        )

        if self.qry_cross_att_per_bin or (
            (task_name == str(MaskingType.TEMPORAL_MASKING))
            or (task_name == str(MaskingType.CAUSAL_MASKING))
        ):
            assert query_assignment.dtype == torch.int64
            assert query_assignment.min() >= 0 and query_assignment.max() < (NL + 1)

            bin_indices = (
                torch.arange(1, NL + 1)
                .to(query_assignment)
                .unsqueeze(0)
                .unsqueeze(1)
                .expand(batch_size, q_len, NL)
            )
            query_assignment_mask = (
                query_assignment.unsqueeze(2).expand(batch_size, q_len, NL)
                - bin_indices
            )
            query_assignment_mask = torch.where(
                query_assignment_mask.abs() < self.spike_latent_bin_tolerance,
                True,
                False,
            )
            query_att_mask = query_att_mask & query_assignment_mask

        rate_prediction_mask = query_att_mask.clone().view(B, q_len, NL)
        if (task_name == str(MaskingType.CAUSAL_MASKING)) or (
            task_name == str(MaskingType.TEMPORAL_MASKING)
        ):
            rate_prediction_mask = (rate_prediction_mask > 0).bool()

            masked_bin_mask = (task_mask.view(B, NL) > 0).bool()
            masked_bin_mask = masked_bin_mask.unsqueeze(1).expand(B, q_len, NL)
            rate_prediction_mask = (
                (rate_prediction_mask & masked_bin_mask).sum(2) > 0
            ).bool()
        else:
            raise NotImplementedError()
        assert rate_prediction_mask.dim() == 2

        if self.decoding_head_model == "ca":
            if self.qry_cross_att_per_bin:
                outputs = self.spike_decoder(
                    latents=concat_latents,  # (B, N_latent, dim) or (N_all_latent, dim)
                    output_queries=output_queries,  # (B, N_out, dim) or (N_all_out, dim)
                    latent_timestamps=concat_latent_timestamps,  # (B, N_latent) or (N_all_latent,)
                    output_query_timestamps=query_timestamps,  # (B, N_out) or (N_all_out,)
                    latent_mask=query_att_mask,  # (B, N_latent, N_latent)
                )
            else:
                assert self.spike_decoder.dec_atn.batch_type == "stacked"
                outputs = self.spike_decoder(
                    latents=concat_latents,  # (B, N_latent, dim) or (N_all_latent, dim)
                    output_queries=output_queries,  # (B, N_out, dim) or (N_all_out, dim)
                    latent_timestamps=concat_latent_timestamps,  # (B, N_latent) or (N_all_latent,)
                    output_query_timestamps=query_timestamps,  # (B, N_out) or (N_all_out,)
                    latent_mask=None,  # (B, N_latent, N_latent)
                )
            # outputs[torch.isnan(outputs)] = 1e5

        assert outputs.dim() == 3
        log_output_rate = self.projection_spike_rate(outputs).squeeze(2)
        taskwise_outputs = [
            {
                "SPIKE_RATE_PREDICTION_" + task_name: log_output_rate[i][
                    rate_prediction_mask[i]
                ]
            }
            for i in range(B)
        ]
        taskwise_loss = {}
        taskwise_metric = {}
        loss = torch.tensor(0, device=outputs.device, dtype=torch.float32)

        taskwise_loss["SPIKE_RATE_PREDICTION_" + task_name] = F.poisson_nll_loss(
            log_output_rate[rate_prediction_mask],
            query_unit_counts[rate_prediction_mask],
            log_input=True,
            reduction="mean",
        )

        with torch.no_grad():
            output_rate = torch.exp(log_output_rate)

            r2score = R2Score().to(output_rate.device)
            taskwise_metric["SPIKE_RATE_INBATCH_R2_" + task_name] = r2score(
                output_rate[rate_prediction_mask],
                query_unit_counts[rate_prediction_mask],
            )

            batch_rate_prediction_mask = rate_prediction_mask.sum(-1) > 0
            r2_batch = []
            for i in range(B):
                if batch_rate_prediction_mask[i]:
                    r2_batch.append(
                        r2score(
                            output_rate[i][rate_prediction_mask[i]],
                            query_unit_counts[i][rate_prediction_mask[i]],
                        )
                    )
            taskwise_metric["SPIKE_RATE_PERBATCH_R2_" + task_name] = torch.mean(
                torch.stack(r2_batch)
            )
            taskwise_metric["SPIKE_RATE_PERBATCH_R2_ALL_" + task_name] = torch.stack(
                r2_batch
            )

        loss = taskwise_loss["SPIKE_RATE_PREDICTION_" + task_name]

        return taskwise_outputs, loss, taskwise_loss, taskwise_metric

    def predict_unit_value_possm(
        self,
        *,
        concat_latents,  # (B, N_latent, dim)
        concat_latent_timestamps,  # (B, N_latent)
        #
        query_unit_index,  # (B, N_out)
        query_timestamps,  # (B, N_out)
        query_unit_type,  # (B, N_out)
        query_unit_value,  # (B, N_out)
        query_assignment=None,  # (B, N_out)
        query_mask=None,
        # rnn implicitly equivalent to causal masking, mask is trianglar,
        task_mask,  # this is to mask out certain time points (eariler) on loss function
        task_name,  # task identifier, needed to construct loss_taskwise (str)
    ):

        B, NL, DL = concat_latents.size()
        if self.backbone == "ssm_bigru":
            assert DL == self.dim_ssm * 2, concat_latents.size()
        else:
            assert DL == self.dim_ssm, concat_latents.size()

        output_queries = self.unit_emb(query_unit_index) + self.spike_type_emb(
            query_unit_type
        )

        batch_size, q_len = query_unit_index.size()
        kv_len = concat_latent_timestamps.size(1)
        assert kv_len == NL
        assert NL == task_mask.size(1)
        assert q_len == query_assignment.size(1)
        query_att_mask = (
            query_mask.clone().unsqueeze(2).expand(batch_size, q_len, kv_len)
        )

        if self.qry_cross_att_per_bin or (
            (task_name == str(MaskingType.TEMPORAL_MASKING))
            or (task_name == str(MaskingType.CAUSAL_MASKING))
        ):
            assert query_assignment.dtype == torch.int64
            assert query_assignment.min() >= 0 and query_assignment.max() < (NL + 1)

            bin_indices = (
                torch.arange(1, NL + 1)
                .to(query_assignment)
                .unsqueeze(0)
                .unsqueeze(1)
                .expand(batch_size, q_len, NL)
            )
            query_assignment_mask = (
                query_assignment.unsqueeze(2).expand(batch_size, q_len, NL)
                - bin_indices
            )
            query_assignment_mask = torch.where(
                query_assignment_mask.abs() < self.spike_latent_bin_tolerance,
                True,
                False,
            )
            query_att_mask = query_att_mask & query_assignment_mask

        unit_projection_mask = query_att_mask.clone().view(B, q_len, NL)
        if (task_name == str(MaskingType.CAUSAL_MASKING)) or (
            task_name == str(MaskingType.TEMPORAL_MASKING)
        ):
            unit_projection_mask = (unit_projection_mask > 0).bool()

            masked_bin_mask = (task_mask.view(B, NL) > 0).bool()
            masked_bin_mask = masked_bin_mask.unsqueeze(1).expand(B, q_len, NL)
            unit_projection_mask = (
                (unit_projection_mask & masked_bin_mask).sum(2) > 0
            ).bool()
        else:
            raise NotImplementedError()
        assert unit_projection_mask.dim() == 2

        if self.decoding_head_model == "ca":
            if self.qry_cross_att_per_bin:
                outputs = self.spike_decoder(
                    latents=concat_latents,  # (B, N_latent, dim) or (N_all_latent, dim)
                    output_queries=output_queries,  # (B, N_out, dim) or (N_all_out, dim)
                    latent_timestamps=concat_latent_timestamps,  # (B, N_latent) or (N_all_latent,)
                    output_query_timestamps=query_timestamps,  # (B, N_out) or (N_all_out,)
                    latent_mask=query_att_mask,  # (B, N_latent, N_latent)
                )
            else:
                assert self.spike_decoder.dec_atn.batch_type == "stacked"
                outputs = self.spike_decoder(
                    latents=concat_latents,  # (B, N_latent, dim) or (N_all_latent, dim)
                    output_queries=output_queries,  # (B, N_out, dim) or (N_all_out, dim)
                    latent_timestamps=concat_latent_timestamps,  # (B, N_latent) or (N_all_latent,)
                    output_query_timestamps=query_timestamps,  # (B, N_out) or (N_all_out,)
                    latent_mask=None,  # (B, N_latent, N_latent)
                )

        assert outputs.dim() == 3
        output_value = self.projection_spike_rate(outputs)
        if query_unit_value.dim() == 2:
            output_value = output_value.squeeze(2)

        taskwise_outputs = [
            {
                "UNIT_VALUE_PROJECTION_" + task_name: output_value[i][
                    unit_projection_mask[i]
                ]
            }
            for i in range(B)
        ]
        taskwise_loss = {}
        taskwise_metric = {}
        loss = torch.tensor(0, device=outputs.device, dtype=torch.float32)

        taskwise_loss["UNIT_VALUE_PROJECTION_" + task_name] = F.mse_loss(
            output_value[unit_projection_mask],
            query_unit_value[unit_projection_mask],
            reduction="mean",
        )

        with torch.no_grad():
            r2score = R2Score().to(output_value.device)
            taskwise_metric["UNIT_VALUE_INBATCH_R2_" + task_name] = r2score(
                output_value[unit_projection_mask],
                query_unit_value[unit_projection_mask],
            )

            batch_unit_prediction_mask = unit_projection_mask.sum(-1) > 0
            r2_batch = []
            for i in range(B):
                if batch_unit_prediction_mask[i]:
                    r2_batch.append(
                        r2score(
                            output_value[i][unit_projection_mask[i]],
                            query_unit_value[i][unit_projection_mask[i]],
                        )
                    )
            taskwise_metric["UNIT_VALUE_PERBATCH_R2_" + task_name] = torch.mean(
                torch.stack(r2_batch)
            )

        loss = taskwise_loss["UNIT_VALUE_PROJECTION_" + task_name]

        return taskwise_outputs, loss, taskwise_loss, taskwise_metric

    def predict_unit_value_poyo(
        self,
        *,
        latents,  # (B, N_latent, dim)
        latent_timestamps,  # (B, N_latent)
        #
        query_unit_index,  # (B, N_out)
        query_timestamps,  # (B, N_out)
        query_unit_type,  # (B, N_out)
        query_unit_value,  # (B, N_out)
        query_assignment=None,  # (B, N_out)
        query_mask=None,
        #
        task_mask,  # mask for current masking strategy: causal / unit / random (B, N_latent)
        task_name,  # task identifier, needed to construct loss_taskwise (str)
    ):

        B, NL, _ = latents.size()
        NS = int(NL // self.num_latents)

        output_queries = self.unit_emb(query_unit_index) + self.spike_type_emb(
            query_unit_type
        )

        batch_size, q_len = query_unit_index.size()
        kv_len = latent_timestamps.size(1)
        assert q_len == query_assignment.size(1)
        query_att_mask = (
            query_mask.clone().unsqueeze(2).expand(batch_size, q_len, kv_len)
        )

        if self.qry_cross_att_per_bin or (
            (task_name == str(MaskingType.TEMPORAL_MASKING))
            or (task_name == str(MaskingType.CAUSAL_MASKING))
        ):
            assert query_assignment.dtype == torch.int64
            assert query_assignment.min() >= 0 and query_assignment.max() < (NS + 1)

            bin_indices = (
                torch.arange(1, NS + 1)
                .to(query_assignment)
                .unsqueeze(0)
                .unsqueeze(1)
                .expand(batch_size, q_len, NS)
            )
            query_assignment_mask = (
                query_assignment.unsqueeze(2).expand(batch_size, q_len, NS)
                - bin_indices
            )
            query_assignment_mask = torch.where(
                query_assignment_mask.abs() < self.spike_latent_bin_tolerance,
                True,
                False,
            )
            query_assignment_mask = query_assignment_mask.repeat_interleave(
                self.num_latents, dim=2
            )
            query_att_mask = query_att_mask & query_assignment_mask

        unit_projection_mask = query_att_mask.clone().view(
            B, q_len, NS, self.num_latents
        )
        if (task_name == str(MaskingType.TEMPORAL_MASKING)) or (
            task_name == str(MaskingType.CAUSAL_MASKING)
        ):
            unit_projection_mask = (unit_projection_mask.sum(3) > 0).bool()

            masked_bin_mask = (
                task_mask.view(B, NS, self.num_latents).sum(2) > 0
            ).bool()
            masked_bin_mask = masked_bin_mask.unsqueeze(1).expand(B, q_len, NS)
            unit_projection_mask = (
                (unit_projection_mask & masked_bin_mask).sum(2) > 0
            ).bool()
        else:
            raise NotImplementedError()
        assert unit_projection_mask.dim() == 2

        if self.decoding_head_model == "ca":
            if self.qry_cross_att_per_bin:
                outputs = self.spike_decoder(
                    latents=latents,  # (B, N_latent, dim) or (N_all_latent, dim)
                    output_queries=output_queries,  # (B, N_out, dim) or (N_all_out, dim)
                    latent_timestamps=latent_timestamps,  # (B, N_latent) or (N_all_latent,)
                    output_query_timestamps=query_timestamps,  # (B, N_out) or (N_all_out,)
                    latent_mask=query_att_mask,  # (B, N_latent, N_latent)
                )
            else:
                assert self.spike_decoder.dec_atn.batch_type == "stacked"
                outputs = self.spike_decoder(
                    latents=latents,  # (B, N_latent, dim) or (N_all_latent, dim)
                    output_queries=output_queries,  # (B, N_out, dim) or (N_all_out, dim)
                    latent_timestamps=latent_timestamps,  # (B, N_latent) or (N_all_latent,)
                    output_query_timestamps=query_timestamps,  # (B, N_out) or (N_all_out,)
                    latent_mask=None,  # (B, N_latent, N_latent)
                )

        assert outputs.dim() == 3
        output_value = self.projection_spike_rate(outputs)
        if query_unit_value.dim() == 2:
            output_value = output_value.squeeze(2)

        taskwise_outputs = [
            {
                "UNIT_VALUE_PROJECTION_" + task_name: output_value[i][
                    unit_projection_mask[i]
                ]
            }
            for i in range(B)
        ]
        taskwise_loss = {}
        taskwise_metric = {}
        loss = torch.tensor(0, device=outputs.device, dtype=torch.float32)

        taskwise_loss["UNIT_VALUE_PROJECTION_" + task_name] = F.mse_loss(
            output_value[unit_projection_mask],
            query_unit_value[unit_projection_mask],
            reduction="mean",
        )

        with torch.no_grad():
            r2score = R2Score().to(output_value.device)
            taskwise_metric["UNIT_VALUE_INBATCH_R2_" + task_name] = r2score(
                output_value[unit_projection_mask],
                query_unit_value[unit_projection_mask],
            )

            batch_unit_prediction_mask = unit_projection_mask.sum(-1) > 0
            r2_batch = []
            for i in range(B):
                if batch_unit_prediction_mask[i]:
                    r2_batch.append(
                        r2score(
                            output_value[i][unit_projection_mask[i]],
                            query_unit_value[i][unit_projection_mask[i]],
                        )
                    )
            taskwise_metric["UNIT_VALUE_PERBATCH_R2_" + task_name] = torch.mean(
                torch.stack(r2_batch)
            )

        loss = taskwise_loss["UNIT_VALUE_PROJECTION_" + task_name]

        return taskwise_outputs, loss, taskwise_loss, taskwise_metric

    def forward(
        self,
        *,
        # input sequence
        spike_unit_index,  # (B, T, N_spike)
        spike_timestamps,  # (B, T, N_spike)
        spike_type,
        input_mask=None,
        n_intervals=None,
        special_token_unit_index=None,
        special_token_mask=None,
        input_value=None,
        # latent sequence
        latent_index_per_bin,  # (B, N_latent_per_bin)
        latent_timestamps_per_bin,  # (B, N_latent_per_bin)
        # ssl output sequence
        query_unit_index=None,
        query_timestamps=None,
        query_spike_type=None,
        query_assignment=None,
        query_unit_counts=None,
        query_mask=None,
        # output sequence
        session_index,  # (B,)
        output_timestamps=None,  # (B, N_out)
        output_decoder_index=None,  # (B, N_out)
        output_batch_index=None,
        output_bin_index=None,
        output_values: Optional[Dict[str, torch.Tensor]] = None,
        output_weights: Optional[Dict[str, torch.Tensor]] = None,
        # # task
        # loss_type,
    ) -> Tuple[
        Dict[str, TensorType["batch", "*nqueries", "*nchannelsout"]],
        torch.Tensor,
        Dict[str, torch.Tensor],
    ]:

        # make sure arguments make sense
        if n_intervals is None:
            assert spike_unit_index.dim() == 2, (
                f"Expected stacked inputs with 2 dimensions (batch, num_tokens), "
                f"got ({spike_unit_index.size()})."
            )
        else:
            assert spike_unit_index.dim() == 3, (
                f"Expected stacked inputs with 3 dimensions (batch, num_tokens), "
                f"got ({spike_unit_index.size()})."
            )
        assert latent_index_per_bin.dim() == 2, (
            f"Expected stacked latents with 2 dimensions (batch, num_tokens), "
            f"got ({latent_index_per_bin.shape})."
        )

        B_s = spike_unit_index.size(0)
        if n_intervals is not None:
            num_intervals = max(n_intervals).long()
        if self.enable_pretrain:
            ssl_output_latents = self.retrieve_latents_iteration(
                spike_unit_index=spike_unit_index,
                spike_timestamps=spike_timestamps,
                spike_type=spike_type,
                spike_mask=input_mask,
                special_token_unit_index=special_token_unit_index,
                special_token_mask=special_token_mask,
                input_value=input_value,
                num_intervals=num_intervals,
                # latent sequence
                latent_index=latent_index_per_bin,  # (B, N_latent_per_bin)
                latent_timestamps=latent_timestamps_per_bin,  # (B, N_latent_pre_bin)
                input_bin_width=self.ssl_bin_width,
                sl_mode=False,
            )
        if self.enable_sl_ft:
            if self.backbone == "transformer":
                if self.separate_atn_enc_path:
                    dec_output_latents = self.retrieve_latents_sequence(
                        spike_unit_index=spike_unit_index,
                        spike_timestamps=spike_timestamps,
                        spike_type=spike_type,
                        spike_mask=input_mask,
                        special_token_unit_index=special_token_unit_index,
                        special_token_mask=special_token_mask,
                        input_value=input_value,
                        num_intervals=num_intervals,
                        # latent sequence
                        latent_index=latent_index_per_bin,
                        latent_timestamps=latent_timestamps_per_bin,
                        sl_mode=self.poyo_special_token,
                    )
                else:
                    dec_output_latents = ssl_output_latents
            elif "ssm" in self.backbone:
                if self.separate_ssm_enc_path:
                    dec_output_latents = self.retrieve_latents_iteration(
                        spike_unit_index=spike_unit_index,
                        spike_timestamps=spike_timestamps,
                        spike_type=spike_type,
                        spike_mask=input_mask,
                        input_value=input_value,
                        num_intervals=num_intervals,
                        # latent sequence
                        latent_index=latent_index_per_bin,
                        latent_timestamps=latent_timestamps_per_bin,
                        input_bin_width=self.possm_bin_width,
                        sl_mode=True,
                    )
                else:
                    dec_output_latents = ssl_output_latents

        num_intervals = num_intervals.item()  # max number of intervals in integar
        expected_seq_len = num_intervals * self.ssl_bin_width
        num_latents_per_bin = latent_index_per_bin.size(-1)
        assert num_latents_per_bin == self.num_latents

        if self.backbone == "transformer":
            if self.enable_pretrain:
                ssl_latent_timestamps = (
                    torch.arange(
                        self.ssl_bin_width, expected_seq_len + 1e-6, self.ssl_bin_width
                    )
                    - self.ssl_bin_width / 2
                )
                ssl_latent_timestamps = ssl_latent_timestamps.to(
                    latent_timestamps_per_bin.device
                )
                ssl_latent_timestamps = (
                    ssl_latent_timestamps.unsqueeze(0)
                    .unsqueeze(2)
                    .repeat(B_s, 1, num_latents_per_bin)
                    .view(B_s, -1)
                )
                ssl_latent_index = latent_index_per_bin.repeat(1, num_intervals)
            if self.enable_sl_ft:
                if self.separate_atn_enc_path:
                    poyo_bin_width = 1.0 / 8
                    dec_latent_timestamps = (
                        torch.arange(
                            poyo_bin_width, expected_seq_len + 1e-6, poyo_bin_width
                        )
                        - poyo_bin_width / 2
                    )
                    num_time_steps = len(dec_latent_timestamps)
                    # assert num_time_steps == 8
                    dec_latent_timestamps = dec_latent_timestamps.to(
                        latent_timestamps_per_bin.device
                    )
                    dec_latent_timestamps = (
                        dec_latent_timestamps.unsqueeze(0)
                        .unsqueeze(2)
                        .repeat(B_s, 1, num_latents_per_bin)
                        .view(B_s, -1)
                    )
                    dec_latent_index = latent_index_per_bin.repeat(1, num_time_steps)
                else:
                    dec_latent_timestamps = ssl_latent_timestamps
                    dec_latent_index = ssl_latent_index

        elif "ssm" in self.backbone:
            assert self.ssl_bin_width == self.possm_bin_width, (
                "only same bin width is supported"
            )
            if self.enable_pretrain:
                ssl_latent_timestamps = torch.arange(
                    self.ssl_bin_width, expected_seq_len + 1e-6, self.ssl_bin_width
                ).to(latent_timestamps_per_bin.device)
                ssl_latent_timestamps = (
                    ssl_latent_timestamps.unsqueeze(0)
                    .unsqueeze(2)
                    .repeat(B_s, 1, num_latents_per_bin)
                    .view(B_s, -1)
                )
            if self.enable_sl_ft:
                if self.separate_ssm_enc_path:
                    dec_latent_timestamps = torch.arange(
                        self.possm_bin_width,
                        expected_seq_len + 1e-6,
                        self.possm_bin_width,
                    ).to(latent_timestamps_per_bin.device)
                    dec_latent_timestamps = (
                        dec_latent_timestamps.unsqueeze(0)
                        .unsqueeze(2)
                        .repeat(B_s, 1, num_latents_per_bin)
                        .view(B_s, -1)
                    )
                else:
                    dec_latent_timestamps = ssl_latent_timestamps

        # Masking for SSL
        if self.enable_pretrain:
            assert ssl_latent_timestamps.size(1) == int(
                num_intervals * self.num_latents
            ), "hardcode seq len"
            if self.temporal_mask_mode == "forward":
                latent_task_name = str(MaskingType.CAUSAL_MASKING)
                latent_task_mask = (
                    torch.zeros((B_s, num_intervals))
                    .to(ssl_output_latents.device)
                    .bool()
                )
                latent_task_mask[:, -int(num_intervals * self.mask_ratio) :] = True
            elif self.temporal_mask_mode == "random":
                latent_task_name = str(MaskingType.TEMPORAL_MASKING)
                mask_probs = torch.full((B_s, num_intervals), self.mask_ratio).float()
                latent_task_mask = torch.bernoulli(mask_probs).to(
                    ssl_output_latents.device
                )
                latent_task_mask_binm = (latent_task_mask.sum(-1) < 1) | (
                    latent_task_mask.sum(-1) > (num_intervals - 1)
                )
                if torch.sum(latent_task_mask_binm) > 0:
                    batch_indices_no_mask = torch.arange(B_s).to(
                        latent_task_mask_binm.device
                    )
                    batch_indices_no_mask = batch_indices_no_mask[latent_task_mask_binm]
                    assert batch_indices_no_mask.size(0) > 0
                    for nmb_idx in batch_indices_no_mask:
                        time_tomask = torch.sort(
                            torch.randperm(num_intervals)[
                                : int(self.mask_ratio * num_intervals)
                            ]
                        )[0]
                        latent_task_mask[nmb_idx] = 0.0
                        latent_task_mask[nmb_idx, time_tomask] = 1.0
            if self.backbone == "transformer":
                latent_task_mask = (
                    latent_task_mask.unsqueeze(2)
                    .expand(-1, -1, self.num_latents)
                    .reshape(B_s, -1)
                    .bool()
                )
            else:
                latent_task_mask = latent_task_mask.bool()

            if self.backbone == "transformer":
                ssl_predicted_latents = self.mask_autoencoder(
                    unmasked_inputs=ssl_output_latents,
                    task_mask=latent_task_mask,
                    input_mask=None,
                    input_timestamps=ssl_latent_timestamps,
                    input_index=ssl_latent_index,
                    task_name=latent_task_name,
                )
            elif "ssm" in self.backbone:
                ssl_predicted_latents = self.mask_autoencoder(
                    unmasked_inputs=ssl_output_latents,
                    task_mask=latent_task_mask,
                    task_name=latent_task_name,
                    latent_timestamps=latent_timestamps_per_bin,
                    num_intervals=num_intervals,
                )
                assert ssl_predicted_latents.size(1) == num_intervals

        if self.enable_sl_ft:
            sl_latent_mask = None
            sl_mask_name = "NoMask"
            if self.backbone == "transformer":
                dec_predicted_latents = self.mask_autoencoder(
                    unmasked_inputs=dec_output_latents,
                    task_mask=sl_latent_mask,
                    input_mask=None,
                    input_timestamps=dec_latent_timestamps,
                    input_index=dec_latent_index,
                    task_name=sl_mask_name,
                )
                if self.enable_pretrain and not self.separate_atn_enc_path:
                    assert dec_predicted_latents.size() == ssl_predicted_latents.size()
            elif "ssm" in self.backbone:
                dec_predicted_latents = self.mask_autoencoder(
                    unmasked_inputs=dec_output_latents,
                    task_mask=sl_latent_mask,
                    task_name=sl_mask_name,
                    latent_timestamps=latent_timestamps_per_bin,
                    num_intervals=num_intervals,
                )
                assert dec_predicted_latents.size(1) == num_intervals
                if self.enable_pretrain:
                    assert dec_predicted_latents.size() == ssl_predicted_latents.size()
            if self.detach_sl_head:
                dec_predicted_latents = dec_predicted_latents.detach()

        loss = torch.tensor(0, device=spike_unit_index.device, dtype=torch.float32)
        results_ssl = None
        results_dec = None

        if self.enable_pretrain:
            if self.backbone == "transformer":
                if self.use_value_emb:
                    output_ssl, loss_ssl, losses_taskwise_ssl, metrics_taskwise_ssl = (
                        self.predict_unit_value_poyo(
                            latents=ssl_predicted_latents,  # (B, N_latent, D)
                            latent_timestamps=ssl_latent_timestamps,  # (B, N_latent)
                            # output queries
                            query_unit_index=query_unit_index,  # (B, N_out) or (N_all_out,)
                            query_timestamps=query_timestamps,  # (B, N_out) or (N_all_out,)
                            query_unit_type=query_spike_type,  # (B, N_out) or (N_all_out,)
                            query_assignment=query_assignment,  # (B, N_out) or (N_all_out,)
                            query_unit_value=query_unit_counts,  # (B, N_out) or (N_all_out,)
                            query_mask=query_mask,  # (B, N_out)
                            task_mask=latent_task_mask,
                            task_name=latent_task_name,
                        )
                    )
                else:
                    output_ssl, loss_ssl, losses_taskwise_ssl, metrics_taskwise_ssl = (
                        self.predict_spike_rate_poyo(
                            latents=ssl_predicted_latents,  # (B, N_latent, D)
                            latent_timestamps=ssl_latent_timestamps,  # (B, N_latent)
                            # output queries
                            query_unit_index=query_unit_index,  # (B, N_out) or (N_all_out,)
                            query_timestamps=query_timestamps,  # (B, N_out) or (N_all_out,)
                            query_spike_type=query_spike_type,  # (B, N_out) or (N_all_out,)
                            query_assignment=query_assignment,  # (B, N_out) or (N_all_out,)
                            query_unit_counts=query_unit_counts,  # (B, N_out) or (N_all_out,)
                            query_mask=query_mask,  # (B, N_out)
                            task_mask=latent_task_mask,
                            task_name=latent_task_name,
                        )
                    )
            if "ssm" in self.backbone:
                if self.use_value_emb:
                    output_ssl, loss_ssl, losses_taskwise_ssl, metrics_taskwise_ssl = (
                        self.predict_unit_value_possm(
                            concat_latents=ssl_predicted_latents,  # (B, T, D)
                            concat_latent_timestamps=ssl_latent_timestamps.view(
                                B_s, num_intervals, self.num_latents
                            )[:, :, 0],  # (B, T)
                            # output queries
                            query_unit_index=query_unit_index,  # (B, N_out) or (N_all_out,)
                            query_timestamps=query_timestamps,  # (B, N_out) or (N_all_out,)
                            query_unit_type=query_spike_type,  # (B, N_out) or (N_all_out,)
                            query_assignment=query_assignment,  # (B, N_out) or (N_all_out,)
                            query_unit_value=query_unit_counts,  # (B, N_out) or (N_all_out,)
                            query_mask=query_mask,  # (B, N_out)
                            task_mask=latent_task_mask,
                            task_name=latent_task_name,
                        )
                    )
                else:
                    output_ssl, loss_ssl, losses_taskwise_ssl, metrics_taskwise_ssl = (
                        self.predict_spike_rate_possm(
                            concat_latents=ssl_predicted_latents,  # (B, T, D)
                            concat_latent_timestamps=ssl_latent_timestamps.view(
                                B_s, num_intervals, self.num_latents
                            )[:, :, 0],  # (B, T)
                            # output queries
                            query_unit_index=query_unit_index,  # (B, N_out) or (N_all_out,)
                            query_timestamps=query_timestamps,  # (B, N_out) or (N_all_out,)
                            query_spike_type=query_spike_type,  # (B, N_out) or (N_all_out,)
                            query_assignment=query_assignment,  # (B, N_out) or (N_all_out,)
                            query_unit_counts=query_unit_counts,  # (B, N_out) or (N_all_out,)
                            query_mask=query_mask,  # (B, N_out)
                            task_mask=latent_task_mask,
                            task_name=latent_task_name,
                        )
                    )

            loss += loss_ssl * self.ssl_loss_coef
            results_ssl = (
                output_ssl,
                loss_ssl,
                losses_taskwise_ssl,
                metrics_taskwise_ssl,
            )

        if self.enable_sl_ft:
            if self.backbone == "transformer":
                projected_latents = self.project_latents_poyo(
                    latents=dec_predicted_latents,  # (B, N_latent, D)
                    latent_timestamps=dec_latent_timestamps,  # (B, N_latent)
                    # output queries
                    session_index=session_index,
                    output_query_timestamps=output_timestamps,  # (B, N_out) or (N_all_out,)
                    output_decoder_index=output_decoder_index,
                )

                output_dec, loss_dec, losses_taskwise_dec = self.readout(
                    output_latents=projected_latents,
                    output_decoder_index=output_decoder_index,
                    output_batch_index=output_batch_index,
                    output_values=output_values,
                    output_weights=output_weights,
                )
            elif "ssm" in self.backbone:
                # code in both poyo & possm style spike tokens, but force the possm style for SL for now
                # if n_intervals is None:
                projected_latents = self.project_latents_possm(
                    concat_latents=dec_predicted_latents,  # (B, T, D)
                    concat_latent_timestamps=dec_latent_timestamps.view(
                        B_s, num_intervals, self.num_latents
                    )[:, :, 0],  # (B, T)
                    # output queries
                    session_index=session_index,
                    output_query_timestamps=output_timestamps,  # (B, N_out) or (N_all_out,)
                    output_decoder_index=output_decoder_index,
                    output_bin_index=output_bin_index,
                )
                output_dec, loss_dec, losses_taskwise_dec = self.readout(
                    output_latents=projected_latents,
                    output_decoder_index=output_decoder_index,
                    output_values=output_values,
                    output_weights=output_weights,
                )
            loss += loss_dec * self.sl_loss_coef
            results_dec = (output_dec, loss_dec, losses_taskwise_dec)

        return loss, results_ssl, results_dec

class JoinSSLValueTokenizer:
    r"""Tokenizer used to tokenize Data for the Latent Join SSL with value
    embedding (POYO Plus style).

    This tokenizer can be called as a transform. If you are applying multiple
    transforms, make sure to apply this one last.

    Args:
        unit_tokenizer (Callable): Tokenizer for the units.
        session_tokenizer (Callable): Tokenizer for the sessions.
        decoder_registry (Dict): Registry of the decoders.
        weight_registry (Dict): Registry of the weights.
        latent_step (float): Step size for generating latent tokens.
        num_latents_per_step (int): Number of latents per step.
    """

    def __init__(
        self,
        unit_tokenizer,
        session_tokenizer,
        decoder_registry,
        latent_step,
        num_latents_per_step,
        batch_type,
        # ssl related
        query_max_unit_counts=10,
        include_decoding_outputs=True,
        fix_qry_unit_sampler=False,
        # possm
        dec_use_last_timestamps=False,
        dec_use_rel_timestamps=False,
        cst_rel_timestamps=0.5,
        cls_repeated_time=False,
        cls_num_repeated_time=1,
        patch_size=4,
        signal_freq=200.0,
        eval=False,
    ):
        self.unit_tokenizer = unit_tokenizer
        self.session_tokenizer = session_tokenizer

        self.decoder_registry = decoder_registry

        self.latent_step = latent_step
        self.num_latents_per_step = num_latents_per_step

        self.batch_type = batch_type
        for k in self.batch_type:
            assert k == "stacked", "forced batch_type to be stacked"
        self.eval = eval
        self.include_decoding_outputs = include_decoding_outputs
        self.fix_qry_unit_sampler = fix_qry_unit_sampler
        self.query_max_unit_counts = query_max_unit_counts  # if this is set to 0, which should be the case by default, then rate reconstruct is not in active
        if not self.include_decoding_outputs:
            assert self.query_max_unit_counts > 0

        self.dec_use_last_timestamps = dec_use_last_timestamps
        self.dec_use_rel_timestamps = dec_use_rel_timestamps
        assert not (self.dec_use_last_timestamps and self.dec_use_rel_timestamps)
        self.cst_rel_timestamps = cst_rel_timestamps
        # self.window_size = window_size
        self.patch_size = patch_size
        self.signal_freq = signal_freq
        self.bin_width = self.patch_size / float(self.signal_freq)
        self.bin_step = self.bin_width
        self.cls_repeated_time = cls_repeated_time
        self.cls_num_repeated_time = cls_num_repeated_time
        assert np.allclose(self.bin_width, 0.02)

    def __call__(self, data):
        # context window
        start, end = 0, round(data.end, 5)
        n_intervals = (
            np.floor(np.round((end - self.bin_width) / self.bin_step, 5)) + 1
        ).astype(int)
        interval_end = np.round(np.arange(self.bin_width, end + 1e-6, self.bin_step), 5)
        interval_start = np.round(interval_end - self.bin_width, 5)
        interval_end[-1] = end + 1e-6

        ### prepare input
        unit_ids = data.units.id
        bad_channel_mask = data.units.bad
        signal_values = data.signals.ecogs
        signal_timestamps = data.signals.timestamps
        assert np.sum(signal_values < -5000) == 0.0
        current_num_channels = signal_values.shape[
            1
        ]  # This is not necessarily the same number due to dropout, including both good & bad

        assert len(unit_ids) > 0, "require at least one unit to train"

        # remove bad channels
        signal_channel_ids = np.arange(current_num_channels)[~bad_channel_mask]
        signal_values = signal_values[:, signal_channel_ids]
        # assert (signal_values >= -1000).all()
        current_num_good_channels = signal_channel_ids.shape[0]

        assert np.allclose(
            signal_timestamps.shape[0], int(self.signal_freq * end), atol=1
        )
        # patchify, will use 'same' padding for conv to maintain the output dimension so that this trim makes sense still
        trim_idx = (
            signal_timestamps.shape[0] % self.patch_size
        )  # we trim at the beginning, as for many dataset used it is less important

        signal_timestamps = np.mean(
            rearrange(signal_timestamps[trim_idx:], "(t w) -> t w", w=self.patch_size),
            axis=1,
        )  # (patch_time,)

        signal_values = rearrange(
            signal_values[trim_idx:, :],
            "(t w) c -> t c w",
            w=self.patch_size,
        )

        # flatten channels
        signal_channel_ids = repeat(
            signal_channel_ids,
            "c -> (c t)",
            t=signal_timestamps.shape[0],
        )

        signal_values = rearrange(
            signal_values,
            "t c w -> (c t) w",
        )

        signal_timestamps = repeat(
            signal_timestamps,
            "t -> (c t)",
            c=current_num_good_channels,
        )

        signal_channel_ids = signal_channel_ids.astype(np.int64)
        signal_timestamps = signal_timestamps.astype(np.float64)
        signal_values = signal_values.astype(np.float32)
        signal_channel_token_types = np.full_like(
            signal_channel_ids, int(TokenType.EXIST_IN_SEQUENCE)
        )

        curr_query_max_unit_counts = min(
            current_num_good_channels, self.query_max_unit_counts
        )

        selection_mask = (signal_timestamps >= interval_start[:, None]) & (
            signal_timestamps < interval_end[:, None]
        )
        local_to_global_map = np.array(self.unit_tokenizer(unit_ids))

        ### prepare latents
        latent_index, latent_timestamps = create_linspace_latent_tokens(
            start,
            self.bin_width,
            step=self.latent_step,
            num_latents_per_step=self.num_latents_per_step,
        )

        if curr_query_max_unit_counts > 0:
            # formulate channel signal per bins for ssl channel signal reconstruction
            query_token_length = curr_query_max_unit_counts
            query_assignment_masks = np.ones(
                (n_intervals, query_token_length)
            )  # not actually used
            query_unit_index_bins = np.zeros_like(query_assignment_masks)
            query_timestamps_bins = np.zeros_like(query_assignment_masks)
            query_token_type_index_bins = np.zeros_like(query_assignment_masks)
            query_assignment_bins = np.zeros_like(query_assignment_masks)
            query_unit_value_bins = np.zeros(
                (n_intervals, query_token_length, self.patch_size)
            )

        if self.fix_qry_unit_sampler:
            rng_sampler = np.random.default_rng(int(data.absolute_start * 1000.0))

        for assignment_idx in np.arange(n_intervals):
            # setup the spike count rate prediction ssl task if needed
            if curr_query_max_unit_counts > 0:
                bin_start = interval_start[assignment_idx]
                bin_end = interval_end[assignment_idx]

                # extract channel units & values within this bin
                # sample from the good channel ids
                bin_mask = selection_mask[assignment_idx]
                bin_mask_len = np.sum(bin_mask)
                unit_index_per_bin = np.unique(signal_channel_ids[bin_mask])
                assert unit_index_per_bin.shape[0] == current_num_good_channels, (
                    unit_index_per_bin.shape[0]
                )
                assert bin_mask_len == current_num_good_channels
                if self.fix_qry_unit_sampler:
                    selected_perm_all_per_group_unit_local_idx = (
                        rng_sampler.permutation(bin_mask_len)[
                            :curr_query_max_unit_counts
                        ]
                    )
                else:
                    selected_perm_all_per_group_unit_local_idx = np.random.permutation(
                        bin_mask_len
                    )[:curr_query_max_unit_counts]
                query_unit_index_per_bin = signal_channel_ids[bin_mask][
                    selected_perm_all_per_group_unit_local_idx
                ]
                query_unit_value_per_bin = signal_values[bin_mask, :][
                    selected_perm_all_per_group_unit_local_idx, :
                ]

                query_unit_index_bins[int(assignment_idx), :] = query_unit_index_per_bin
                query_unit_value_bins[int(assignment_idx), :, :] = (
                    query_unit_value_per_bin
                )
                query_timestamps_bins[int(assignment_idx), :] = (
                    bin_start + bin_end
                ) / 2.0
                query_assignment_bins[int(assignment_idx), :] = assignment_idx + 1
                query_token_type_index_bins[int(assignment_idx), :] = int(
                    TokenType.EXIST_IN_SEQUENCE
                )

        signal_channel_ids = local_to_global_map[signal_channel_ids]

        ## prepare outputs
        session_index = self.session_tokenizer(data.session)

        if curr_query_max_unit_counts > 0:
            query_unit_index = query_unit_index_bins.reshape(-1).astype(np.int64)
            query_unit_index = local_to_global_map[query_unit_index]
            query_timestamps = query_timestamps_bins.reshape(-1).astype(np.float64)
            query_token_type_index = query_token_type_index_bins.reshape(-1).astype(
                np.int64
            )
            query_assignment = query_assignment_bins.reshape(-1).astype(np.int64)
            query_unit_value = (
                query_unit_value_bins.mean(-1).reshape(-1).astype(np.float32)
            )
            output_len = len(query_timestamps)

        if self.include_decoding_outputs:
            (
                output_timestamps,
                output_task_index,
                output_values,
                output_weights,
                output_subtask_index,
            ) = prepare_for_multitask_readout(
                data,
                self.decoder_registry,
            )

            if self.dec_use_last_timestamps:
                assert output_timestamps.shape[0] <= 3  # hack for ecog-syllable task
                output_timestamps[:] = end
            elif self.dec_use_rel_timestamps:
                assert output_timestamps.shape[0] <= 3
                output_timestamps[:] = self.cst_rel_timestamps

            if self.cls_repeated_time:
                num_tasks = output_timestamps.shape[0]

                task_output_timestamps = np.linspace(
                    0.0, end, self.cls_num_repeated_time + 1
                )[1:]
                task_output_timestamps = torch.from_numpy(
                    np.tile(task_output_timestamps, num_tasks)
                ).to(output_timestamps)
                output_timestamps = task_output_timestamps

                output_task_index = np.repeat(
                    output_task_index, self.cls_num_repeated_time, axis=0
                )
                for k in output_values:
                    output_values[k] = np.repeat(
                        output_values[k], self.cls_num_repeated_time, axis=0
                    )
                    output_weights[k] = np.repeat(
                        output_weights[k], self.cls_num_repeated_time, axis=0
                    )
                    if self.eval:
                        output_subtask_index[k] = np.repeat(
                            output_subtask_index[k], self.cls_num_repeated_time, axis=0
                        )

            output_len = len(output_timestamps)
            session_index = np.repeat(session_index, output_len)

        batch = {
            "latent_index_per_bin": latent_index,
            "latent_timestamps_per_bin": latent_timestamps,
            "session_index": pad(session_index),
            "n_intervals": n_intervals,
        }
        batch.update(
            {
                f"spike_unit_index": pad_sequence(
                    torch.nn.utils.rnn.pad_sequence(
                        [
                            torch.as_tensor(
                                signal_channel_ids[selection_mask[i]]
                                if selection_mask[i].any()
                                else np.array([0])
                            )
                            for i in range(n_intervals)
                        ],
                        batch_first=True,
                    )
                ),
                f"spike_timestamps": pad_sequence(
                    torch.nn.utils.rnn.pad_sequence(
                        [
                            torch.as_tensor(
                                signal_timestamps[selection_mask[i]]
                                if selection_mask[i].any()
                                else np.array([0])
                            )
                            for i in range(n_intervals)
                        ],
                        batch_first=True,
                    )
                ),
                f"spike_type": pad_sequence(
                    torch.nn.utils.rnn.pad_sequence(
                        [
                            torch.as_tensor(
                                signal_channel_token_types[selection_mask[i]]
                                if selection_mask[i].any()
                                else np.array([0])
                            )
                            for i in range(n_intervals)
                        ],
                        batch_first=True,
                    )
                ),
                f"input_mask": pad_sequence(
                    torch.nn.utils.rnn.pad_sequence(
                        [
                            torch.ones(
                                max(selection_mask[i].sum(), 1), dtype=torch.bool
                            )
                            for i in range(n_intervals)
                        ],
                        batch_first=True,
                    )
                ),
                f"input_value": pad_sequence(
                    torch.nn.utils.rnn.pad_sequence(
                        [
                            torch.as_tensor(
                                signal_values[selection_mask[i]]
                                if selection_mask[i].any()
                                else np.array([0])
                            )
                            for i in range(n_intervals)
                        ],
                        batch_first=True,
                    )
                ),
            }
        )

        if curr_query_max_unit_counts > 0:
            # ssl phase for pretraining
            batch = {
                **batch,
                "query_unit_index": pad(query_unit_index),
                "query_timestamps": pad(query_timestamps),
                "query_spike_type": pad(query_token_type_index),
                "query_assignment": pad(query_assignment),
                "query_unit_counts": pad(query_unit_value),
                "query_mask": track_mask(query_unit_index),
            }

        if self.include_decoding_outputs:
            # sl phase for downstreaming task
            output_bin_index = (
                output_timestamps[:, None] < torch.from_numpy(interval_end)
            ).to(torch.int).argmax(axis=-1) + 1
            batch.update(
                {
                    "output_timestamps": pad(output_timestamps),
                    "output_bin_index": pad(output_bin_index),
                    "output_decoder_index": pad(output_task_index),
                    "output_values": chain(output_values, allow_missing_keys=True),
                    "output_weights": chain(output_weights, allow_missing_keys=True),
                }
            )

        if self.eval:
            # we will add a few more fields needed for evaluation
            batch["session_id"] = data.session
            batch["absolute_start"] = data.absolute_start
            batch["output_subtask_index"] = chain(
                output_subtask_index, allow_missing_keys=True
            )
        return batch
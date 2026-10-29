import torch
import torch.nn as nn


class Embedding(nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        init_scale: float = 0.02,
        **kwargs,
    ):
        self.init_scale = init_scale
        super().__init__(num_embeddings, embedding_dim, **kwargs)

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        torch.nn.init.normal_(self.weight, mean=0, std=self.init_scale)
        self._fill_padding_idx_with_zero()

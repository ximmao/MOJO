import torch

from .tokenizers import (
    create_linspace_latent_tokens,
    TokenType,
)

from .dir_utils import find_files_by_extension, make_directory


def move_to(data: dict, device):
    for key in data.keys():
        if isinstance(data[key], torch.Tensor):
            data[key] = data[key].to(device)
        elif isinstance(data[key], dict):
            data[key] = move_to(data[key], device)
        elif isinstance(data[key], list):
            if len(data[key]) > 0 and not isinstance(data[key][0], torch.Tensor):
                continue
            data[key] = [v.to(device) for v in data[key]]
    return data

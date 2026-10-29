from typing import Any, Callable, List

import numpy as np

import mojo_kirby


class Compose:
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, data: mojo_kirby.data.Data) -> mojo_kirby.data.Data:
        for transform in self.transforms:
            data = transform(data)
        return data

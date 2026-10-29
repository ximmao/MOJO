import logging
from typing import Optional

import numpy as np

from mojo_kirby.data import Data, RegularTimeSeries, IrregularTimeSeries


class TriangleDistribution:
    def __init__(
        self,
        min_units: int = 20,
        mode_units: int = 100,
        max_units: int = 300,
        tail_right: Optional[int] = None,
        peak: float = 4,
        M: int = 10,
        max_attempts: int = 100,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.min_units = min_units
        self.mode_units = mode_units
        self.max_units = max_units
        self.tail_right = tail_right if tail_right is not None else max_units
        self.peak = peak
        self.M = M
        self.max_attempts = max_attempts

        self.rng = np.random.default_rng(seed=seed)

    def unnormalized_density_function(self, x):
        if x < self.min_units:
            return 0
        if x <= self.mode_units:
            return 1 + (self.peak - 1) * (x - self.min_units) / (
                self.mode_units - self.min_units
            )
        if x <= self.tail_right:
            return self.peak - (self.peak - 1) * (x - self.mode_units) / (
                self.tail_right - self.mode_units
            )
        return 1

    def proposal_distribution(self, x):
        return self.rng.uniform()

    def sample(self, num_units):
        if num_units < self.min_units:
            return num_units

        # uses rejection sampling
        num_attempts = 0
        while True:
            x = self.min_units + self.rng.uniform() * (
                self.max_units - self.min_units
            )  # Sample from the proposal distribution
            u = self.rng.uniform()
            if u <= self.unnormalized_density_function(x) / (
                self.M * self.proposal_distribution(x)
            ):
                return x
            num_attempts += 1
            if num_attempts > self.max_attempts:
                logging.warning(
                    f"Could not sample from distribution after {num_attempts} attempts,"
                    " using all units."
                )
                return num_units


class UnitDropout:
    def __init__(self, field: str = "spikes", reset_index=True, *args, **kwargs):
        self.field = field
        self.reset_index = reset_index
        self.distribution = TriangleDistribution(*args, **kwargs)

    def __call__(self, data: Data):
        # get units from data
        unit_ids = data.units.id
        num_units = len(unit_ids)

        # sample the number of units to keep from the population
        num_units_to_sample = int(self.distribution.sample(num_units))

        # shuffle units and take the first num_units_to_sample
        keep_indices = np.random.permutation(num_units)[:num_units_to_sample]

        unit_mask = np.zeros_like(unit_ids, dtype=bool)
        unit_mask[keep_indices] = True
        if self.reset_index:
            data.units = data.units.select_by_mask(unit_mask)

        nested_attr = self.field.split(".")
        target_obj = getattr(data, nested_attr[0])
        if isinstance(target_obj, IrregularTimeSeries):
            # make a mask to select spikes that are from the units we want to keep
            spike_mask = np.isin(target_obj.unit_index, keep_indices)

            # using lazy masking, we will apply the mask for all attributes from spikes
            # and units.
            setattr(data, self.field, target_obj.select_by_mask(spike_mask))

            if self.reset_index:
                relabel_map = np.zeros(num_units, dtype=int)
                relabel_map[unit_mask] = np.arange(unit_mask.sum())

                target_obj = getattr(data, self.field)
                target_obj.unit_index = relabel_map[target_obj.unit_index]
        elif isinstance(target_obj, RegularTimeSeries):
            assert len(nested_attr) == 2
            setattr(
                target_obj,
                nested_attr[1],
                getattr(target_obj, nested_attr[1])[:, unit_mask],
            )
        else:
            raise ValueError(f"Unsupported type for {self.field}: {type(target_obj)}")

        return data

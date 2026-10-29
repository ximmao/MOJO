import numpy as np
from einops import repeat


from mojo_kirby.taxonomy.core import StringIntEnum


class TokenType(StringIntEnum):
    DEFAULT = 0
    START_OF_SEQUENCE = 1
    END_OF_SEQUENCE = 2
    EXIST_IN_SEQUENCE = 3
    CONSIDER_REGION = 4


def create_linspace_latent_tokens(start, end, step, num_latents_per_step):
    sequence_len = end - start
    latent_timestamps = np.arange(0, sequence_len, step) + step / 2 + start
    latent_index = np.arange(num_latents_per_step, dtype=np.int64)

    num_timestamps = len(latent_timestamps)
    latent_timestamps = repeat(latent_timestamps, "t -> (t u)", u=len(latent_index))

    latent_index = repeat(latent_index, "u -> (t u)", t=num_timestamps)
    return latent_index, latent_timestamps


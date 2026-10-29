from typing import Dict, List, Tuple, Optional, Union, Any

from pydantic.dataclasses import dataclass

from .core import StringIntEnum
from mojo_kirby.taxonomy.writing import Character, Line
from mojo_kirby.taxonomy.speech import CVSyllable, Consonant, Vowel, Phoneme


class OutputType(StringIntEnum):
    CONTINUOUS = 0
    BINARY = 1
    MULTILABEL = 2
    MULTINOMIAL = 3
    POISSON = 4


class Decoder(StringIntEnum):
    NA = 0
    # Classic BCI outputs.
    ARMVELOCITY2D = 1
    CURSORPOSITION2D = 2

    # Shenoy handwriting style outputs.
    WRITING_CHARACTER = 5
    WRITING_LINE = 6

    CURSORVELOCITY2D = 9

    # Allen data
    DRIFTING_GRATINGS = 13
    DRIFTING_GRATINGS_TEMP_FREQ = 23
    NATURAL_SCENES = 28

    # speech
    SPEAKING_CVSYLLABLE = 51
    SPEAKING_CONSONANT = 52
    SPEAKING_VOWEL = 53
    SPEAKING_SENTENCE = 55

    # ibl
    CHOICE = 31
    BLOCK = 32
    WHEEL = 33
    WHISKER = 34


@dataclass
class DecoderSpec:
    dim: int
    type: OutputType
    loss_fn: str
    timestamp_key: str
    value_key: str
    # Optional fields
    task_key: Optional[str] = None
    subtask_key: Optional[str] = None
    # target_dtype: str = "float32"  # torch.dtype is not serializable.
    task_weight: Optional[float] = 1.0


decoder_registry = {
    str(Decoder.ARMVELOCITY2D): DecoderSpec(
        dim=2,
        target_dim=2,
        type=OutputType.CONTINUOUS,
        timestamp_key="behavior.timestamps",
        value_key="behavior.hand_vel",
        subtask_key="behavior.subtask_index",
        loss_fn="mse",
    ),
    str(Decoder.CURSORVELOCITY2D): DecoderSpec(
        dim=2,
        target_dim=2,
        type=OutputType.CONTINUOUS,
        timestamp_key="cursor.timestamps",
        value_key="cursor.vel",
        subtask_key="cursor.subtask_index",
        loss_fn="mse",
    ),
    str(Decoder.CURSORPOSITION2D): DecoderSpec(
        dim=2,
        target_dim=2,
        type=OutputType.CONTINUOUS,
        timestamp_key="cursor.timestamps",
        value_key="cursor.pos",
        subtask_key="cursor.subtask_index",
        loss_fn="mse",
    ),
    str(Decoder.WRITING_CHARACTER): DecoderSpec(
        dim=len(Character),
        target_dim=1,
        target_dtype="long",
        type=OutputType.MULTINOMIAL,
        timestamp_key="stimuli.timestamps",
        value_key="stimuli.letters",
        loss_fn="bce",
    ),
    str(Decoder.WRITING_LINE): DecoderSpec(
        dim=len(Line),
        target_dim=1,
        target_dtype="long",
        type=OutputType.MULTINOMIAL,
        timestamp_key="stimuli.timestamps",
        value_key="stimuli.letters",
        loss_fn="bce",
    ),
    str(Decoder.DRIFTING_GRATINGS): DecoderSpec(
        dim=8,
        target_dim=1,
        target_dtype="long",
        type=OutputType.MULTINOMIAL,
        timestamp_key="drifting_gratings.timestamps",
        value_key="drifting_gratings.orientation",
        loss_fn="bce",
    ),
    str(Decoder.DRIFTING_GRATINGS_TEMP_FREQ): DecoderSpec(
        dim=5,  # [1,2,4,8,15]
        target_dim=1,
        target_dtype="long",
        type=OutputType.MULTINOMIAL,
        timestamp_key="drifting_gratings.timestamps",
        value_key="drifting_gratings.temp_freq",
        loss_fn="bce",
    ),
    str(Decoder.SPEAKING_CVSYLLABLE): DecoderSpec(
        dim=len(CVSyllable),
        target_dim=1,
        target_dtype="long",
        type=OutputType.MULTINOMIAL,
        timestamp_key="syllables.timestamps",
        value_key="syllables.cvsyllables",
        loss_fn="bce",
    ),
    str(Decoder.SPEAKING_CONSONANT): DecoderSpec(
        dim=len(Consonant),
        target_dim=1,
        target_dtype="long",
        type=OutputType.MULTINOMIAL,
        timestamp_key="syllables.timestamps",
        value_key="syllables.consonants",
        loss_fn="bce",
    ),
    str(Decoder.SPEAKING_VOWEL): DecoderSpec(
        dim=len(Vowel),
        target_dim=1,
        target_dtype="long",
        type=OutputType.MULTINOMIAL,
        timestamp_key="syllables.timestamps",
        value_key="syllables.vowels",
        loss_fn="bce",
    ),
    str(Decoder.SPEAKING_SENTENCE): DecoderSpec(
        dim=len(Phoneme),  # empty label is included
        target_dim=1,
        target_dtype="long",
        type=OutputType.MULTINOMIAL,
        timestamp_key="speech.timestamps",
        value_key="speech.phonemes",
        loss_fn="ctc",
    ),
    str(Decoder.NATURAL_SCENES): DecoderSpec(
        dim=119,  # image classes [0,...,118]
        target_dim=1,
        target_dtype="long",
        type=OutputType.MULTINOMIAL,
        timestamp_key="natural_scenes.timestamps",
        value_key="natural_scenes.image_ids",
        loss_fn="bce",
        task_weight=0.3,
    ),
    str(Decoder.CHOICE): DecoderSpec(
        dim=2,
        target_dim=1,
        target_dtype="long",
        type=OutputType.MULTINOMIAL,
        timestamp_key="choice_repeated.timestamps",
        value_key="choice_repeated.label",
        subtask_key="choice_repeated.subtask_index",
        loss_fn="bce",
    ),
    str(Decoder.BLOCK): DecoderSpec(
        dim=3,
        target_dim=1,
        target_dtype="long",
        type=OutputType.MULTINOMIAL,
        timestamp_key="block_repeated.timestamps",
        value_key="block_repeated.label",
        subtask_key="block_repeated.subtask_index",
        loss_fn="bce",
    ),
    str(Decoder.WHEEL): DecoderSpec(
        dim=1,
        target_dim=1,
        type=OutputType.CONTINUOUS,
        timestamp_key="wheel_interpolated.timestamps",
        value_key="wheel_interpolated.speed",
        subtask_key="wheel_interpolated.subtask_index",
        loss_fn="mse",
    ),
    str(Decoder.WHISKER): DecoderSpec(
        dim=1,
        target_dim=1,
        type=OutputType.CONTINUOUS,
        timestamp_key="whisker_interpolated.timestamps",
        value_key="whisker_interpolated.motion_energy",
        subtask_key="whisker_interpolated.subtask_index",
        loss_fn="mse",
    ),
}

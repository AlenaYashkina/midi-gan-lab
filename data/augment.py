import random
from copy import deepcopy

from pretty_midi import PrettyMIDI
from config import (TRANSPOSE_RANGE,
                    VELOCITY_RANGE,
                    _TRANSPOSE_PADDING,
                    _SEMITONE_LIMITS,
                    _VELOCITY_PADDING,
                    _VELOCITY_LIMITS)
from utils.logger import logger


def augment_midi(
    midi: PrettyMIDI,
    transpose_range: int = TRANSPOSE_RANGE,
    velocity_range: int = VELOCITY_RANGE,
) -> PrettyMIDI:
    try:
        midi_aug = deepcopy(midi)

        semitone_min = -transpose_range - _TRANSPOSE_PADDING
        semitone_max = transpose_range + _TRANSPOSE_PADDING
        semitone_shift = random.randint(semitone_min, semitone_max)
        logger.debug("Applying semitone shift: %d", semitone_shift)

        for instrument in midi_aug.instruments:
            for note in instrument.notes:
                orig_pitch = note.pitch
                note.pitch = _clamp(orig_pitch + semitone_shift, *_SEMITONE_LIMITS)
                logger.debug(
                    "Pitch %d -> %d", orig_pitch, note.pitch
                )

                vel_min = -velocity_range - _VELOCITY_PADDING
                vel_max = velocity_range + _VELOCITY_PADDING
                jitter = random.randint(vel_min, vel_max)
                orig_vel = note.velocity
                note.velocity = _clamp(orig_vel + jitter, *_VELOCITY_LIMITS)
                logger.debug(
                    "Velocity %d -> %d", orig_vel, note.velocity
                )

        return midi_aug

    except Exception:
        logger.exception("Error while augmenting MIDI")
        raise


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))

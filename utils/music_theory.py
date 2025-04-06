import logging
from typing import Iterable, Optional, Tuple

from config import MODES

logger = logging.getLogger(__name__)


def determine_tonality(notes: Iterable[int]) -> Tuple[Optional[int], Optional[str]]:
    note_list = list(notes)
    if not note_list:
        logger.debug("No notes provided; returning (None, None).")
        return None, None

    pitch_classes = [n % 12 for n in note_list]

    best_key: Optional[int] = None
    best_mode: Optional[str] = None
    best_count = -1

    for key in range(12):
        for mode_name, intervals in MODES.items():
            scale = {(key + interval) % 12 for interval in intervals}
            count = sum(pc in scale for pc in pitch_classes)
            if count > best_count:
                best_count = count
                best_key = key
                best_mode = mode_name

    logger.debug(
        "Determined tonality: key=%s, mode=%s, matched %d/%d notes",
        best_key, best_mode, best_count, len(pitch_classes)
    )
    return best_key, best_mode

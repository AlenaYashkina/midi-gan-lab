from pathlib import Path
from typing import List, Union

import pretty_midi
from tqdm import tqdm

from utils.logger import logger


def load_midi_files(folder_path: Union[str, Path]) -> List[pretty_midi.PrettyMIDI]:
    path = Path(folder_path)
    if not path.is_dir():
        logger.error("Not a directory: %s", path)
        raise NotADirectoryError(f"{path} is not a directory")

    midi_paths = sorted(
        p for p in path.iterdir() if p.suffix.lower() in (".mid", ".midi")
    )
    if not midi_paths:
        logger.warning("No MIDI files found in %s", path)
        return []

    midi_list: List[pretty_midi.PrettyMIDI] = []
    for midi_path in tqdm(midi_paths, desc="Loading MIDI files"):
        try:
            midi = pretty_midi.PrettyMIDI(midi_path.as_posix())
            midi_list.append(midi)
            logger.debug("Loaded MIDI: %s", midi_path.name)
        except (IOError, ValueError) as exc:
            logger.warning("Failed to load %s: %s", midi_path.name, exc)

    logger.info("Successfully loaded %d MIDI files from %s", len(midi_list), path)
    return midi_list

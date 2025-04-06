from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pretty_midi
from tqdm import tqdm

from config import (
    DURATION_THRESHOLDS,
    TRANSPOSE_RANGE,
    VELOCITY_THRESHOLDS,
    TICKS_PER_STEP,
    MASK_PROBABILITY,
    REST_TOKEN, MAX_TRACKS, SEQ_LEN
)
from utils.logger import logger
from utils.utils import fill_rest

STRUCTURE_LABELS: List[str] = ["intro", "verse", "chorus", "bridge", "breakdown"]

ROLE_RANGES: Dict[str, Tuple[int, int]] = {
    "bass": (32, 39),
    "guitar": (24, 31),
    "piano": (0, 7),
    "strings": (48, 55),
    "pad": (88, 95),
}


def _quantize(value: int, thresholds: List[int]) -> int:
    for idx, threshold in enumerate(thresholds):
        if value <= threshold * 1.1:
            return idx
    return len(thresholds) - 1


def _get_track_name(instr: pretty_midi.Instrument) -> str:
    return instr.name.strip() if instr.name else f"Program {instr.program}"


def _get_role(instr: pretty_midi.Instrument) -> str:
    if instr.is_drum:
        return "drums"
    for role, (low, high) in ROLE_RANGES.items():
        if low <= instr.program <= high:
            return role
    return "other"


def build_vocab(
    midi_list: List[pretty_midi.PrettyMIDI],
    save_path: Union[str, Path],
) -> Dict[str, Any]:
    pitch_set: set[int] = set()
    velocity_set: set[int] = set()
    duration_set: set[int] = set()
    program_set: set[int] = set()
    drum_set: set[int] = set()
    track_names: set[str] = set()
    instrument_counts: Dict[int, int] = {}

    for midi in tqdm(midi_list, desc="Building vocabulary"):
        seen: set[str] = set()
        for instr in midi.instruments:
            name = _get_track_name(instr).lower()
            if name in seen:
                continue
            seen.add(name)
            track_names.add(name)

            prog = 128 if instr.is_drum else instr.program
            program_set.add(prog)
            if instr.is_drum:
                drum_set.add(prog)
            instrument_counts[prog] = instrument_counts.get(prog, 0) + 1

            for note in instr.notes:
                pitch_set.add(note.pitch)
                pitch_set.update(
                    p
                    for p in range(
                        note.pitch - TRANSPOSE_RANGE,
                        note.pitch + TRANSPOSE_RANGE + 1,
                    )
                    if 0 <= p <= 127
                )
                duration_set.add(int((note.end - note.start) * midi.resolution))
                velocity_set.add(note.velocity)

    programs = sorted(program_set)
    drums = sorted(drum_set)

    role2idx = {
        "drums": 0,
        "bass": 1,
        "guitar": 2,
        "piano": 3,
        "strings": 4,
        "pad": 5,
        "other": 6,
    }

    vocab: Dict[str, Any] = {
        "pitch2idx": {p: i for i, p in enumerate(sorted(pitch_set))},
        "velocity2idx": {v: i for i, v in enumerate(sorted(velocity_set))},
        "duration2idx": {d: i for i, d in enumerate(sorted(duration_set))},
        "track2idx": {t: i for i, t in enumerate(sorted(track_names))},
        "instrument2idx": {p: i for i, p in enumerate(programs)},
        "idx2instrument": {i: p for i, p in enumerate(programs)},
        "drum_programs": drums,
        "instrument_freqs": instrument_counts,
        "key_sig2idx": {},
        "role2idx": role2idx,
        "idx2role": {i: r for r, i in role2idx.items()},
    }

    if REST_TOKEN not in vocab["pitch2idx"]:
        vocab["pitch2idx"][REST_TOKEN] = len(vocab["pitch2idx"])

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with save_path.open("wb") as fp:
            pickle.dump(vocab, fp)
        logger.info("Vocabulary saved to %s", save_path)
    except Exception as exc:
        logger.exception("Failed to save vocabulary to %s: %s", save_path, exc)
        raise

    return vocab


def extract_note_matrix(
    midi: pretty_midi.PrettyMIDI,
    vocab: Dict[str, Any],
    max_tracks: int | None = None,
    max_measures: int | None = None,
    measure_length: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tick_per_quarter = midi.resolution // 4
    if max_measures and measure_length:
        steps_per_measure = measure_length // tick_per_quarter
        total_steps = max_measures * steps_per_measure
    else:
        steps_per_measure = None
        total_steps = int(midi.get_end_time() * (midi.resolution / TICKS_PER_STEP))

    if max_tracks is None:
        max_tracks = len(midi.instruments)

    feature_sizes = [
        len(vocab["pitch2idx"]),
        len(vocab["duration2idx"]),
        len(vocab["velocity2idx"]),
        len(vocab["instrument2idx"]),
        len(vocab["role2idx"]),
    ]
    n_channels = len(feature_sizes)
    tensor = np.zeros(
        (max_tracks, n_channels, max(feature_sizes), total_steps),
        dtype=np.uint8,
    )

    segment_ids = np.zeros(total_steps, dtype=np.int64)
    structure_ids = np.zeros(total_steps, dtype=np.int64)
    section_len = total_steps // len(STRUCTURE_LABELS)
    for idx, _ in enumerate(STRUCTURE_LABELS):
        start = idx * section_len
        end = (idx + 1) * section_len if idx < len(STRUCTURE_LABELS) - 1 else total_steps
        segment_ids[start:end] = idx
        structure_ids[start:end] = idx

    for track_idx, instr in enumerate(midi.instruments[:max_tracks]):
        prog = 128 if instr.is_drum else instr.program
        role_idx = vocab["role2idx"].get(_get_role(instr), 0)

        for note in instr.notes:
            if steps_per_measure is not None:
                start_tick = int(note.start * midi.resolution)
                measure_idx = start_tick // measure_length
                if measure_idx >= max_measures:
                    continue
                local_tick = start_tick % measure_length
                time_idx = measure_idx * steps_per_measure + (local_tick // tick_per_quarter)
            else:
                time_idx = min(
                    int((note.start * midi.resolution) // TICKS_PER_STEP),
                    total_steps - 1,
                )

            _write_note_to_tensor(
                tensor, vocab, prog, role_idx, note, track_idx, time_idx, midi.resolution
            )

    fill_rest(tensor, vocab, max_tracks, total_steps)
    mask = np.random.rand(*tensor.shape) < MASK_PROBABILITY
    tensor[mask] = 0

    if tensor.shape[0] > MAX_TRACKS:
        tensor = tensor[:MAX_TRACKS]
    elif tensor.shape[0] < MAX_TRACKS:
        pad_shape = (MAX_TRACKS - tensor.shape[0],) + tensor.shape[1:]
        tensor = np.concatenate([tensor, np.zeros(pad_shape, dtype=tensor.dtype)], axis=0)

    if tensor.shape[-1] > SEQ_LEN:
        tensor = tensor[..., :SEQ_LEN]
    elif tensor.shape[-1] < SEQ_LEN:
        pad_width = [(0, 0)] * (tensor.ndim - 1) + [(0, SEQ_LEN - tensor.shape[-1])]
        tensor = np.pad(tensor, pad_width, mode="constant")

    return tensor, segment_ids, structure_ids



def _write_note_to_tensor(
    tensor: np.ndarray,
    vocab: Dict[str, Any],
    program: int,
    role_idx: int,
    note: pretty_midi.Note,
    track_idx: int,
    time_idx: int,
    midi_resolution: int,
) -> None:
    pitch_map = vocab["pitch2idx"]
    if note.pitch not in pitch_map:
        logger.debug("Skipping unknown pitch %d", note.pitch)
        return

    p_idx = pitch_map[note.pitch]
    tensor[track_idx, 0, p_idx, time_idx] = 1

    dur_ticks = int((note.end - note.start) * midi_resolution)
    dur_bucket = _quantize(dur_ticks, DURATION_THRESHOLDS)
    for offset in range(1, dur_bucket):
        if time_idx + offset < tensor.shape[-1]:
            tensor[track_idx, 0, p_idx, time_idx + offset] = 1

    vel_bucket = _quantize(note.velocity, VELOCITY_THRESHOLDS)
    tensor[track_idx, 2, p_idx, time_idx] = vel_bucket

    instr_idx = vocab["instrument2idx"].get(program, 0)
    tensor[track_idx, 3, instr_idx, time_idx] = 1

    tensor[track_idx, 4, role_idx, time_idx] = 1

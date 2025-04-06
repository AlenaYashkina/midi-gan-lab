import json
import random
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pretty_midi
import torch

from config import (
    GM_PROGRAMS,
    TIME_STEP,
    duration_map,
    velocity_map,
    UNISON_THRESHOLD,
    MIN_NOTES_THRESHOLD,
)
from utils.logger import logger


def convert_tensor_to_midi(
    tensor: torch.Tensor,
    vocab: Dict[str, Any],
    out_dir: Union[str, Path],
) -> pretty_midi.PrettyMIDI:
    logger.debug("convert_tensor_to_midi — tensor shape: %s", tuple(tensor.shape))
    tensor = tensor.clone().cpu()
    if tensor.ndim != 4:
        raise ValueError(f"Expected 4D tensor, got {tensor.ndim}D")

    midi = pretty_midi.PrettyMIDI()
    n_tracks, _, _, n_steps = tensor.shape

    idx2instr = vocab.get("idx2instrument")
    if not isinstance(idx2instr, dict):
        raise KeyError("vocab['idx2instrument'] must be a dict[int, int]")

    used_progs: set = set()
    all_progs = sorted(set(idx2instr.values()) | {128})
    seen_maps: List[Tuple[np.ndarray, int]] = []

    for tr in range(n_tracks):
        instr_layer = tensor[tr, 3]
        instr_layer.sum(dim=1).numpy()
        counts = instr_layer.sum(dim=1).numpy()
        prog_idxs = list(idx2instr.keys())
        valid_prog_idxs = [i for i in prog_idxs if i < len(counts)]
        if not valid_prog_idxs:
            logger.debug("Track %d skipped: no valid program indices.", tr)
            continue

        subcounts = counts[valid_prog_idxs]
        prog_idx = valid_prog_idxs[int(subcounts.argmax())]

        prog_num = idx2instr[prog_idx]

        pitch_mask = tensor[tr, 0].numpy() > 0.1
        dur_probs = tensor[tr, 1].numpy()
        vel_probs = tensor[tr, 2].numpy()

        fill_rate = pitch_mask.sum() / pitch_mask.size
        if fill_rate == 0:
            logger.info("Track %d skipped: empty", tr)
            continue

        is_unison = False
        if 24 <= prog_num <= 40:
            for prev_map, prev_prog in seen_maps:
                sim = np.mean(pitch_mask == prev_map)
                if sim > UNISON_THRESHOLD:
                    logger.info(
                        "Track %d prog %d skipped (unison with prog %d, sim=%.2f)",
                        tr, prog_num, prev_prog, sim,
                    )
                    is_unison = True
                    break
        if is_unison:
            continue

        seen_maps.append((pitch_mask, prog_num))
        if prog_num in used_progs:
            for alt in all_progs:
                if alt not in used_progs:
                    prog_num = alt
                    break
        used_progs.add(prog_num)

        inst = pretty_midi.Instrument(
            program=0 if prog_num == 128 else prog_num,
            is_drum=(prog_num == 128),
            name=f"{GM_PROGRAMS.get(prog_num, 'Unknown')} ({prog_num})",
        )

        note_count = 0
        for pitch in range(pitch_mask.shape[0]):
            for ts in range(n_steps):
                if not pitch_mask[pitch, ts]:
                    continue
                start = ts * TIME_STEP
                dur_idx = int(dur_probs[pitch].argmax())
                velocity_idx = int(vel_probs[pitch].argmax())
                duration = duration_map.get(dur_idx, TIME_STEP)
                velocity = velocity_map.get(velocity_idx, 64)

                note = pretty_midi.Note(
                    velocity=velocity,
                    pitch=pitch,
                    start=start + random.uniform(-0.02, 0.02),
                    end=start + duration + random.uniform(-0.02, 0.02),
                )
                inst.notes.append(note)
                note_count += 1

        if (24 <= prog_num <= 31 and note_count == 0) or (0 < note_count < MIN_NOTES_THRESHOLD):
            logger.info("Track %d skipped: note_count=%d", tr, note_count)
            continue

        midi.instruments.append(inst)
        logger.debug("Added track %d prog %d (%d notes)", tr, prog_num, note_count)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    meta = {
        "uuid": uuid.uuid4().hex,
        "tracks": len(midi.instruments),
        "programs": [ins.program for ins in midi.instruments],
        "names": [ins.name for ins in midi.instruments],
    }
    meta_file = out_path / f"meta_{meta['uuid']}.json"
    try:
        meta_file.write_text(json.dumps(meta, indent=2))
        logger.info("Saved metadata to %s", meta_file)
    except Exception as exc:
        logger.exception("Failed to save metadata: %s", exc)

    return midi


def save_midi(midi: pretty_midi.PrettyMIDI, path: Union[str, Path]) -> None:
    out_file = Path(path)
    try:
        midi.write(out_file.as_posix())
        logger.info("MIDI saved to %s", out_file)
    except Exception as exc:
        logger.exception("Failed to save MIDI to %s: %s", out_file, exc)

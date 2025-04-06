from pathlib import Path
import pickle
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from data.augment import augment_midi
from data.midi_loader import load_midi_files
from data.preprocess import extract_note_matrix
from utils.logger import logger


def _get_cache_path(cache_dir: Path, midi_path: Path) -> Path:
    return cache_dir / f"{midi_path.stem}.npy"


def _extract_program_ids(
    midi: Any, vocab: Dict[str, Any]
) -> torch.Tensor:
    program_ids: List[int] = []
    for instr in midi.instruments:
        prog = 128 if instr.is_drum else instr.program
        program_ids.append(vocab["instrument2idx"].get(prog, 0))
    return torch.tensor(program_ids, dtype=torch.long)


class MusicDataset(Dataset):
    def __init__(
        self,
        midi_folder: Optional[Union[str, Path]] = None,
        vocab_path: Optional[Union[str, Path]] = None,
        config_module: Any = None,
        midi_list: Optional[List[Any]] = None,
    ) -> None:
        super().__init__()
        if config_module is None:
            raise ValueError("config_module must be provided")

        self.cache_dir = Path(config_module.CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if not vocab_path:
            raise ValueError("vocab_path must be provided")
        vp = Path(vocab_path)
        if not vp.exists():
            logger.error("Vocab file not found at %s", vp)
            raise FileNotFoundError(f"Vocab not found: {vp}")
        try:
            with vp.open("rb") as f:
                self.vocab: Dict[str, Any] = pickle.load(f)
            logger.info("Loaded vocab from %s", vp)
        except (OSError, pickle.UnpicklingError) as exc:
            logger.error("Failed to load vocab: %s", exc)
            raise

        if midi_list is None:
            if midi_folder is None:
                raise ValueError("Either midi_list or midi_folder must be provided")
            self.midi_folder = Path(midi_folder)
            midi_list = load_midi_files(self.midi_folder)
            logger.info("Found %d MIDI files in %s", len(midi_list), self.midi_folder)

        self.max_measures = config_module.MAX_MEASURES
        self.measure_length = config_module.MEASURE_LENGTH
        self.segment_map = config_module.SEGMENT_MAP

        self.samples: List[Dict[str, torch.Tensor]] = []
        for midi in tqdm(midi_list, desc="Building dataset"):
            fname_attr = getattr(midi, "filename", None)
            midi_path = Path(fname_attr) if fname_attr else None
            fname = midi_path.name if midi_path else "<unknown>"

            cache_path = (
                _get_cache_path(self.cache_dir, midi_path)
                if midi_path
                else None
            )

            tensor: Optional[np.ndarray] = None
            segment_ids: Optional[np.ndarray] = None

            if cache_path and cache_path.exists():
                try:
                    tensor = np.load(str(cache_path))
                    logger.debug("Loaded tensor from cache %s", cache_path.name)
                except OSError as exc:
                    logger.warning("Cache read failed %s: %s", cache_path.name, exc)

            if tensor is not None:
                total = tensor.shape[-1]
                segment_ids = np.zeros(total, dtype=np.int64)
                seg_len = total // len(self.segment_map)
                for i, seg_label in enumerate(self.segment_map):
                    start = i * seg_len
                    end = (i + 1) * seg_len if i < len(self.segment_map) - 1 else total
                    segment_ids[start:end] = seg_label

            if tensor is None:
                try:
                    midi_aug = augment_midi(midi)
                    tensor, segment_ids, _ = extract_note_matrix(
                        midi_aug,
                        self.vocab,
                        max_tracks=len(midi_aug.instruments),
                        max_measures=self.max_measures,
                        measure_length=self.measure_length,
                    )
                    if cache_path:
                        try:
                            np.save(str(cache_path), tensor.astype(np.uint8))
                            logger.debug("Saved tensor to cache %s", cache_path.name)
                        except OSError as exc:
                            logger.warning("Cache write failed %s: %s", cache_path.name, exc)
                except Exception as exc:
                    logger.exception("Error processing MIDI %s: %s", fname, exc)
                    continue

            if tensor[:, 0, :, :].sum() == 0:
                continue

            prog_ids = _extract_program_ids(midi, self.vocab)

            self.samples.append({
                "tensor": torch.tensor(tensor, dtype=torch.uint8),  # <--- вот тут
                "program_ids": prog_ids,
                "segment_ids": torch.tensor(segment_ids, dtype=torch.long),
            })

        logger.info("Dataset ready with %d samples", len(self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        tensor = sample["tensor"]

        # Приводим к float32, если вдруг он сохранён как uint8
        if tensor.dtype != torch.float32:
            tensor = tensor.to(dtype=torch.float32)

        assert tensor.shape[0] == 4, f"Expected 4 tracks, got {tensor.shape[0]}"
        assert tensor.shape[-1] == 512, f"Expected seq_len=512, got {tensor.shape[-1]}"

        sample["tensor"] = tensor
        return sample

import multiprocessing as mp
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config
from config import LR, MIDI_FOLDER, RANDOM_SEED, VOCAB_PATH, SEQ_LEN
from data.dataset import MusicDataset
from data.midi_loader import load_midi_files
from data.preprocess import build_vocab
from training.build_gan import build_gan
from training.trainer import Trainer
from utils.logger import logger, init_tb_logger
from utils.utils import set_random_seed

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.autograd.set_detect_anomaly(True)


def _custom_collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    SEQ_LEN = 512

    data_list = [item["tensor"] for item in batch]
    prog_list = [item["program_ids"] for item in batch]
    seg_list = [item["segment_ids"] for item in batch]

    max_tracks = max(d.shape[0] for d in data_list)
    padded_data = []

    for d in data_list:
        t, c, f, tm = d.shape  # [tracks, channels, freq, time]
        pad_tracks = max(0, max_tracks - t)
        pad_T = max(0, SEQ_LEN - f)  # ❗ если ось времени — это freq (f)
        arr = np.pad(
            d.numpy(),
            ((0, pad_tracks), (0, 0), (0, pad_T), (0, 0)),  # ⬅ паддинг по f (ось 2)
            mode="constant",
            constant_values=0,
        )
        padded_data.append(torch.from_numpy(arr))
    data_tensor = torch.stack(padded_data)

    padded_prog = []
    for p in prog_list:
        pad_tracks = max(0, max_tracks - p.shape[0])
        padded_prog.append(F.pad(p, (0, pad_tracks), value=0))
    prog_tensor = torch.stack(
        [p[:4] if p.shape[0] > 4 else F.pad(p, (0, 0), value=0, mode='constant') for p in padded_prog])

    padded_seg = []
    for s in seg_list:
        pad_time = max(0, SEQ_LEN - s.shape[0])
        padded_seg.append(F.pad(s, (0, pad_time), value=0))
    seg_tensor = torch.stack(padded_seg)

    return {
        "tensor": data_tensor,
        "program_ids": prog_tensor,
        "segment_ids": seg_tensor,
    }


def main() -> None:
    mp.set_start_method("spawn", force=True)
    set_random_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    midi_folder = Path(MIDI_FOLDER)
    if not midi_folder.is_dir():
        logger.error("MIDI folder not found: %s", midi_folder)
        return

    try:
        midi_list = load_midi_files(midi_folder)
    except Exception as exc:
        logger.exception("Failed to load MIDI files: %s", exc)
        return

    vocab_path = Path(VOCAB_PATH)
    if not vocab_path.exists():
        logger.info("Building vocabulary at %s", vocab_path)
        try:
            build_vocab(midi_list, vocab_path)
        except Exception as exc:
            logger.exception("Failed to build vocab: %s", exc)
            return

    dataset = MusicDataset(
        midi_folder=midi_folder,
        vocab_path=vocab_path,
        config_module=config,
        midi_list=midi_list,
    )
    if len(dataset) == 0:
        logger.error("Empty dataset after preprocessing; aborting.")
        return

    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        collate_fn=_custom_collate,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    gan = build_gan(config, dataset, LR, device)
    writer = init_tb_logger(config.TENSORBOARD_LOGDIR)
    hparam_dict = {
        "lr": config.LR,
        "batch_size": config.BATCH_SIZE,
        "embed_dim": config.EMBED_DIM,
        "z_dim": config.Z_DIM,
        "loss_kl": config.LOSS_WEIGHT_KL,
        "loss_instr": config.LOSS_WEIGHT_INSTRUMENT,
    }
    writer.add_hparams(hparam_dict, metric_dict={})
    trainer = Trainer(gan, loader, device, config, writer=writer)

    try:
        torch.autograd.set_detect_anomaly(True)
        trainer.train()
    except Exception as exc:
        logger.exception("Training failed: %s", exc)


if __name__ == "__main__":
    main()

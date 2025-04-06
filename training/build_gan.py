import torch
import torch.nn as nn
from torch.optim import AdamW
from typing import Any

from config import EMBED_DIM
from models.discriminator import Discriminator
from models.generator import Generator
from models.rhythm_discriminator import RhythmDiscriminator
from models.gan import GANWrapper
from models.transformer import Transformer
from training.losses import InstrumentClassifier
from utils.logger import logger


def build_gan(
    cfg: Any, dataset: Any, lr: float, device: torch.device
) -> GANWrapper:
    try:
        raw_sample = dataset.dataset[0] if hasattr(dataset, "dataset") else dataset[0]
    except Exception:
        raw_sample = dataset[0]

    if not isinstance(raw_sample, dict) or "tensor" not in raw_sample:
        raise KeyError(f"Invalid dataset sample: {type(raw_sample)}. Expected dict with 'tensor' key.")

    data_tensor = raw_sample["tensor"]
    n_pitches = data_tensor.shape[2]
    n_tracks = data_tensor.shape[0]
    n_time_steps = data_tensor.shape[-1]
    n_instr = len(dataset.vocab["instrument2idx"])

    transformer = Transformer(embed_dim=EMBED_DIM, num_tokens=512)

    segment_embedding = nn.Embedding(cfg.NUM_SEGMENTS, cfg.SEGMENT_DIM)
    instrument_embedding = nn.Embedding(n_instr, cfg.EMBED_DIM)
    time_embedding = nn.Embedding(512, cfg.EMBED_DIM)

    G = Generator(
        transformer=transformer,
        segment_embedding=segment_embedding,
        instrument_embedding=instrument_embedding,
        time_embedding=time_embedding,
        embedding_dim=cfg.EMBED_DIM,
    ).to(device)

    D = Discriminator(
        n_tracks=n_tracks,
        n_pitches=n_pitches,
        n_time_steps=n_time_steps,
        device=device
    )

    R_D = RhythmDiscriminator(
        n_tracks=n_tracks,
        n_time_steps=n_time_steps,
        device=device
    )

    instrument_classifier = InstrumentClassifier(num_instruments=9)

    g_opt = AdamW(G.parameters(), lr=lr, weight_decay=cfg.WEIGHT_DECAY_G)
    d_opt = AdamW(
        D.parameters(),
        lr=lr * cfg.D_LR_MULTIPLIER,
        weight_decay=cfg.WEIGHT_DECAY_D,
    )
    r_opt = AdamW(R_D.parameters(), lr=lr, weight_decay=cfg.WEIGHT_DECAY_R)

    gan = GANWrapper(
        generator=G,
        discriminator=D,
        g_opt=g_opt,
        d_opt=d_opt,
        rhythm_discriminator=R_D,
        r_opt=r_opt,
        device=device,
        instrument_classifier=instrument_classifier,
    )
    gan.vocab = dataset.vocab

    g_params = sum(p.numel() for p in G.parameters()) / 1e6
    d_params = sum(p.numel() for p in D.parameters()) / 1e6
    logger.info("GAN built: Generator %.2fM params, Discriminator %.2fM params", g_params, d_params)

    return gan

from __future__ import annotations

import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from config import EMBED_DIM, SEGMENT_DIM, DROPOUT, VOCAB_PATH
from utils.logger import logger


class Discriminator(nn.Module):
    def __init__(
            self,
            n_tracks: int,
            n_pitches: int,
            n_time_steps: int,
            device: torch.device,
            vocab_path: str = VOCAB_PATH,
            dropout: float = DROPOUT,
            gn_groups: int = 16,
    ):
        super().__init__()
        self.n_time_steps = n_time_steps
        self.n_pitches = n_pitches
        self.n_tracks = n_tracks
        self.device = device
        self.dropout = nn.Dropout(dropout)
        self.gn_groups = gn_groups

        try:
            with open(vocab_path, "rb") as f:
                vocab = pickle.load(f)
            self.instr2idx = vocab.get("instrument2idx", {})
            if not isinstance(self.instr2idx, dict):
                raise ValueError("instrument2idx should be a dictionary.")
            if not self.instr2idx:
                raise ValueError("instrument2idx is empty or missing in vocab.")
            if "instrument2idx" not in vocab:
                raise ValueError(f"Key 'instrument2idx' not found in the vocab file: {vocab_path}")
            self.instr2idx = vocab["instrument2idx"]
        except Exception as exc:
            logger.exception("Failed to load vocab from %s: %s", vocab_path, exc)
            raise

        self.n_instruments = len(self.instr2idx)
        self.prog_embed = nn.Embedding(self.n_instruments, EMBED_DIM).to(device)
        self.segment_emb = nn.Embedding(SEGMENT_DIM, SEGMENT_DIM).to(device)

        self.conv: nn.Sequential | None = None
        self.fc: nn.Linear | None = None
        self.last_in_channels: int | None = None
        with torch.no_grad():
            dummy_x = torch.zeros(1, 4, self.n_time_steps, 1, 1, device=device)
            dummy_prog = torch.zeros(1, self.n_tracks, dtype=torch.long, device=device)
            self.forward(dummy_x, dummy_prog)

        self.to(device)

    def _group_norm(self, channels: int) -> nn.GroupNorm:
        return nn.GroupNorm(self.gn_groups, channels)

    def forward(self, x: Tensor, program_ids: Tensor | None = None) -> Tensor:
        logger.debug(f"x.shape before view: {x.shape}")
        if x.ndimension() == 3:
            x = x.unsqueeze(3).unsqueeze(4)
        elif x.ndimension() != 5:
            raise ValueError(f"Expected 5D input, got {x.ndimension()}D tensor with shape {x.shape}")
        B, C, T, H, W = x.shape
        assert C == 4, f"Expected 4 tracks (channels), got {C}"
        assert T == 512, f"Expected sequence length 512, got {T}"

        device = x.device

        if program_ids is not None:
            assert program_ids.ndim == 2, ...
            prog_embed = self.prog_embed(program_ids)
            B, D, Tracks = prog_embed.permute(0, 2, 1).shape
            prog_embed = prog_embed.permute(0, 2, 1).reshape(B, D * Tracks, 1, 1, 1).repeat(1, 1, T, H, W)
        else:
            prog_embed = torch.zeros(B, 0, T, H, W, device=device)

        seg_input = torch.zeros(B, dtype=torch.long, device=device)
        seg_embed = self.segment_emb(seg_input).view(B, -1, 1, 1, 1)
        seg_embed = seg_embed.expand(B, seg_embed.shape[1], T, H, W)

        assert x.size(2) == prog_embed.size(2) == seg_embed.size(2) and \
               x.size(3) == prog_embed.size(3) == seg_embed.size(3) and \
               x.size(4) == prog_embed.size(4) == seg_embed.size(4), f"Shape mismatch: x: {x.shape}, " \
                                                                     f"prog_embed: {prog_embed.shape}, " \
                                                                     f"seg_embed: {seg_embed.shape}"

        x = torch.cat([x, prog_embed, seg_embed], dim=1)

        logger.debug(f"x.shape before conv: {x.shape}")

        if self.conv is None or self.last_in_channels != x.size(1):
            logger.info(f"Initializing conv with in_channels={x.size(1)}")
            self.last_in_channels = x.size(1)
            self.conv = nn.Sequential(
                nn.Conv3d(x.size(1), 128, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=False),
                nn.Dropout3d(self.dropout.p),
                nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1),
                self._group_norm(256),
                nn.LeakyReLU(0.2, inplace=False),
                nn.Dropout3d(self.dropout.p),
            ).to(device)

        features = self.conv(x)
        features = F.adaptive_avg_pool3d(features, 1).flatten(1)
        if torch.isnan(features).any():
            print("‼ NaN in features before FC:", features)
        print("features stats:", features.min(), features.max(), features.mean())
        print("features.shape", features.shape)
        assert features.numel() > 0, "features is empty!"

        if self.fc is None:
            self.fc = nn.Linear(features.size(1), 1).to(device)

        return self.fc(self.dropout(features))

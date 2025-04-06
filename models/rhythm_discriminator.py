from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from utils.logger import logger


class RhythmDiscriminator(nn.Module):
    def __init__(
        self,
        n_tracks: int,
        n_time_steps: int,
        hidden: int = config.RD_HIDDEN,
        dropout: float = config.RD_DROPOUT,
        in_channels: int = 1,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_channels
        self.n_tracks = n_tracks
        self.n_time_steps = n_time_steps

        self.conv = nn.Sequential(
            nn.Conv3d(self.in_channels + 1, hidden // 2, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Dropout3d(dropout),

            nn.Conv3d(hidden // 2, hidden, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(hidden),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Dropout3d(dropout),

            nn.Conv3d(hidden, hidden * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(hidden * 2),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Dropout3d(dropout),

            nn.Conv3d(hidden * 2, hidden * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(hidden * 4),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Dropout3d(dropout),
        ).to(device)

        dummy = torch.randn(
            1,
            self.in_channels + 1,
            self.n_tracks,
            self.n_time_steps,
            1,
            device=device,
        )

        assert dummy.ndimension() == 5, f"Expected dummy to have 5 dimensions, got {dummy.ndimension()} with shape {dummy.shape}"
        assert self.conv is not None, "Convolutional layers not initialized correctly."

        conv_out = self.conv(dummy)
        conv_out = F.adaptive_avg_pool3d(conv_out, 1).view(1, -1)
        self.fc = nn.Linear(conv_out.size(1), 1).to(device)
        assert self.fc is not None, "Fully connected layer not initialized correctly."

        conv_out = self.conv(dummy)
        conv_out = F.adaptive_avg_pool3d(conv_out, 1).view(1, -1)
        assert conv_out.size(1) > 0, "Conv output has invalid size"

        logger.info("RhythmDiscriminator initialized with dummy input shape %s", dummy.shape)

    def forward(self, x: torch.Tensor, program_ids: torch.Tensor) -> torch.Tensor:
        B, C, T, TS, H, W = x.shape

        if C != self.in_channels:
            logger.warning(f"Adjusting input channels: expected {self.in_channels}, got {C}")
            if C > self.in_channels:
                x = x[:, :self.in_channels]
            else:
                pad = torch.zeros((B, self.in_channels - C, T, TS, H, W), device=x.device)
                x = torch.cat([x, pad], dim=1)

        assert program_ids.ndim in [1, 2], f"Unexpected dimension for program_ids: {program_ids.ndim}"

        if program_ids.ndim == 2:
            prog = program_ids.float().mean(dim=1, keepdim=True)
        else:
            prog = program_ids.float()

        prog = prog.view(B, 1, 1, 1, 1, 1).repeat(B, 1, T, TS, H, W)

        assert x.size(2) == prog.size(2) and \
               x.size(3) == prog.size(3) and \
               x.size(4) == prog.size(4) and \
               x.size(5) == prog.size(5), f"Shape mismatch: x: {x.shape}, prog: {prog.shape}"
        x = torch.cat([x, prog], dim=1)

        C = x.shape[1]
        x = x.reshape(B, C, T, TS * H, W)

        out = self.conv(x)
        out = F.adaptive_avg_pool3d(out, 1).flatten(1)
        return self.fc(out)

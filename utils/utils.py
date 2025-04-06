import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Union, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import config
from utils.logger import logger


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Random seed set to %d", seed)


def save_checkpoint(
        generator: torch.nn.Module,
        discriminator: torch.nn.Module,
        path: Union[str, os.PathLike],
) -> None:
    path_obj = Path(path)
    try:
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
            },
            str(path_obj),
        )
        logger.info("Checkpoint saved to %s", path_obj)
    except Exception as exc:
        logger.exception("Failed to save checkpoint to %s: %s", path_obj, exc)
        raise


def fill_rest(
        tensor: np.ndarray,
        vocab: Dict[str, Any],
        max_tracks: Optional[int] = None,
        max_time_steps: Optional[int] = None,
) -> None:
    rest_token = config.REST_TOKEN
    rest_idx = vocab.get("pitch2idx", {}).get(rest_token)
    if rest_idx is None:
        logger.warning("REST token '%s' not found in vocab", rest_token)
        return

    tracks, _, _, steps = tensor.shape
    n_tracks = max_tracks or tracks
    n_steps = max_time_steps or steps

    tracks, _, pitches, steps = tensor.shape
    for tr in range(min(n_tracks, tracks)):
        for t in range(min(n_steps, steps)):
            if not tensor[tr, 0, :, t].any():
                tensor[tr, 0, :, t] = 0
                tensor[tr, 0, rest_idx, t] = 1


def pad_to_match(
        x: torch.Tensor,
        y: torch.Tensor,
        dims: Tuple[int, int, int] = (2, 3, 4),
) -> Tuple[torch.Tensor, torch.Tensor]:
    for dim in dims:
        max_size = max(x.shape[dim], y.shape[dim])
        for tensor in [x, y]:
            pad = [0, 0] * tensor.ndim
            diff = max_size - tensor.shape[dim]
            if diff > 0:
                pad_index = (tensor.ndim - dim - 1) * 2 + 1
                pad[pad_index] = diff
                if tensor is x:
                    x = F.pad(x, pad, mode="constant", value=0)
                else:
                    y = F.pad(y, pad)
    return x, y

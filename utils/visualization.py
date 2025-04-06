import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Optional
from utils.logger import logger


def plot_pianoroll(
    tensor: np.ndarray,
    out_path: Optional[Path] = None,
    title: Optional[str] = "Piano Roll",
    channel: int = 0,
    track: int = 0,
) -> None:
    logger.debug("Rendering pianoroll: shape=%s", tensor.shape)
    try:
        data = tensor[track, channel]
        plt.figure(figsize=(10, 4))
        plt.imshow(data, aspect="auto", origin="lower", cmap="gray_r")
        plt.xlabel("Time")
        plt.ylabel("Pitch")
        plt.title(title or "Piano Roll")
        plt.tight_layout()

        if out_path:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_path)
            logger.info("Pianoroll saved to %s", out_path)
        else:
            plt.show()
        plt.close()
    except Exception as e:
        logger.exception("Failed to render pianoroll: %s", e)

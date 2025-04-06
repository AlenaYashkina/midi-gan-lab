import logging
from pathlib import Path
from typing import Union

from torch.utils.tensorboard import SummaryWriter

LOG_FMT: str = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)

logger = logging.getLogger("GAN-Music")
logging.getLogger("matplotlib").setLevel(logging.WARNING)


def init_tb_logger(log_dir: Union[str, Path]) -> SummaryWriter:
    path = Path(log_dir)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        logger.exception("Failed to create TensorBoard log directory %s: %s", path, exc)
        raise

    writer = SummaryWriter(log_dir=str(path))
    logger.info("Initialized TensorBoard logger at %s", path)
    return writer

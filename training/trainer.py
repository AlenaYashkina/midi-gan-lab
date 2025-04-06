from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Dict

import torch
from matplotlib import pyplot as plt
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import EARLY_STOPPING_PATIENCE
from evaluation.metrics import calculate_fid as _calc_fid
from inference.postprocess import convert_tensor_to_midi, save_midi
from utils.logger import init_tb_logger, logger
from utils.utils import save_checkpoint
from utils.visualization import plot_pianoroll


class Trainer:
    def __init__(
        self,
        gan: Any,
        dataloader: DataLoader,
        device: torch.device,
        cfg: Any,
        writer: SummaryWriter | None = None
    ) -> None:
        self.gan = gan
        self.loader = dataloader
        self.device = device
        self.epochs = cfg.EPOCHS

        self.save_dir = Path(cfg.MODEL_SAVE_PATH)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.tb = writer if writer is not None else init_tb_logger(cfg.TENSORBOARD_LOGDIR)
        self.best_fid = float("inf")

        self._init_ema_models(cfg)

        self.g_sched = CosineAnnealingWarmRestarts(self.gan.g_opt, T_0=10)
        self.d_sched = CosineAnnealingWarmRestarts(self.gan.d_opt, T_0=10)

        self.scaler = GradScaler()
        self.cfg = cfg

    def _init_ema_models(self, cfg: Any) -> None:
        decay = getattr(cfg, "EMA_DECAY", 0.999)
        logger.info("Initializing EMA with decay=%.4f", decay)

        def ema_avg(avg, param, count):
            return avg + (1 - decay) / (count + 1) * (param - avg)

        self.gan.G.cpu()
        self.ema_G = torch.optim.swa_utils.AveragedModel(self.gan.G, avg_fn=ema_avg)
        self.gan.G.to(self.device)
        self.ema_G.to(self.device)

        self.gan.rhythm_D.cpu()
        self.ema_R = torch.optim.swa_utils.AveragedModel(self.gan.rhythm_D, avg_fn=ema_avg)
        self.gan.rhythm_D.to(self.device)
        self.ema_R.to(self.device)

    def _apply_gradients(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer, parameters) -> None:
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(parameters, self.cfg.CLIP_NORM)
        self.scaler.step(optimizer)
        self.scaler.update()
        optimizer.zero_grad(set_to_none=True)

    def _run_epoch(self, epoch: int) -> Dict[str, Dict[str, float]]:
        loss_G_acc: Dict[str, float] = {}
        loss_D_acc: Dict[str, float] = {}

        for batch in tqdm(self.loader, desc=f"Epoch {epoch + 1}", leave=False):
            real = batch["tensor"].to(self.device).float()
            prog_ids = batch["program_ids"].to(self.device)
            segment_ids = batch["segment_ids"].to(self.device)

            loss_D, d_metrics = self.gan.step_D(real, prog_ids, segment_ids)

            self._apply_gradients(loss_D, self.gan.d_opt, self.gan.D.parameters())

            loss_G, g_metrics = self.gan.step_G(real, prog_ids, segment_ids)
            if torch.isnan(loss_G):
                logger.warning("Generator loss is NaN, skipping update")
            else:
                self._apply_gradients(loss_G, self.gan.g_opt, self.gan.G.parameters())
                self.ema_G.update_parameters(self.gan.G)
                self.ema_R.update_parameters(self.gan.rhythm_D)
                for k, v in g_metrics.items():
                    if not torch.isnan(torch.tensor(v)):
                        loss_G_acc[k] = loss_G_acc.get(k, 0.0) + v

            for k, v in d_metrics.items():
                loss_D_acc[k] = loss_D_acc.get(k, 0.0) + v

        num_batches = len(self.loader)
        avg_G = {k: v / num_batches for k, v in loss_G_acc.items()}
        avg_D = {k: v / num_batches for k, v in loss_D_acc.items()}
        return {"G": avg_G, "D": avg_D}

    def _sample_and_log(self, epoch: int) -> None:
        batch = next(iter(self.loader))
        real = batch["tensor"][:1].to(self.device)
        prog = batch["program_ids"][:1].to(self.device)

        with torch.no_grad():
            fake, _, _ = self.ema_G.module(real, prog)
        fake = fake.cpu()

        logger.info("Sample shape: %s", tuple(fake.shape))
        sample = fake[0]
        midi = convert_tensor_to_midi(sample, self.gan.vocab, self.save_dir / "samples")
        save_midi(midi, self.save_dir / f"samples/epoch_{epoch+1}.mid")

        plot_path = self.save_dir / f"samples/epoch_{epoch + 1}.png"
        plot_pianoroll(sample.numpy(), out_path=plot_path, title=f"Sample epoch {epoch + 1}")
        self.tb.add_image("Samples/PianoRoll", plt.imread(plot_path), epoch, dataformats="HWC")

    def train(self) -> None:
        patience = EARLY_STOPPING_PATIENCE
        counter = 0

        for epoch in range(self.epochs):
            metrics = self._run_epoch(epoch)
            self.g_sched.step(epoch)
            self.d_sched.step(epoch)

            for phase in ("G", "D"):
                logger.info("[%s] Epoch %d losses:", phase, epoch+1)
                for name, val in metrics[phase].items():
                    self.tb.add_scalar(f"Loss/{phase}/{name}", val, epoch)
                    logger.info("  %s: %.4f", name, val)

            batch = next(iter(self.loader))
            real_b = batch["tensor"].to(self.device)
            prog_b = batch["program_ids"].to(self.device)
            with torch.no_grad():
                fake_b, _, _ = self.ema_G.module(real_b, prog_b)
                print("real_b:", real_b.shape)
                print("fake_b:", fake_b.shape)

            if fake_b.shape != real_b.shape:
                logger.warning(f"Skipping FID: fake_b shape {fake_b.shape}, real_b shape {real_b.shape}")
            else:
                real_feats = real_b.float().mean((2, 3)).view(real_b.size(0), -1).cpu().numpy()
                fake_feats = fake_b.float().mean(dim=1).view(fake_b.size(0), -1).cpu().numpy()
                fid = _calc_fid(real_feats, fake_feats)
                self.tb.add_scalar("Metrics/FID", fid, epoch)
                logger.info("[Metrics] Epoch %d FID: %.4f", epoch + 1, fid)

                if fid < self.best_fid:
                    self.best_fid = fid
                    counter = 0
                    save_checkpoint(self.ema_G.module, self.gan.D, self.save_dir / "gan_best.pt")
                    logger.info("New best FID, checkpoint saved")
                else:
                    counter += 1
                    if counter >= patience:
                        logger.info("Early stopping at epoch %d", epoch + 1)
                        break

            self._sample_and_log(epoch)
            gc.collect()

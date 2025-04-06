import pickle
from typing import Any, Dict, Tuple

import torch
from torch import Tensor

import config
from training.losses import (
    compute_instrument_loss,
    diversity_loss,
    generator_wgan_loss,
    harmonic_loss,
    kl_divergence,
    regularization_loss,
    discriminator_wgan_gp_loss as d_loss_fn,
    InstrumentClassifier,
)
from utils.logger import logger
from utils.utils import pad_to_match


class GANWrapper:
    def __init__(
        self,
        generator: torch.nn.Module,
        discriminator: torch.nn.Module,
        g_opt: torch.optim.Optimizer,
        d_opt: torch.optim.Optimizer,
        rhythm_discriminator: torch.nn.Module,
        r_opt: torch.optim.Optimizer,
        device: torch.device,
        instrument_classifier: InstrumentClassifier,
    ) -> None:
        self.G = generator.to(device)
        self.D = discriminator.to(device)
        self.rhythm_D = rhythm_discriminator.to(device)

        self.g_opt = g_opt
        self.d_opt = d_opt
        self.r_opt = r_opt

        self.device = device
        self.inst_cls = instrument_classifier.to(device)

        try:
            with open(config.VOCAB_PATH, "rb") as f:
                self.vocab: Dict[str, Any] = pickle.load(f)
            logger.info("Loaded vocab from %s", config.VOCAB_PATH)
        except Exception as exc:
            logger.warning("Could not load vocab: %s", exc)
            self.vocab = {}

    def _prepare_real_fake(self, real: Tensor, fake: Tensor) -> Tuple[Tensor, Tensor]:
        if real.ndim == 5:
            real = real.permute(0, 2, 4, 3,
                                1)
        else:
            raise ValueError(f"Expected 5D input tensor, got shape: {real.shape}")

        if fake.ndim == 3:
            fake = fake.unsqueeze(3).unsqueeze(4)

        fake = fake.to(self.device).float()
        real = real.to(self.device).float()

        real, fake = pad_to_match(real, fake)

        min_c = min(real.shape[1], fake.shape[1])
        real = real[:, :min_c]
        fake = fake[:, :min_c]

        return real, fake

    def step_D(
        self,
        real: Tensor,
        prog_ids: Tensor,
        segment_ids: Tensor,
    ) -> Tuple[Tensor, Dict[str, float]]:
        logger.debug("step_D start: real %s, prog_ids %s", real.shape, prog_ids.shape)

        self.D.zero_grad(set_to_none=True)

        prog_ids = prog_ids.to(self.device)
        segment_ids = segment_ids.to(self.device)

        if real.ndim == 3:
            real = real.unsqueeze(3).unsqueeze(4)
        with torch.no_grad():
            fake, _, _ = self.G(real, prog_ids, segment_ids)
        fake = fake.detach()

        real, fake = self._prepare_real_fake(real, fake)

        preds_real = self.D(real, prog_ids)
        preds_fake = self.D(fake, prog_ids)

        loss_gp = d_loss_fn(
            self.D,
            preds_real,
            preds_fake,
            real,
            fake,
            device=self.device,
            program_ids=prog_ids,
        )

        self.d_opt.step()

        metrics = {
            "total": loss_gp.item(),
            "wgan+gp": loss_gp.item(),
        }
        logger.debug("step_D done: %s", metrics)
        return loss_gp, metrics

    def step_G(
        self,
        real: Tensor,
        prog_ids: Tensor,
        segment_ids: Tensor,
    ) -> Tuple[Tensor, Dict[str, float]]:
        prog_ids = prog_ids.to(self.device)
        segment_ids = segment_ids.to(self.device)

        fake, mu, logvar = self.G(real, prog_ids, segment_ids)
        real, fake = self._prepare_real_fake(real, fake)

        fake_rhythm = fake.mean(dim=1, keepdim=True)
        fake_rhythm = fake_rhythm.permute(0, 1, 3, 2, 4).unsqueeze(-1)

        preds_fake = self.D(fake, prog_ids)

        loss_wgan = generator_wgan_loss(preds_fake)
        loss_kl = kl_divergence(mu, logvar)
        loss_harmonic = harmonic_loss(fake)
        loss_div = diversity_loss(fake)
        loss_reg = regularization_loss(fake)

        target_prog = prog_ids.view(prog_ids.size(0), -1)
        if fake.shape[1] < target_prog.shape[1]:
            logger.warning("fake shape %s too small for prog shape %s", fake.shape, target_prog.shape)
            pad_len = target_prog.shape[1] - fake.shape[1]
            pad_shape = (fake.shape[0], pad_len, *fake.shape[2:])
            pad = torch.zeros(pad_shape, device=fake.device)
            fake = torch.cat([fake, pad], dim=1)

        fake_cut = fake[:, :target_prog.shape[1]]
        loss_instr = compute_instrument_loss(fake_cut, target_prog, self.inst_cls)

        loss_rhythm = self.rhythm_D(fake_rhythm, prog_ids).mean()
        loss_rhythm.retain_grad()
        print("fake_rhythm.shape:", fake_rhythm.shape)

        logger.debug("mu: mean=%.4f std=%.4f", mu.mean().item(), mu.std().item())
        logger.debug("logvar: mean=%.4f std=%.4f", logvar.mean().item(), logvar.std().item())
        logger.debug("fake: min=%.4f max=%.4f mean=%.4f", fake.min().item(), fake.max().item(), fake.mean().item())
        total_loss = (
            loss_wgan
            + config.LOSS_WEIGHT_KL * loss_kl
            + config.LOSS_WEIGHT_HARMONIC * loss_harmonic
            + config.LOSS_WEIGHT_DIVERSITY * loss_div
            + config.LOSS_WEIGHT_REGULARIZATION * loss_reg
            + config.LOSS_WEIGHT_INSTRUMENT * loss_instr
            + config.LOSS_WEIGHT_RHYTHM * loss_rhythm
        )

        if not hasattr(self, 'vocab') or self.vocab is None:
            logger.warning("Vocabulary not loaded or missing.")
        if self.vocab is not None:
            idx2token = self.vocab.get("idx2token", [])
            vocab_size = len(idx2token)
            if vocab_size and ((fake < 0) | (fake >= vocab_size)).any():
                logger.warning("Generated fake contains invalid token indices")

        losses = [loss_wgan, loss_kl, loss_harmonic, loss_div, loss_reg, loss_instr, loss_rhythm]
        if any(torch.isnan(loss).any() for loss in losses):
            logger.warning("NaN detected in loss components!")
            for name, val in zip(
                    ["wgan", "kl", "harmonic", "diversity", "regularization", "instrument", "rhythm"],
                    losses,
            ):
                if torch.isnan(val).any():
                    logger.warning(f"⚠ {name} loss is NaN")
            zero = torch.tensor(0.0, requires_grad=True, device=self.device)
            return zero, {k: float("nan") for k in
                          ["total", "wgan", "kl", "harmonic", "diversity", "regularization", "instrument", "rhythm"]}

        metrics = {
            "total": total_loss.item(),
            "wgan": loss_wgan.item(),
            "kl": loss_kl.item(),
            "harmonic": loss_harmonic.item(),
            "diversity": loss_div.item(),
            "regularization": loss_reg.item(),
            "instrument": loss_instr.item(),
            "rhythm": loss_rhythm.item(),
        }
        logger.debug("step_G done: %s", metrics)
        self.G.zero_grad(set_to_none=True)
        total_loss.backward(retain_graph=True)
        self.g_opt.step()
        return total_loss, metrics

from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import MODES, INSTRUMENT_LOSS_DRUM_WEIGHT, INSTRUMENT_LOSS_DRUM_IDX
from models.instrument_classifier import InstrumentClassifier
from utils.logger import logger
from utils.music_theory import determine_tonality


def _gradient_penalty(
    discriminator: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    device: torch.device,
    program_ids: Optional[torch.Tensor] = None,
    lambda_gp: float = 1.0,
) -> torch.Tensor:
    batch_size = real.size(0)
    min_t = min(real.size(2), fake.size(2))
    if real.dim() == 5:
        real_sliced = real[:, :, :min_t, :, :]
        fake_sliced = fake[:, :, :min_t, :, :]
    elif real.dim() == 3:
        real_sliced = real[:, :min_t, :]
        fake_sliced = fake[:, :min_t, :]
    else:
        raise ValueError(f"Unsupported tensor shape for gradient penalty: {real.shape}")

    alpha = torch.rand(batch_size, 1, 1, 1, 1, device=device)
    interp = (alpha * real_sliced + (1 - alpha) * fake_sliced).requires_grad_(True)

    if program_ids is None:
        program_ids = torch.zeros(batch_size, interp.size(2), dtype=torch.long, device=device)

    d_interp = discriminator(interp, program_ids)
    grads = torch.autograd.grad(
        outputs=d_interp,
        inputs=interp,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    grads = grads.view(batch_size, -1)
    gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()
    logger.debug("Gradient penalty: %f", gp.item())
    return lambda_gp * gp


def discriminator_wgan_gp_loss(
    discriminator: nn.Module,
    real_preds: torch.Tensor,
    fake_preds: torch.Tensor,
    real: torch.Tensor,
    fake: torch.Tensor,
    device: torch.device,
    program_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    loss = fake_preds.mean() - real_preds.mean()
    loss_gp = _gradient_penalty(discriminator, real, fake, device, program_ids)
    total = loss + loss_gp
    logger.debug("Discriminator loss: %f (wgan %f + gp %f)", total.item(), loss.item(), loss_gp.item())
    return total


def generator_wgan_loss(fake_preds: torch.Tensor) -> torch.Tensor:
    loss = -fake_preds.mean()
    logger.debug("Generator WGAN loss: %f", loss.item())
    return loss


def harmonic_loss(x: torch.Tensor) -> torch.Tensor:
    pitch_logits = x[:, :, 0]
    notes = pitch_logits.argmax(dim=2)
    intervals = notes[:, :, 1:] - notes[:, :, :-1]

    ok_intervals = torch.tensor([0, 3, -3, 4, -4, 5, -5, 7, -7, 8, -8, 9, -9], device=x.device)
    interval_penalty = (~torch.isin(intervals, ok_intervals)).float().mean()

    batch, tracks, T = notes.shape
    total_pen = 0.0
    count = 0
    for b in range(batch):
        for tr in range(tracks):
            seq = notes[b, tr].tolist()
            key, mode = determine_tonality(seq)
            if key is None:
                continue
            scale = [(key + interval) % 12 for interval in MODES[mode]]
            out_of_scale = sum(1 for n in seq if n % 12 not in scale) / len(seq)
            total_pen += out_of_scale
            count += 1
    scale_pen = total_pen / count if count > 0 else 0.0

    penalty = interval_penalty + scale_pen
    logger.debug("Harmonic loss: %f", penalty.item())
    return penalty


def regularization_loss(x: torch.Tensor) -> torch.Tensor:
    pitch_logits = x[:, :, 0]
    notes = pitch_logits.argmax(dim=2)
    diffs = notes[:, :, 1:] - notes[:, :, :-1]

    allowed = torch.tensor([-9, -8, -7, -5, -4, -3, 0, 3, 4, 5, 7, 8, 9], device=x.device, dtype=torch.float)
    dist = (diffs.unsqueeze(-1).float() - allowed).abs().min(dim=-1).values
    penalty = dist.mean()
    logger.debug("Regularization loss: %f", penalty.item())
    return penalty


def diversity_loss(x: torch.Tensor) -> torch.Tensor:
    b = x.size(0)
    flat = x.reshape(b, -1)
    normed = flat / (flat.norm(dim=1, keepdim=True) + 1e-8)
    sim = normed @ normed.T
    sim = sim.masked_fill(torch.eye(b, device=x.device).bool(), 0.0)
    penalty = sim.mean()
    logger.debug("Diversity loss: %f", penalty.item())
    return penalty


def _instrument_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    drum_weight: float = INSTRUMENT_LOSS_DRUM_WEIGHT,
    drum_idx: int = INSTRUMENT_LOSS_DRUM_IDX,
) -> torch.Tensor:
    base_loss = F.cross_entropy(logits, labels)
    drum_mask = (labels == drum_idx)
    if not drum_mask.any():
        logger.debug("Instrument loss (no drums): %f", base_loss.item())
        return base_loss

    per_sample = F.cross_entropy(logits, labels, reduction="none")
    drum_loss = per_sample[drum_mask].mean()
    total = base_loss + drum_weight * drum_loss
    logger.debug(
        "Instrument loss: base=%.4f, drum=%.4f, total=%.4f",
        base_loss.item(), drum_loss.item(), total.item()
    )
    return total


def compute_instrument_loss(
    fake: torch.Tensor,
    target_prog: torch.Tensor,
    classifier: InstrumentClassifier,
) -> torch.Tensor:
    B, T, TS, P, C = fake.shape
    losses: List[torch.Tensor] = []

    for tr in range(T):
        tr_data = fake[:, tr]

        x = tr_data.permute(0, 3, 1, 2).contiguous()

        logits = classifier(x)

        labels = target_prog[:, tr]

        loss = _instrument_loss(logits, labels)
        losses.append(loss)

    avg_loss = torch.stack(losses).mean()
    logger.debug("Computed instrument loss: %f", avg_loss.item())
    return avg_loss


def kl_divergence(mu, logvar):
    logvar = torch.clamp(logvar, min=-10.0, max=10.0)
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

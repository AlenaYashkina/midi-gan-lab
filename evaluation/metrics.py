from typing import Final

import numpy as np
from scipy.linalg import sqrtm, LinAlgError

from utils.logger import logger

_EPS: Final[float] = 1e-9


def calculate_fid(real: np.ndarray, fake: np.ndarray) -> float:
    n_real, n_fake = real.shape[0], fake.shape[0]
    if n_real < 2 or n_fake < 2:
        logger.warning("Too few samples for FID: real=%d, fake=%d", n_real, n_fake)
        return np.nan

    if real.shape[1] != fake.shape[1]:
        logger.error(
            "Feature dimension mismatch: real=%d, fake=%d",
            real.shape[1],
            fake.shape[1],
        )
        return np.nan

    mu_r = real.mean(axis=0)
    mu_f = fake.mean(axis=0)
    sigma_r = np.cov(real, rowvar=False)
    sigma_f = np.cov(fake, rowvar=False)

    if sigma_r.ndim != 2 or sigma_f.ndim != 2:
        logger.error(
            "Covariance has wrong ndim: real=%d, fake=%d",
            sigma_r.ndim,
            sigma_f.ndim,
        )
        return np.nan

    try:
        cov_prod = sigma_r @ sigma_f
        covmean_raw = sqrtm(cov_prod, disp=False)
    except (ValueError, LinAlgError) as exc:
        logger.exception("Error computing matrix square root for FID: %s", exc)
        return np.nan

    covmean = np.real(covmean_raw)

    diff = mu_r - mu_f
    fid_score = diff @ diff + np.trace(sigma_r + sigma_f - 2.0 * covmean)

    return float(fid_score + _EPS)

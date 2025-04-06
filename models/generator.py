import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import TOP_K_DEFAULT, TOP_P_DEFAULT, Z_DIM, SEQ_LEN

_log = logging.getLogger(__name__)


def top_k_top_p_filtering(logits, top_k=TOP_K_DEFAULT, top_p=TOP_P_DEFAULT, filter_value=-float("Inf")):
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        top_k_values, _ = torch.topk(logits, top_k, dim=-1)
        min_top_k = top_k_values[:, -1, None]
        logits[logits < min_top_k] = -float("inf")

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_mask = cumulative_probs > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False

        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(1, sorted_indices, sorted_mask)
        logits = logits.masked_fill(mask, filter_value)

    return logits


class Generator(nn.Module):
    def __init__(
            self,
            transformer: nn.Module,
            segment_embedding: nn.Embedding,
            instrument_embedding: nn.Embedding,
            time_embedding: nn.Embedding,
            embedding_dim: int,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.seq_len = SEQ_LEN
        self.transformer = transformer
        self.segment_embedding = segment_embedding
        self.instrument_embedding = instrument_embedding
        self.time_embedding = time_embedding
        self.seg_dim = segment_embedding.embedding_dim
        self.inst_dim = instrument_embedding.embedding_dim
        self.time_dim = time_embedding.embedding_dim

        self.input_proj = None
        assert hasattr(transformer, 'embedding_dim'), "Transformer should have 'embedding_dim' attribute"
        self.encoder = nn.GRU(
            input_size=transformer.embedding_dim,
            hidden_size=Z_DIM,
            batch_first=True,
        )
        self.z_proj = nn.Linear(Z_DIM, transformer.embedding_dim)
        self.mu_proj = nn.Linear(Z_DIM, Z_DIM)
        self.logvar_proj = nn.Linear(Z_DIM, Z_DIM)

    def forward(
            self,
            input_ids: torch.Tensor,
            prog_ids: Optional[torch.Tensor] = None,
            segment_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.validate_ids(segment_ids, prog_ids)
        device = input_ids.device

        if input_ids.dim() == 5:
            B, C, T, freq, W = input_ids.shape
            assert C == 4, f"Expected 4 tracks, got {C}"
            if T != 512:
                input_ids = F.pad(input_ids, (0, 0, 0, 512 - T)) if T < 512 else input_ids[:, :, :512]
                B, C, T, F_, W = input_ids.shape
            T = input_ids.shape[2]
            token_emb = input_ids.view(B, C, T, -1)  # [B, C, T, F*W]
            token_emb = token_emb.permute(0, 2, 1, 3)  # [B, T, C, F*W]
            token_emb = token_emb.reshape(B, T, C * freq * W)  # [B, T, D]
            seq_len = T  # теперь seq_len = T, а не C*F*W
        else:
            token_emb = self.transformer.token_embedding(input_ids)
            seq_len = token_emb.size(1)
        assert seq_len <= self.time_embedding.num_embeddings

        if self.input_proj is None:
            _log.warning("Initializing input_proj: %d -> %d", token_emb.size(-1), self.transformer.embedding_dim)
            self.input_proj = nn.Linear(
                in_features=token_emb.size(-1),
                out_features=self.transformer.embedding_dim,
            )

        self.input_proj = self.input_proj.to(device)

        token_emb = token_emb.to(dtype=torch.float32)
        token_emb = self.input_proj(token_emb)

        print("token_emb shape:", token_emb.shape)
        batch, seq_len = token_emb.size(0), token_emb.size(1)
        _, h_n = self.encoder(token_emb.contiguous())
        h_n = h_n.squeeze(0)

        mu = self.mu_proj(h_n)
        logvar = self.logvar_proj(h_n)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        z_proj_out = self.z_proj(z)
        z_emb = z_proj_out.unsqueeze(1).expand(-1, seq_len, -1)

        inst_emb, time_emb = self.get_inst_time_embeddings(prog_ids, seq_len, batch, device)
        seg_emb = self.segment_embedding(segment_ids) if segment_ids is not None else torch.zeros_like(z_emb)

        self.check_embedding_dims(seg_emb, inst_emb, time_emb)
        emb = z_emb.clone() + inst_emb.clone() + time_emb.clone() + seg_emb.clone()

        output = self.transformer(emb, segment_ids=segment_ids, prog_ids=prog_ids)

        total = output.size(1) * output.size(2)
        C, H, W = 2, 4, 1
        T = total // (C * H * W)
        _log.warning(f"Output size: {total}, target reshape size: {C * H * W}")
        if T * C * H * W != total:
            raise ValueError(f"Cannot reshape output of size {total} into (B, {C}, {T}, {H}, {W})")

        output = output.reshape(batch, T, C, H, W).permute(0, 2, 1, 3, 4)
        return output, mu, logvar

    def get_inst_time_embeddings(
        self,
        prog_ids: Optional[torch.Tensor],
        seq_len: int,
        batch: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if prog_ids is not None:
            assert prog_ids.ndimension() == 2, f"Expected prog_ids to have 2 dimensions, got {prog_ids.ndimension()}"
            inst_ids = prog_ids[:, 0].unsqueeze(1).repeat(1, seq_len)
        else:
            inst_ids = torch.zeros((batch, seq_len), dtype=torch.long, device=device)
        inst_emb = self.instrument_embedding(inst_ids)

        max_len = min(seq_len, self.time_embedding.num_embeddings)
        time_ids = torch.arange(max_len, device=device).unsqueeze(0).repeat(batch, -1)
        time_emb = self.time_embedding(time_ids)
        return inst_emb, time_emb

    @staticmethod
    def validate_ids(segment_ids: Optional[torch.Tensor], prog_ids: Optional[torch.Tensor]) -> None:
        for name, ids in [("segment_ids", segment_ids), ("prog_ids", prog_ids)]:
            if ids is not None and (not isinstance(ids, torch.Tensor) or ids.dim() != 2):
                raise ValueError(f"{name} must be a 2D torch.Tensor")

    def check_embedding_dims(
        self,
        segment_emb: Optional[torch.Tensor],
        instrument_emb: torch.Tensor,
        time_emb: torch.Tensor,
    ) -> None:
        if segment_emb is not None and segment_emb.size(-1) != self.seg_dim:
            raise ValueError(f"segment_emb last dim {segment_emb.size(-1)} != {self.seg_dim}")
        if instrument_emb is not None and instrument_emb.size(-1) != self.inst_dim:
            raise ValueError(f"instrument_emb last dim {instrument_emb.size(-1)} != {self.inst_dim}")
        if time_emb is not None and time_emb.size(-1) != self.time_dim:
            raise ValueError(f"time_emb last dim {time_emb.size(-1)} != {self.time_dim}")

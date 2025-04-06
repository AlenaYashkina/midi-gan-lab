from typing import Optional

import torch
from torch import nn

from utils.logger import logger


class Transformer(nn.Module):
    def __init__(self, embed_dim: int, num_tokens: int, num_encoder_layers: int = 2,
                 num_decoder_layers: int = 2) -> None:
        super().__init__()
        self.embedding_dim = embed_dim
        self.token_embedding = nn.Embedding(num_embeddings=num_tokens, embedding_dim=embed_dim)
        self.transformer = nn.Transformer(
            d_model=embed_dim,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
        )

    def forward(
            self,
            x: torch.Tensor,
            segment_ids: Optional[torch.Tensor] = None,
            prog_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        logger.debug(
            "Transformer received x=%s, segment_ids=%s, prog_ids=%s",
            tuple(x.shape),
            None if segment_ids is None else tuple(segment_ids.shape),
            None if prog_ids is None else tuple(prog_ids.shape),
        )

        token_emb = x

        seq = token_emb.permute(1, 0, 2)
        trans_out = self.transformer(seq, seq)
        return trans_out.permute(1, 0, 2)

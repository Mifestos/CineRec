"""
InfoNCE loss с in-batch negatives, learnable temperature и опциональной LogQ correction.

Логиты: (user_emb @ item_emb.T) * scale - alpha * log_q
где scale = exp(log_temperature) — learnable параметр.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    def __init__(
        self,
        item_counts: torch.Tensor,
        alpha: float = 0.0,
        init_temperature: float = 0.025,
        eps: float = 1e-8,
    ):
        super().__init__()
        total = item_counts.sum()
        log_q = torch.log(item_counts / total + eps)
        log_q = torch.clamp(log_q, min=-10.0)
        self.register_buffer("log_q", log_q)

        # Learnable temperature: scale = exp(log_temperature)
        # init_temperature=0.025 → scale = 40
        self.log_temperature = nn.Parameter(
            torch.log(torch.tensor(1.0 / init_temperature))
        )
        self.alpha = alpha

    def forward(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
        positive_idx: torch.Tensor,
    ) -> torch.Tensor:
        cosine = user_emb @ item_emb.T
        scale = torch.exp(self.log_temperature)
        logits = cosine * scale

        if self.alpha > 0:
            log_q_batch = self.log_q[positive_idx]
            logits = logits - self.alpha * log_q_batch.unsqueeze(0)

        target = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, target)
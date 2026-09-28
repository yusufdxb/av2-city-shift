"""Per-scenario forecasting metrics and the winner-takes-all training loss.

Metric definitions (K = number of predicted modes, all in metres):
    minFDE  endpoint error of the mode whose endpoint is closest to ground truth
    minADE  average displacement error of that same mode (the Argoverse 2 convention:
            the best mode is selected by endpoint, not by ADE)
    MR      1 if minFDE > 2.0 m
    brier-minFDE  minFDE + (1 - p_best)^2, p_best = softmax probability of that mode
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

MISS_THRESHOLD_M = 2.0


def per_sample_metrics(traj: torch.Tensor, logits: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    """traj [B,K,T,2], logits [B,K], target [B,T,2] -> dict of [B] tensors."""
    err = (traj - target.unsqueeze(1)).norm(dim=-1)  # [B,K,T]
    fde = err[..., -1]
    best = fde.argmin(1)
    ar = torch.arange(len(best), device=traj.device)
    min_fde = fde[ar, best]
    min_ade = err.mean(-1)[ar, best]
    prob = logits.softmax(-1)
    p_best = prob[ar, best]
    return {
        "min_ade": min_ade,
        "min_fde": min_fde,
        "miss": (min_fde > MISS_THRESHOLD_M).float(),
        "brier_min_fde": min_fde + (1.0 - p_best) ** 2,
        "best_mode": best,
    }


def wta_loss(traj: torch.Tensor, logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    """Winner-takes-all: regress only the closest mode, classify which mode that was.

    The winner is chosen by mean displacement (ADE) over the whole horizon.
    """
    err = (traj - target.unsqueeze(1)).norm(dim=-1)  # [B,K,T]
    winner = err.mean(-1).argmin(1).detach()
    ar = torch.arange(len(winner), device=traj.device)
    reg = F.smooth_l1_loss(traj[ar, winner], target, reduction="none").sum(-1).mean()
    cls = F.cross_entropy(logits, winner)
    return reg + cls, {"reg": reg.item(), "cls": cls.item()}

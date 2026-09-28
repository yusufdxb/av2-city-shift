"""Train one predictor. Every arm uses this script; arms differ only in CLI flags."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time

import numpy as np
import torch

from .data import Split, dev_indices, loader, to_device, training_indices
from .metrics import per_sample_metrics, wta_loss
from .model import Predictor


def evaluate(model: Predictor, split: Split, idx: np.ndarray, device, batch_size: int = 256) -> dict[str, float]:
    model.eval()
    sums: dict[str, float] = {}
    n = 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for b in loader(split, idx, batch_size, shuffle=False, workers=4):
            b = to_device(b, device)
            traj, logits, _ = model(b["agent_hist"], b["agent_valid"], b["agent_type"], b["lane_pts"], b["lane_attr"])
            m = per_sample_metrics(traj.float(), logits.float(), b["target"])
            for k in ("min_ade", "min_fde", "miss", "brier_min_fde"):
                sums[k] = sums.get(k, 0.0) + m[k].sum().item()
            n += len(b["target"])
    model.train()
    return {k: v / n for k, v in sums.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="preprocessed data root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude-city", default=None)
    ap.add_argument("--n-train", type=int, required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-map", action="store_true")
    ap.add_argument("--dev-frac", type=float, default=0.02)
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    train = Split(args.root, "train")
    # The training *sample* depends on the seed too: seeds vary both init and data draw.
    tr_idx = training_indices(train.meta, args.exclude_city, args.n_train, args.dev_frac, seed=1000 + args.seed)
    dev_idx = dev_indices(train.meta, args.dev_frac)
    np.save(os.path.join(args.out, "train_indices.npy"), tr_idx)
    cfg = vars(args) | {
        "train_indices_sha1": hashlib.sha1(tr_idx.tobytes()).hexdigest(),
        "train_city_counts": train.meta.city.iloc[tr_idx].value_counts().to_dict(),
        "n_dev": len(dev_idx),
    }
    json.dump(cfg, open(os.path.join(args.out, "config.json"), "w"), indent=2)

    model = Predictor(use_map=not args.no_map).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    warm = 1000

    def lr_at(s: int) -> float:
        if s < warm:
            return args.lr * (s + 1) / warm
        return args.lr * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, args.steps - warm)))

    log = open(os.path.join(args.out, "train_log.jsonl"), "w")
    step, epoch, t0 = 0, 0, time.time()
    wins = torch.zeros(model.k, device=device)
    run = {"loss": 0.0, "reg": 0.0, "cls": 0.0, "n": 0}
    spread_sum, spread_n = 0.0, 0
    while step < args.steps:
        for b in loader(train, tr_idx, args.batch, shuffle=True, seed=args.seed * 100003 + epoch, workers=args.workers):
            if step >= args.steps:
                break
            b = to_device(b, device)
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                traj, logits, _ = model(b["agent_hist"], b["agent_valid"], b["agent_type"], b["lane_pts"], b["lane_attr"])
            loss, parts = wta_loss(traj.float(), logits.float(), b["target"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            with torch.no_grad():
                err = (traj.float() - b["target"].unsqueeze(1)).norm(dim=-1).mean(-1)
                wins += torch.bincount(err.argmin(1), minlength=model.k).float()
                ep = traj.float()[:, :, -1]
                spread_sum += (ep.unsqueeze(1) - ep.unsqueeze(2)).norm(dim=-1).sum().item() / (model.k * (model.k - 1))
                spread_n += len(ep)
            run["loss"] += loss.item()
            run["reg"] += parts["reg"]
            run["cls"] += parts["cls"]
            run["n"] += 1
            step += 1
            if step % 200 == 0:
                share = (wins / wins.sum()).cpu()
                ent = float(-(share * (share + 1e-12).log()).sum() / math.log(model.k))
                rec = {
                    "step": step,
                    "lr": lr_at(step),
                    **{k: run[k] / run["n"] for k in ("loss", "reg", "cls")},
                    "mode_share": [round(float(x), 4) for x in share],
                    "mode_entropy": round(ent, 4),
                    # mean pairwise endpoint distance between modes: ~0 means the modes collapsed
                    "mode_spread_m": round(spread_sum / max(1, spread_n), 3),
                    "sec": round(time.time() - t0, 1),
                }
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(json.dumps(rec), flush=True)
                wins.zero_()
                spread_sum, spread_n = 0.0, 0
                run = {"loss": 0.0, "reg": 0.0, "cls": 0.0, "n": 0}
            if step % args.eval_every == 0 or step == args.steps:
                dm = evaluate(model, train, dev_idx, device)
                rec = {"step": step, "dev": dm}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(json.dumps(rec), flush=True)
        epoch += 1
    torch.save({"model": model.state_dict(), "config": cfg}, os.path.join(args.out, "model.pt"))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

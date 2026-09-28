"""Small query-based multi-modal trajectory predictor (focal agent, K modes).

Scene tokens = agents (temporal conv over history) + map polylines (PointNet).
A transformer encoder mixes them; K learned mode queries cross-attend to the scene
and each emits a 60-step trajectory and a logit. Every op is chosen to export to
ONNX opset 17 and build under TensorRT: no RNNs, no data-dependent shapes.
"""

from __future__ import annotations

import torch
from torch import nn

from .preprocess import FUT, HIST, OBJECT_TYPES, POLYLINE_TYPES


def mlp(i: int, h: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.LayerNorm(h), nn.ReLU(inplace=True), nn.Linear(h, o))


class AgentEncoder(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.inp = nn.Linear(7, 64)  # 6 kinematic features + valid flag
        self.conv = nn.Sequential(
            nn.Conv1d(64, 96, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv1d(96, 128, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv1d(128, d, 3, stride=2, padding=1), nn.ReLU(inplace=True),
        )  # fmt: skip
        self.type_emb = nn.Embedding(len(OBJECT_TYPES) + 1, d)
        self.out = mlp(2 * d, d, d)
        self.register_buffer("feat_scale", torch.tensor([1 / 50, 1 / 50, 1 / 10, 1 / 10, 1.0, 1.0]), persistent=False)

    def forward(self, hist: torch.Tensor, valid: torch.Tensor, atype: torch.Tensor) -> torch.Tensor:
        b, a, t, _ = hist.shape
        v = valid.float().unsqueeze(-1)
        # Scale to O(1): positions /50 m, velocities /10 m/s. Keeps LayerNorm statistics
        # inside FP16 range when the engine runs in half precision.
        hist = hist * self.feat_scale
        x = self.inp(torch.cat([hist * v, v], -1)).reshape(b * a, t, -1).transpose(1, 2)
        x = self.conv(x)  # [B*A, d, T/4]
        # last-step feature + max over time: keeps "where it is now" and the history summary.
        x = torch.cat([x[:, :, -1], x.amax(-1)], -1).reshape(b, a, -1)
        return self.out(x) + self.type_emb(atype.long() + 1)


class PolylineEncoder(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.pt = mlp(4, 64, 128)
        self.out = mlp(256, d, d)
        self.type_emb = nn.Embedding(len(POLYLINE_TYPES) + 1, d)
        self.inter_emb = nn.Embedding(3, d)

    def forward(self, pts: torch.Tensor, attr: torch.Tensor) -> torch.Tensor:
        # direction to the next point, repeated on the last point
        dirs = torch.cat([pts[:, :, 1:] - pts[:, :, :-1], pts[:, :, -1:] - pts[:, :, -2:-1]], 2)
        h = self.pt(torch.cat([pts / 50.0, dirs / 5.0], -1))  # [B, L, P, 128]
        g = h.amax(2, keepdim=True).expand_as(h)
        h = torch.cat([h, g], -1).amax(2)  # PointNet with one global-context round
        return self.out(h) + self.type_emb(attr[..., 0].long() + 1) + self.inter_emb(attr[..., 1].long() + 1)


class PosEnc(nn.Module):
    """Embeds a token's reference pose (x, y, cos, sin) so attention knows geometry."""

    def __init__(self, d: int):
        super().__init__()
        self.net = mlp(4, d, d)

    def forward(self, xy: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([xy / 50.0, heading], -1))


class Predictor(nn.Module):
    def __init__(self, d: int = 128, k: int = 6, enc_layers: int = 3, dec_layers: int = 2, heads: int = 4, use_map: bool = True, query_std: float = 1.0, dropout: float = 0.1):
        super().__init__()
        self.k = k
        self.use_map = use_map
        self.agent_enc = AgentEncoder(d)
        self.lane_enc = PolylineEncoder(d)
        self.pos = PosEnc(d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=dropout, batch_first=True, norm_first=True)
        # Pre-LN stacks need a final norm, or the residual stream grows without bound
        # (observed: focal-token RMS 1.7 -> 17 in 5k steps, then gradient blow-up).
        self.encoder = nn.TransformerEncoder(layer, enc_layers, norm=nn.LayerNorm(d), enable_nested_tensor=False)
        dlayer = nn.TransformerDecoderLayer(d, heads, 4 * d, dropout=dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dlayer, dec_layers, norm=nn.LayerNorm(d))
        # Mode queries must be on the scale of the focal token they are added to (RMS ~1).
        # At std 0.02 the modes were near-identical under dropout noise, the winner was
        # arbitrary, and the probability head stayed at log(K) (overfit test, 512 scenes).
        self.queries = nn.Parameter(torch.randn(k, d) * query_std)
        self.traj_head = mlp(d, 2 * d, FUT * 2)
        self.logit_head = mlp(d, d, 1)
        self.d = d

    def encode(self, agent_hist, agent_valid, agent_type, lane_pts, lane_attr):
        """Returns scene tokens, key-padding mask (True = pad), and the focal token."""
        a_tok = self.agent_enc(agent_hist, agent_valid, agent_type)
        a_xy = agent_hist[:, :, HIST - 1, :2]
        a_hd = agent_hist[:, :, HIST - 1, 4:6]
        a_tok = a_tok + self.pos(a_xy, a_hd)
        a_pad = agent_type < 0
        l_tok = self.lane_enc(lane_pts, lane_attr)
        l_xy = lane_pts.mean(2)
        l_dir = lane_pts[:, :, -1] - lane_pts[:, :, 0]
        l_hd = l_dir / (l_dir.norm(dim=-1, keepdim=True) + 1e-6)
        l_tok = l_tok + self.pos(l_xy, l_hd)
        l_pad = lane_attr[..., 0] < 0
        if not self.use_map:
            l_pad = torch.ones_like(l_pad)
            l_tok = l_tok * 0.0
        tokens = torch.cat([a_tok, l_tok], 1)
        pad = torch.cat([a_pad, l_pad], 1)
        # Mask as additive float bias: exports cleanly and never produces all-masked rows
        # (the focal agent is always present).
        bias = pad.float() * -1e4
        tokens = self.encoder(tokens, src_key_padding_mask=bias)
        return tokens, bias, tokens[:, 0]

    def forward(self, agent_hist, agent_valid, agent_type, lane_pts, lane_attr):
        tokens, bias, focal = self.encode(agent_hist, agent_valid, agent_type, lane_pts, lane_attr)
        b = tokens.shape[0]
        q = self.queries.unsqueeze(0).expand(b, -1, -1) + focal.unsqueeze(1)
        h = self.decoder(q, tokens, memory_key_padding_mask=bias)
        traj = self.traj_head(h).reshape(b, self.k, FUT, 2)
        # Predict displacements and integrate: smooth trajectories, easier optimisation.
        traj = traj.cumsum(2)
        logits = self.logit_head(h).squeeze(-1)
        return traj, logits, focal

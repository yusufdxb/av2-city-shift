"""Independent re-derivation of every registered decision (H1 to H13) from the released per-row tables.

For every claim this recomputes, with its own code and its own random seed:
  * every gate the preregistrations make the verdict depend on: positive controls (PC1, PC2, PC3, Stage 3a STATIC,
    Stage 3/4 STATIC and LOG), event floors, undefined-ratio and undefined-bootstrap-replicate rules, the Stage 3
    SHAM stopped-specific rule, the Stage 4 SHAM2 dose gate (realised dose >= 99% of PATCH, never above it), the
    Stage 4 pool hash, and H2's "detectable, below the 0.25 useful magnitude" category;
  * each decision component (point estimate plus a fresh city-stratified, scenario-cluster, paired, seeds-averaged
    bootstrap interval at the level fixed in the preregistration text, not read from the report under audit);
  * the per-component decision and the claim verdict.

Pass means all of the following:
  * every CI level hard-coded below agrees with the level the report records or implies;
  * every point estimate and every gate value matches the report within 1e-9;
  * every interval endpoint matches within Monte Carlo tolerance (15% of the reported interval width);
  * every component decision and every claim verdict equals the reported one.
A decision bound within that tolerance of its threshold (independent or reported bound) is NEAR-BOUNDARY. It is
resolved by a high-draw bootstrap (default 100,000 draws, fresh seed) whose Monte Carlo standard error must put the
bound clearly on one side of the threshold, and, where the report's resampling is reproducible (H1, H4, Stage 4),
by replaying the report's own seed and draw layout, which must give the reported bound exactly.
Any mismatch, NaN, or unresolved boundary exits nonzero.

Imports nothing from cityshift. Needs releases v1.0, v1.1 and v1.2 unpacked in the repository root.

    python scripts/audit_decisions.py [--draws 10000] [--resolve-draws 100000]
"""

import argparse
import hashlib
import json
import sys
import time

import numpy as np
import pandas as pd

CITIES = ["austin", "dearborn", "miami", "palo-alto", "pittsburgh", "washington-dc"]

# CI levels, fixed from the preregistration text (docs/preregistration/), never read from a report.
# Stage 1 deviation 3 and Stage 2 deviation 5: Bonferroni over the four-hypothesis family H1 to H4.
LEVEL_S12 = 1 - 0.05 / 4  # 98.75%
# Stage 3a: "The pre-set decision uses a 98.75% scenario-cluster bootstrap CI" (kept for H5, H6a, H6b).
LEVEL_S3A = 1 - 0.05 / 4  # 98.75%
# Stage 3: "two-sided 99.1667% percentile CIs (six-comparison Bonferroni family at 5%)".
LEVEL_S3 = 1 - 0.05 / 6  # 99.1667%
# Stage 4: "for this eight-component family use 99.375% CIs (0.05/8 two-sided) for decisions".
LEVEL_S4 = 1 - 0.05 / 8  # 99.375%
LEVEL_S4_DESCRIPTIVE = 0.9875
REGISTERED_DRAWS = 10000
POOL_SHA256 = "a7489d7bd7b95986807ee360e62a01c45a15f4c3297258ba4751f8520caee0c8"  # Stage 4 prereg, first paragraph
N_VAL = 24988
N_POOL = 8140
MC_TOL = 0.15  # share of the reported interval width
SE_MARGIN = 4.0  # a resolved bound must sit this many Monte Carlo SEs from its threshold

FAIL: list[str] = []


def fail(msg):
    FAIL.append(msg)


def close(x, y, tol=1e-9):
    """NaN-safe closeness: NaN never counts as close."""
    x, y = float(x), float(y)
    return abs(x - y) <= tol * max(1.0, abs(y))


def fnum(x):
    return "nan" if x is None or not np.isfinite(x) else f"{x: .5f}"


# ---------------------------------------------------------------- bootstrap engine ----------------------------------
def boot(strata, stat, draws, rng, chunk=256):
    """City-stratified cluster bootstrap.

    strata: {key: n_clusters}. For each chunk, every stratum gets an independent multinomial resample expressed as a
    weight matrix w (chunk x n) of resampled counts. stat({key: w}) -> {name: array(chunk)}.
    Returns {name: array(draws)}.
    """
    out = {}
    for s in range(0, draws, chunk):
        k = min(chunk, draws - s)
        ws = {}
        for key, n in strata.items():
            idx = rng.integers(0, n, size=(k, n))
            ws[key] = np.bincount((idx + (np.arange(k) * n)[:, None]).ravel(), minlength=k * n).reshape(k, n).astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            res = stat(ws)
        for name, v in res.items():
            out.setdefault(name, np.empty(draws))[s:s + k] = v
    return out


def unit(strata):
    return {key: np.ones((1, n)) for key, n in strata.items()}


def point_of(strata, stat):
    with np.errstate(divide="ignore", invalid="ignore"):
        return {name: float(v[0]) for name, v in stat(unit(strata)).items()}


def interval(draws, level):
    a = (1 - level) / 2
    return [float(x) for x in np.quantile(draws, [a, 1 - a])]


# ---------------------------------------------------------------- components and rules -------------------------------
class Component:
    """One decision component: point estimate, CI, the registered CI rules and point bar."""

    def __init__(self, name, level, point, draws, reported_point, reported_ci, ci_rules, point_rule=None,
                 resolver=None, replay=None, drop_undefined=True):
        self.name, self.level = name, level
        self.point, self.reported_point = point, reported_point
        valid = draws[np.isfinite(draws)] if drop_undefined else draws
        self.undefined_share = float((~np.isfinite(draws)).mean())
        self.ci = interval(valid, level) if len(valid) else [np.nan, np.nan]
        self.reported_ci = [np.nan if v is None else float(v) for v in reported_ci]
        self.ci_rules = ci_rules  # list of (side, op, threshold); side in {"lo", "hi"}; joined by all() unless "any"
        self.point_rule = point_rule  # (op, bar) or None
        self.resolver, self.replay = resolver, replay
        self.notes = []

    @staticmethod
    def _cmp(x, op, thr):
        return bool(x > thr) if op == ">" else bool(x < thr) if op == "<" else bool(x >= thr) if op == ">=" else bool(x <= thr)

    def ci_pass(self, ci):
        rules = [r for r in self.ci_rules if r != "any"]
        vals = [self._cmp(ci[0] if side == "lo" else ci[1], op, thr) for side, op, thr in rules]
        return any(vals) if "any" in self.ci_rules else all(vals)

    def point_pass(self, p):
        return True if self.point_rule is None else self._cmp(p, *self.point_rule)

    def evaluate(self, resolve_draws, rng):
        self.tol = MC_TOL * abs(self.reported_ci[1] - self.reported_ci[0])
        self.point_ok = close(self.point, self.reported_point)
        self.ci_ok = all(abs(a - b) <= self.tol for a, b in zip(self.ci, self.reported_ci))
        self.rep_ci_pass = self.ci_pass(self.reported_ci)
        self.ind_ci_pass = self.ci_pass(self.ci)
        self.rep_point_pass = self.point_pass(self.reported_point)
        self.ind_point_pass = self.point_pass(self.point)
        near = []
        for rule in self.ci_rules:
            if rule == "any":
                continue
            side, _, thr = rule
            i = 0 if side == "lo" else 1
            if abs(self.ci[i] - thr) <= self.tol or abs(self.reported_ci[i] - thr) <= self.tol:
                near.append((side, thr))
        self.near = near
        self.final_ci_pass = self.ind_ci_pass
        if near:
            self.resolve(resolve_draws, rng)
        self.ok = (self.point_ok and self.ci_ok and self.final_ci_pass == self.rep_ci_pass
                   and self.ind_point_pass == self.rep_point_pass)
        if not self.ok:
            fail(f"component {self.name}")

    def resolve(self, resolve_draws, rng):
        draws = self.resolver(resolve_draws, rng)
        valid = draws[np.isfinite(draws)]
        ci = interval(valid, self.level)
        batches = np.array_split(valid, 20)
        se = np.std([interval(b, self.level) for b in batches], axis=0, ddof=1) / np.sqrt(len(batches))
        self.resolved_ci, self.resolved_se = ci, se
        clear = True
        for side, thr in self.near:
            i = 0 if side == "lo" else 1
            margin = abs(ci[i] - thr) / se[i] if se[i] > 0 else np.inf
            self.notes.append(f"NEAR-BOUNDARY {side} vs {thr:g}: independent {self.ci[i]:.5f}, reported "
                              f"{self.reported_ci[i]:.5f}, tol {self.tol:.5f}; {len(draws)} draws -> "
                              f"{ci[i]:.5f} (MC SE {se[i]:.5f}, {margin:.1f} SE from threshold)")
            if not (margin > SE_MARGIN):
                clear = False
                self.notes.append(f"UNRESOLVED: {side} bound within {SE_MARGIN} MC SE of {thr:g}")
        self.final_ci_pass = self.ci_pass(ci)
        if not clear:
            self.final_ci_pass = None
        if self.replay is not None:
            rci = self.replay()
            exact = all(close(a, b, 1e-9) for a, b in zip(rci, self.reported_ci))
            self.notes.append(f"replay of the report's seed and draw layout: [{rci[0]:.6f}, {rci[1]:.6f}] "
                              f"({'equals' if exact else 'DIFFERS FROM'} reported)")
            if not exact:
                self.final_ci_pass = None
        else:
            self.notes.append("no seed replay for this component (report resampling not reproduced); high-draw only")

    def passes(self):
        """Resolved component decision (None if unresolved)."""
        if self.final_ci_pass is None:
            return None
        return self.final_ci_pass and self.ind_point_pass

    def line(self):
        tag = "OK " if self.ok else "BAD"
        rep = self.rep_ci_pass and self.rep_point_pass
        res = self.passes()
        s = (f"{tag} {self.name:16s} level {self.level * 100:.4f}% point {fnum(self.point)} (reported {fnum(self.reported_point)}) "
             f"ci [{fnum(self.ci[0])}, {fnum(self.ci[1])}] reported [{fnum(self.reported_ci[0])}, {fnum(self.reported_ci[1])}] "
             f"component {'pass' if res else 'fail' if res is not None else 'UNRESOLVED'} (reported {'pass' if rep else 'fail'})")
        if self.undefined_share > 0:
            s += f" undefined replicates {self.undefined_share:.4f}"
        return "\n".join([s] + [f"      {n}" for n in self.notes])


GATES: list[str] = []
VERDICTS: list[str] = []


def gate(name, value, reported, passed, reported_pass=None, tol=1e-9):
    """A registered gate (control, floor, dose). Value must match the report; pass must match if reported."""
    ok = True
    if reported is not None:
        ok &= close(value, reported, tol)
    if reported_pass is not None:
        ok &= bool(passed) == bool(reported_pass)
    GATES.append(f"{'OK ' if ok else 'BAD'} {name:44s} value {value!s:>22} reported {reported!s:>22} "
                 f"{'pass' if passed else 'FAIL'}" + ("" if reported_pass is None else f" (reported {'pass' if reported_pass else 'FAIL'})"))
    if not ok:
        fail(f"gate {name}")
    return bool(passed)


def verdict(name, mine, reported):
    ok = mine == reported
    VERDICTS.append(f"{'OK ' if ok else 'BAD'} {name:5s} verdict {mine!r} (reported {reported!r})")
    if not ok:
        fail(f"verdict {name}")


# ---------------------------------------------------------------- helpers ---------------------------------------------
def seed_avg(df, prefix, metric, seeds=3):
    cols = [f"{prefix}_s{s}_{metric}" if prefix else f"s{s}_{metric}" for s in range(seeds)]
    return df[cols].to_numpy(float).mean(1)


def auroc(pos, neg):
    ranks = pd.Series(np.concatenate([pos, neg])).rank(method="average").to_numpy()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def ratio_pooled(ws, num, den, cities, sign=1.0, offset=-1.0, nan_mean=False, keep=None):
    """mean over cities of sign*(sum num / sum den) + offset, with NaN where the denominator sum is zero."""
    vals = []
    for c in cities:
        if keep is not None and not keep[c]:
            continue
        d = ws[c] @ den[c]
        vals.append(np.where(d > 0, sign * (ws[c] @ num[c]) / np.where(d > 0, d, 1) + offset, np.nan))
    return np.nanmean(vals, 0) if nan_mean else np.mean(vals, 0)


def kept_mean(w, miss_sorted, n_keep):
    """Mean miss over the n_keep lowest-score units of weighted resample; columns already in score order."""
    cs = np.cumsum(w, 1)
    kept = np.clip(n_keep - (cs - w), 0, w)
    return kept @ miss_sorted / n_keep


def capture_fraction(w, miss, order_u, order_o):
    n = int(round(w[0].sum()))
    n_keep = int(round(0.8 * n))
    base = w @ miss / n
    keep_u = kept_mean(w[:, order_u], miss[order_u], n_keep)
    keep_o = kept_mean(w[:, order_o], miss[order_o], n_keep)
    denom = base - keep_o
    return np.where(denom > 0, (base - keep_u) / np.where(denom > 0, denom, 1), np.nan)


# ---------------------------------------------------------------- Stage 1 --------------------------------------------
def stage1(args, rng, comps):
    s1 = json.load(open("reports/confirmatory/stage1_results.json"))
    for h in ("H1", "H2", "H3"):  # level check: the report's decision CI is the Bonferroni one and is wider than its 95% CI
        lo, hi = s1[h]["ci_bonferroni"]
        lo95, hi95 = s1[h]["ci95"]
        if not (lo <= lo95 and hi >= hi95):
            fail(f"{h} ci_bonferroni is not wider than ci95")
    A = pd.read_parquet("evals/ALL.parquet")
    L = {c: pd.read_parquet(f"evals/LOCO-{c}.parquet") for c in CITIES}
    NM = pd.read_parquet("evals/NOMAP.parquet")
    SW = pd.read_parquet("evals/ALL_mapswap.parquet")
    for name, d in [("NOMAP", NM), ("mapswap", SW)] + list(L.items()):
        if len(d) != N_VAL or not (d.val_index.to_numpy() == A.val_index.to_numpy()).all():
            fail(f"Stage 1 row alignment {name}")
    city = A.city.to_numpy()
    a_miss = seed_avg(A, "", "miss")

    # ---- positive controls (A5) ----
    r = s1["PC1"]
    mr_a, mr_n = float(A.s0_miss.to_numpy(float).mean()), float(NM.s0_miss.to_numpy(float).mean())
    # the report averages float32 miss columns in float32 here; this audit uses float64, hence the 1e-6 tolerance
    pc1 = gate("PC1 NOMAP rel MR vs ALL (seed 0) >= 0.10", (mr_n - mr_a) / mr_a, r["rel"],
               (mr_n - mr_a) / mr_a >= 0.10, r["pass"], tol=1e-6)
    r = s1["PC2"]
    mr_o, mr_s = float(a_miss.mean()), float(seed_avg(SW, "", "miss").mean())
    au = auroc(SW.disagree.to_numpy(float), A.disagree.to_numpy(float))
    gate("PC2 map-swap rel MR >= 0.10", (mr_s - mr_o) / mr_o, r["rel"], (mr_s - mr_o) / mr_o >= 0.10)
    gate("PC2 U1 AUROC swapped vs original > 0.6", au, r["U1_auroc_swapped_vs_orig"], au > 0.6)
    pc2 = gate("PC2 overall", int((mr_s - mr_o) / mr_o >= 0.10 and au > 0.6), int(r["pass"]),
               (mr_s - mr_o) / mr_o >= 0.10 and au > 0.6, r["pass"])
    # A8 floor: flag only (a city with ALL MR < 0.05 is reported but flagged; it does not change the decision)
    low = [c for c in CITIES if a_miss[city == c].mean() < 0.05]
    gate("A8 floor: cities with ALL MR < 0.05 (flag only)", len(low), None, True)

    # ---- H1 ----
    miss = {c: (a_miss[city == c], seed_avg(L[c], "", "miss")[city == c]) for c in CITIES}
    st1 = {c: int((city == c).sum()) for c in CITIES}

    def h1_stat(ws):
        return {"H1": ratio_pooled(ws, {c: miss[c][1] for c in CITIES}, {c: miss[c][0] for c in CITIES}, CITIES)}

    p1 = point_of(st1, h1_stat)["H1"]
    d1 = boot(st1, h1_stat, args.draws, rng)["H1"]

    def h1_replay():
        g = np.random.default_rng(0)
        pooled = []
        for c in CITIES:
            a, lo = miss[c]
            ii = g.integers(0, len(a), size=(REGISTERED_DRAWS, len(a)))
            am = a[ii].mean(1)
            pooled.append((lo[ii].mean(1) - am) / am)
            del ii
        pooled = np.mean(pooled, 0)
        return [float(np.percentile(pooled, 100 * (1 - LEVEL_S12) / 2)), float(np.percentile(pooled, 100 * (1 + LEVEL_S12) / 2))]

    c1 = Component("H1", LEVEL_S12, p1, d1, s1["H1"]["pooled_rel_change"], s1["H1"]["ci_bonferroni"],
                   [("lo", ">", 0.0)], (">=", 0.05),
                   resolver=lambda n, g: boot(st1, h1_stat, n, g)["H1"], replay=h1_replay)

    # ---- H2 / H3 (capture fraction at 80% coverage) ----
    held, ind = {}, {}
    for c in CITIES:
        d = L[c]
        m = seed_avg(d, "", "miss")
        e = seed_avg(d, "", "min_fde")  # exact column names: never brier_min_fde (Stage 1 deviation 4)
        u = d.disagree.to_numpy(float)
        for store, mask in ((held, city == c), (ind, city != c)):
            mm, ee, uu = m[mask], e[mask], u[mask]
            store[c] = (mm, np.argsort(uu, kind="stable"), np.argsort(ee, kind="stable"))
    st2 = {("h", c): len(held[c][0]) for c in CITIES} | {("i", c): len(ind[c][0]) for c in CITIES}

    def h23_stat(ws):
        cfh = [capture_fraction(ws[("h", c)], *held[c]) for c in CITIES]
        cfi = [capture_fraction(ws[("i", c)], *ind[c]) for c in CITIES]
        return {"H2": np.nanmean(cfh, 0), "H3": np.nanmean([h - i for h, i in zip(cfh, cfi)], 0),
                "H2_undef": np.mean([~np.isfinite(h) for h in cfh], 0)}

    pts = point_of(st2, h23_stat)
    d23 = boot(st2, h23_stat, args.cf_draws, rng, chunk=64)
    und = s1["H2"].get("folds_undefined", [])
    gate("H2 folds with undefined CF at the point", int(round(pts["H2_undef"] * 6)), len(und), True)
    gate("H2 undefined per-fold CF share in bootstrap (info)", round(float(d23["H2_undef"].mean()), 6), None, True)
    res23 = lambda n, g: boot(st2, h23_stat, min(n, 20000), g, chunk=64)  # noqa: E731
    c2 = Component("H2", LEVEL_S12, pts["H2"], d23["H2"], s1["H2"]["pooled_CF_U1_heldout"], s1["H2"]["ci_bonferroni"],
                   [("lo", ">", 0.0)], None, resolver=lambda n, g: res23(n, g)["H2"])
    c3 = Component("H3", LEVEL_S12, pts["H3"], d23["H3"], s1["H3"]["pooled_CF_diff_heldout_minus_indist"],
                   s1["H3"]["ci_bonferroni"], [("lo", ">", 0.0), ("hi", "<", 0.0), "any"], None,
                   resolver=lambda n, g: res23(n, g)["H3"])
    for c in (c1, c2, c3):
        c.evaluate(args.resolve_draws, rng)
        comps.append(c)

    # ---- verdicts (Stage 1 kill criteria; PC1/PC2 gate only the nulls) ----
    if c1.passes() and c1.ind_point_pass:
        v1 = "supported"
    else:
        v1 = "dead" if pc1 else "null, uninterpretable (PC1 failed or missing)"
    verdict("H1", v1, s1["decisions"]["H1"])
    if c2.passes():
        v2 = "supported" if pts["H2"] >= 0.25 else "detectable, below the 0.25 useful magnitude"
    else:
        v2 = "dead" if pc2 else "null, uninterpretable (PC2 failed or missing)"
    gate("H2 point vs 0.25 useful magnitude", pts["H2"], s1["H2"]["pooled_CF_U1_heldout"], pts["H2"] >= 0.25)
    verdict("H2", v2, s1["decisions"]["H2"])
    verdict("H3", "differs" if c3.passes() else "no detectable difference", s1["decisions"]["H3"])


# ---------------------------------------------------------------- Stage 2 --------------------------------------------
def stage2(args, rng, comps):
    s2 = json.load(open("reports/confirmatory/stage2_results.json"))
    lo, hi = s2["H4"]["ci_bonferroni"]
    if not (lo <= s2["H4"]["ci95"][0] and hi >= s2["H4"]["ci95"][1]):
        fail("H4 ci_bonferroni is not wider than ci95")
    C = pd.read_parquet("evals/stage2_closedloop.parquet")
    if len(C) != N_VAL or C.scenario_id.duplicated().any():
        fail("Stage 2 scenario count")
    city = C.city.to_numpy()
    f_all, f_loco = seed_avg(C, "ALL", "failure"), seed_avg(C, "LOCO", "failure")
    fa = {c: f_all[city == c] for c in CITIES}
    fl = {c: f_loco[city == c] for c in CITIES}
    st = {c: len(fa[c]) for c in CITIES}

    def per_city(ws):
        return {c: v for c, v in zip(CITIES, [ratio_pooled({c: ws[c]}, fl, fa, [c]) for c in CITIES])}

    draws_city = boot(st, per_city, args.draws, rng)
    point_city = {c: fl[c].mean() / fa[c].mean() - 1 if fa[c].mean() > 0 else np.nan for c in CITIES}
    share = {c: float(np.isnan(draws_city[c]).mean()) for c in CITIES}
    keep = {c: bool(np.isfinite(point_city[c]) and share[c] <= 0.01) for c in CITIES}  # Stage 2 deviation 5
    gate("H4 folds undefined (ALL zero or >1% undefined)", len([c for c in CITIES if not keep[c]]),
         len(s2["H4"]["folds_undefined"]), True)
    events = float(f_all.sum())
    powered = gate("H4 A8 floor: ALL failure events >= 100", events, s2["H4"]["events_ALL_seed_avg"], events >= 100,
                   not s2["H4"]["underpowered"])
    col_all = float(seed_avg(C, "ALL", "collision").mean())
    col_static = float(C.static_collision.to_numpy(float).mean())
    col_log = float(C.log_collision.to_numpy(float).mean())
    pc3 = gate("PC3 STATIC collision >= 2x ALL", col_static / col_all,
               s2["PC3"]["collision_STATIC"] / s2["PC3"]["collision_ALL"], col_static >= 2 * col_all, s2["PC3"]["pass"])
    chk = gate("checker calibration LOG collision < 1%", col_log, s2["checker_calibration"]["collision_LOG"],
               col_log < 0.01, s2["checker_calibration"]["pass"])

    def pooled_stat(ws):
        return {"H4": ratio_pooled(ws, fl, fa, CITIES, nan_mean=True, keep=keep)}

    p = float(np.mean([point_city[c] for c in CITIES if keep[c]]))
    d = np.nanmean([draws_city[c] for c in CITIES if keep[c]], 0)

    def h4_replay():
        g = np.random.default_rng(0)
        boots = []
        for c in CITIES:
            a, lo_ = fa[c], fl[c]
            ii = g.integers(0, len(a), size=(REGISTERED_DRAWS, len(a)))
            am = a[ii].mean(1)
            boots.append(np.where(am > 0, (lo_[ii].mean(1) - am) / np.where(am > 0, am, 1), np.nan))
            del ii
        pooled = np.nanmean([b for b, c in zip(boots, CITIES) if keep[c]], 0)
        return [float(np.nanpercentile(pooled, 100 * (1 - LEVEL_S12) / 2)),
                float(np.nanpercentile(pooled, 100 * (1 + LEVEL_S12) / 2))]

    c4 = Component("H4", LEVEL_S12, p, d, s2["H4"]["pooled_rel_change"], s2["H4"]["ci_bonferroni"], [("lo", ">", 0.0)],
                   (">=", 0.10), resolver=lambda n, g: boot(st, pooled_stat, n, g)["H4"], replay=h4_replay)
    c4.evaluate(args.resolve_draws, rng)
    comps.append(c4)
    if not powered:
        v = "underpowered"
    elif not (pc3 and chk):
        v = "uninterpretable (PC3 or checker calibration failed)"
    else:
        v = "supported" if c4.passes() else "dead"
    verdict("H4", v, s2["H4"]["decision"])


# ---------------------------------------------------------------- Stage 3a -------------------------------------------
def stage3a(args, rng, comps):
    s3a = json.load(open("reports/stage3a/results.json"))
    for h in ("H5", "H6a", "H6b"):  # level check: the report stores its decision CI under the 98.75% key
        if "ci_98_75" not in s3a[h]:
            fail(f"{h} has no ci_98_75")
    P = pd.read_parquet("evals/stage3a/per_agent.parquet", columns=["scenario_id", "agent_id", "city", "focal",
                                                                     "planner_relevant", "speed", "predictor", "miss"])
    P = P[P.predictor.isin(["CV", "LANE", "STATIC", "ALL_s0", "ALL_s1", "ALL_s2"])]
    P["arm"] = np.where(P.predictor.str.startswith("ALL"), "ALL", P.predictor)
    n = P.groupby(["scenario_id", "agent_id", "arm"]).size().unstack(fill_value=0)
    if not ((n.ALL == 3) & (n.CV == 1) & (n.LANE == 1) & (n.STATIC == 1)).all():
        fail("Stage 3a pairing: every agent needs 3 ALL rows and one CV, LANE, STATIC row")
    W = P.groupby(["scenario_id", "agent_id", "arm"]).miss.mean().unstack()
    meta = P[P.predictor == "CV"].set_index(["scenario_id", "agent_id"])[["city", "focal", "planner_relevant", "speed"]]
    W = W.join(meta).reset_index()
    if W[["ALL", "CV", "LANE", "STATIC"]].isna().any().any():
        fail("Stage 3a missing outcomes")
    planner = W.planner_relevant.astype(bool)

    def clusters(mask, cols):
        out = {}
        for c in CITIES:
            sub = W[mask & (W.city == c)]
            if sub.empty:
                out[c] = None
                continue
            g = sub.groupby("scenario_id")[cols].sum()
            g["count"] = sub.groupby("scenario_id").size()
            out[c] = g
        return out

    # positive control: STATIC miss >= 1.1x CV miss for planner agents moving faster than 2 m/s
    mv = W[planner & (W.speed > 2)]
    cv_r, st_r = float(mv.CV.mean()), float(mv.STATIC.mean())
    rpc = s3a["positive_control"]
    pc = gate("3a PC STATIC >= 1.1x CV (planner, >2 m/s)", st_r / cv_r, rpc["static_miss_moving"] / rpc["cv_miss_moving"],
              cv_r > 0 and st_r >= 1.1 * cv_r, rpc["pass"])
    lane_misses = float(W.LANE[planner].sum())
    floor = gate("H5 A8 floor: LANE misses >= 100", lane_misses, s3a["H5"]["lane_misses"], lane_misses >= 100)

    out = {}
    specs = [("H5", planner, ["ALL", "LANE"], s3a["H5"], "pooled_reduction", (">=", 0.10)),
             ("H6a", planner & ~W.focal.astype(bool) & (W.speed < 0.5), ["ALL", "CV"], s3a["H6a"],
              "pooled_difference_ALL_minus_CV", (">=", 0.15)),
             ("H6b", planner & ~W.focal.astype(bool) & (W.speed >= 2.0), ["ALL", "CV"], s3a["H6b"],
              "pooled_reduction_ALL_vs_CV", (">=", 0.30))]
    for name, mask, cols, rep, key, bar in specs:
        fr = clusters(mask, cols)
        present = all(fr[c] is not None for c in CITIES)
        if not present:
            gate(f"{name} group present in all six cities", 0, None, False)
            out[name] = None
            continue
        arr = {c: {k: fr[c][k].to_numpy(float) for k in cols + ["count"]} for c in CITIES}
        st = {c: len(fr[c]) for c in CITIES}
        ctrl = cols[1]
        if name == "H6a":
            def stat(ws, arr=arr):
                return {"v": np.mean([ws[c] @ (arr[c]["ALL"] - arr[c]["CV"]) / (ws[c] @ arr[c]["count"]) for c in CITIES], 0)}
        else:
            def stat(ws, arr=arr, ctrl=ctrl):
                return {"v": ratio_pooled(ws, {c: arr[c]["ALL"] for c in CITIES}, {c: arr[c][ctrl] for c in CITIES},
                                          CITIES, sign=-1.0, offset=1.0)}
        pt = point_of(st, stat)["v"]
        d = boot(st, stat, args.draws, rng)["v"]
        defined_share = float(np.isfinite(d).mean())
        defined = bool(np.isfinite(pt) and defined_share >= 0.99)
        gate(f"{name} defined (finite point, >=99% defined replicates)", round(defined_share, 6),
             rep.get("bootstrap_defined_share"), defined)
        comp = Component(name, LEVEL_S3A, pt, d, rep[key], rep["ci_98_75"], [("lo", ">", 0.0)], bar,
                         resolver=lambda n_, g, st=st, stat=stat: boot(st, stat, n_, g)["v"])
        comp.evaluate(args.resolve_draws, rng)
        comps.append(comp)
        out[name] = (comp, defined)

    comp, defined = out["H5"]
    if not defined or not floor:
        v = "inconclusive"
    elif comp.passes() and pc:
        v = "supported"
    else:
        v = "killed" if pc else "uninterpretable (positive control failed)"
    verdict("H5", v, s3a["H5"]["decision"])
    for name in ("H6a", "H6b"):  # H6 decision: CI and magnitude only (no control gate registered); absent group -> inconclusive
        if out[name] is None or not out[name][1]:
            v = "inconclusive"
        else:
            v = "supported" if out[name][0].passes() else "killed"
        verdict(name, v, s3a[name]["decision"])


# ---------------------------------------------------------------- Stage 3 --------------------------------------------
def stage3(args, rng, comps):
    s3 = json.load(open("reports/stage3/results.json"))
    if not close(s3["ci_level"], LEVEL_S3, 1e-12):
        fail(f"Stage 3 report ci_level {s3['ci_level']} != registered {LEVEL_S3}")
    if s3["bootstraps"] != REGISTERED_DRAWS:
        fail("Stage 3 report bootstraps != 10000")
    C = pd.read_parquet("runs/stage3/closedloop_val.parquet")
    if len(C) != N_VAL or C.scenario_id.duplicated().any():
        fail("Stage 3 scenario count")
    city = C.city.astype(str).to_numpy()
    arms = ("ALL", "MULTI", "PATCH", "SHAM")
    br = {a: seed_avg(C, a, "unnecessary_hard_brake") for a in arms}
    co = {a: seed_avg(C, a, "collision") for a in arms}
    B = {c: {a: br[a][city == c] for a in arms} for c in CITIES}
    K = {c: {a: co[a][city == c] for a in arms} for c in CITIES}
    st = {c: int((city == c).sum()) for c in CITIES}
    rc = s3["closedloop"]["controls"]

    events = float(br["ALL"].sum())
    powered = gate("H7/H8 A8 floor: ALL brake events >= 100", events, None, events >= 100,
                   not s3["closedloop"]["H8"]["underpowered"])
    col_all, col_static, col_log = float(co["ALL"].mean()), float(C.static_collision.mean()), float(C.log_collision.mean())
    pc_static = gate("Stage 3 STATIC collision >= 2x ALL", col_static / col_all, rc["STATIC_over_ALL"],
                     col_all > 0 and col_static >= 2 * col_all, rc["STATIC_pass"])
    pc_log = gate("Stage 3 LOG collision < 1%", col_log, rc["LOG_collision"], col_log < 0.01, rc["LOG_pass"])
    undefined_cities = [c for c in CITIES if B[c]["ALL"].mean() == 0]
    gate("H7/H8 cities with zero ALL brake rate", len(undefined_cities), len(s3["closedloop"]["H8"]["undefined_cities"]), True)

    def stat(ws):
        o = {}
        for arm in ("MULTI", "PATCH"):
            o[f"{arm}_brake"] = ratio_pooled(ws, {c: B[c][arm] for c in CITIES}, {c: B[c]["ALL"] for c in CITIES}, CITIES,
                                             sign=-1.0, offset=1.0)
            o[f"{arm}_coll"] = np.mean([ws[c] @ (K[c][arm] - K[c]["ALL"]) / st[c] for c in CITIES], 0)
        return o

    pts = point_of(st, stat)
    d = boot(st, stat, args.draws, rng)
    claims = {}
    for name, arm in (("H7", "MULTI"), ("H8", "PATCH")):
        rep = s3["closedloop"][name]
        undef = float((~np.isfinite(d[f"{arm}_brake"])).mean())
        finite = gate(f"{name} undefined brake replicates <= 1%", undef, rep["undefined_bootstrap_share"], undef <= 0.01,
                      tol=0.01)
        cb = Component(f"{name} brake", LEVEL_S3, pts[f"{arm}_brake"], d[f"{arm}_brake"], rep["brake_reduction"],
                       rep["brake_ci"], [("lo", ">", 0.0)], (">=", 0.30),
                       resolver=lambda n, g, k=f"{arm}_brake": boot(st, stat, n, g)[k])
        cc = Component(f"{name} collision", LEVEL_S3, pts[f"{arm}_coll"], d[f"{arm}_coll"], rep["collision_difference"],
                       rep["collision_ci"], [("hi", "<", 0.003)], None,
                       resolver=lambda n, g, k=f"{arm}_coll": boot(st, stat, n, g)[k])
        for c in (cb, cc):
            c.evaluate(args.resolve_draws, rng)
            comps.append(c)
        claims[name] = (cb, cc, finite)
    # SHAM rule (Stage 3 deviation 1): H8 is stopped-specific only if SHAM's pooled reduction < half of PATCH's.
    sham_red = float(np.mean([1 - B[c]["SHAM"].mean() / B[c]["ALL"].mean() for c in CITIES if B[c]["ALL"].mean() > 0]))
    sham_ok = gate("H8 SHAM reduction < 0.5 x PATCH reduction", sham_red, rc["SHAM_brake_reduction"],
                   sham_red < 0.5 * pts["PATCH_brake"], rc["SHAM_pass"])
    tr = {a: float(seed_avg(C, a, "substituted_count").mean()) for a in ("PATCH", "SHAM")}
    gate("Stage 3 SHAM/PATCH substituted dose (info, deviation 3)", round(tr["SHAM"] / tr["PATCH"], 4), None, True)
    for name in ("H7", "H8"):
        cb, cc, finite = claims[name]
        both = cb.passes() and cc.passes()
        if not powered:
            v = "underpowered"
        elif undefined_cities or not finite:
            v = "inconclusive"
        elif not (pc_static and pc_log):
            v = "uninterpretable"
        elif name == "H8" and both and not sham_ok:
            v = "reduction real but not stopped-specific (dose-matched sham matched it)"
        else:
            v = "supported" if both else "killed"
        verdict(name, v, s3["closedloop"][name]["status"])

    # ---- H9 open loop ----
    OL = pd.read_parquet("runs/stage3/openloop_val.parquet", columns=["scenario_id", "agent_id", "city", "focal",
                                                                     "planner_relevant", "speed", "predictor", "miss"])
    OL = OL[OL.predictor.isin([f"{a}_s{s}" for a in ("ALL", "MULTI") for s in range(3)])]
    OL["arm"] = OL.predictor.str.split("_").str[0]
    cnt = OL.groupby(["scenario_id", "agent_id", "arm"]).size().unstack(fill_value=0)
    if not ((cnt.ALL == 3) & (cnt.MULTI == 3)).all():
        fail("Stage 3 open-loop pairing: every agent needs 3 ALL and 3 MULTI rows")
    G = OL.groupby(["scenario_id", "agent_id", "arm"]).miss.mean().unstack()
    info = OL[OL.predictor == "ALL_s0"].set_index(["scenario_id", "agent_id"])[["city", "focal", "planner_relevant", "speed"]]
    G = G.join(info).reset_index()
    G["d"] = G.MULTI - G.ALL
    if not np.isfinite(G.d).all():
        fail("Stage 3 open-loop NaN")
    groups = {"stopped": G.planner_relevant.astype(bool) & ~G.focal.astype(bool) & (G.speed < 0.5),
              "focal": G.focal.astype(bool)}
    h9 = s3["H9"]
    parts = {}
    for gname, mask in groups.items():
        fr = {c: G[mask & (G.city == c)].groupby("scenario_id").d.agg(["sum", "count"]) for c in CITIES}
        present = all(len(fr[c]) for c in CITIES)
        gate(f"H9 {gname} group present in all six cities", int(present), None, present)
        arr = {c: (fr[c]["sum"].to_numpy(float), fr[c]["count"].to_numpy(float)) for c in CITIES}
        sts = {c: len(fr[c]) for c in CITIES}

        def st9(ws, arr=arr):
            return {"v": np.mean([ws[c] @ arr[c][0] / (ws[c] @ arr[c][1]) for c in CITIES], 0)}

        key = "stopped_nonfocal" if gname == "stopped" else "focal"
        rules = ([("hi", "<", 0.0)], ("<=", -0.15)) if gname == "stopped" else ([("hi", "<", 0.02)], None)
        comp = Component(f"H9 {gname}", LEVEL_S3, point_of(sts, st9)["v"], boot(sts, st9, args.draws, rng)["v"],
                         h9[f"{key}_difference"], h9[f"{key}_ci"], rules[0], rules[1],
                         resolver=lambda n, g, sts=sts, st9=st9: boot(sts, st9, n, g)["v"])
        comp.evaluate(args.resolve_draws, rng)
        comps.append(comp)
        parts[gname] = (comp, present)
    if not all(p for _, p in parts.values()):
        v = "inconclusive"
    else:
        v = "supported" if parts["stopped"][0].passes() and parts["focal"][0].passes() else "killed"
    verdict("H9", v, h9["status"])


# ---------------------------------------------------------------- Stage 4 --------------------------------------------
def stage4(args, rng, comps):
    s4 = json.load(open("reports/stage4/results.json"))
    if not close(s4["ci_level"], LEVEL_S4, 1e-12):
        fail(f"Stage 4 report ci_level {s4['ci_level']} != registered {LEVEL_S4}")
    if not close(s4["descriptive_ci_level"], LEVEL_S4_DESCRIPTIVE, 1e-12):
        fail("Stage 4 descriptive level != 98.75%")
    if s4["bootstraps"] != REGISTERED_DRAWS:
        fail("Stage 4 report bootstraps != 10000")
    C = pd.read_parquet("runs/stage4/closedloop_pool.parquet")
    ids = sorted(C.scenario_id.astype(str))
    digest = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    # reported_pass=True: the registered pool must match, so a count or hash mismatch fails the audit
    gate("Stage 4 pool: 8,140 unique IDs, registered SHA-256", len(set(ids)), N_POOL,
         len(set(ids)) == N_POOL and digest == POOL_SHA256, reported_pass=True)
    arms = ("ALL", "PATCH", "TRIM", "SHAM2", "MIX")
    cols = {f"{a}_b": seed_avg(C, a, "unnecessary_hard_brake") for a in arms}
    cols |= {f"{a}_c": seed_avg(C, a, "collision") for a in arms}
    OL = pd.read_parquet("runs/stage4/openloop_pool.parquet")
    if not OL.focal.astype(bool).all() or OL.duplicated(["scenario_id", "predictor"]).any():
        fail("Stage 4 open-loop file must hold one focal row per scenario and predictor")
    F = OL.pivot(index="scenario_id", columns="predictor", values="miss")
    need = [f"{a}_s{s}" for a in ("ALL", "MIX") for s in range(3)]
    if F[need].isna().any().any() or set(F.index.astype(str)) != set(ids):
        fail("Stage 4 focal pairing")
    F = F.reindex(C.scenario_id)
    cols["focal_ALL"] = F[[f"ALL_s{s}" for s in range(3)]].to_numpy(float).mean(1)
    cols["focal_MIX"] = F[[f"MIX_s{s}" for s in range(3)]].to_numpy(float).mean(1)
    names = list(cols)
    M = np.column_stack([cols[k] for k in names])
    city = C.city.astype(str).to_numpy()
    if set(city) != set(CITIES):
        fail("Stage 4 missing city")
    Mc = {c: M[city == c] for c in CITIES}
    st = {c: len(Mc[c]) for c in CITIES}
    ix = {k: i for i, k in enumerate(names)}

    def effects(means):
        """means: {col: array}; returns the eight registered components plus SHAM2's reduction (for the gap)."""
        den = means["ALL_b"]
        e = {f"{a}_red": np.where(den > 0, 1 - means[f"{a}_b"] / np.where(den > 0, den, 1), np.nan)
             for a in ("PATCH", "TRIM", "SHAM2", "MIX")}
        e["gap"] = e["PATCH_red"] - e["SHAM2_red"]
        for a in ("PATCH", "TRIM", "MIX"):
            e[f"{a}_coll"] = means[f"{a}_c"] - means["ALL_c"]
        e["focal"] = means["focal_MIX"] - means["focal_ALL"]
        return e

    def stat(ws):
        per = [effects({k: ws[c] @ Mc[c][:, ix[k]] / st[c] for k in names}) for c in CITIES]
        return {k: np.mean([p[k] for p in per], 0) for k in per[0]}

    pts = point_of(st, stat)
    d = boot(st, stat, args.draws, rng)
    ctl = s4["controls"]
    events = float(cols["ALL_b"].sum())
    powered = gate("Stage 4 A8 floor: ALL brake events >= 100", events, ctl["ALL_brake_events"], events >= 100,
                   ctl["power_pass"])
    zero_city = [c for c in CITIES if Mc[c][:, ix["ALL_b"]].mean() == 0]
    undef = max(float((~np.isfinite(d[k])).mean()) for k in ("PATCH_red", "TRIM_red", "SHAM2_red", "MIX_red", "gap"))
    finite = gate("Stage 4 zero-ALL-brake cities = 0, undefined <= 1%", undef, s4["undefined_bootstrap_share"],
                  not zero_city and undef <= 0.01, tol=0.01)
    col_all = float(cols["ALL_c"].mean())
    col_static, col_log = float(C.static_collision.mean()), float(C.log_collision.mean())
    pc_static = gate("Stage 4 STATIC collision >= 2x ALL", col_static / col_all, ctl["STATIC_over_ALL"],
                     col_all > 0 and col_static >= 2 * col_all, ctl["STATIC_pass"])
    pc_log = gate("Stage 4 LOG collision < 1%", col_log, ctl["LOG_collision"], col_log < 0.01, ctl["LOG_pass"])
    # Dose gate (A7): SHAM2 never above PATCH at any scenario, seed and replan; realised total >= 99% of PATCH.
    tp = ts = 0
    above = 0
    for s in range(3):
        for t in (49, 59, 69, 79, 89, 99):
            p_, s_ = C[f"PATCH_s{s}_dose_t{t}"].to_numpy(int), C[f"SHAM2_s{s}_dose_t{t}"].to_numpy(int)
            above += int((s_ > p_).sum())
            tp, ts = tp + int(p_.sum()), ts + int(s_.sum())
    gate("Stage 4 SHAM2 dose above PATCH (must be 0; aborts)", above, None, above == 0)
    if above:
        fail("Stage 4 dose abort: SHAM2 above PATCH")
    gate("Stage 4 PATCH total dose", tp, ctl["dose_PATCH_total"], True)
    gate("Stage 4 SHAM2 total dose", ts, ctl["dose_SHAM2_total"], True)
    dose_ok = gate("Stage 4 dose gate SHAM2/PATCH >= 0.99", ts / tp, ctl["dose_SHAM2_over_PATCH"], ts / tp >= 0.99,
                   ctl["dose_dose_pass"])

    replay_cache = {}

    def replay(key):
        """Replay the report's resampling: default_rng(4404), per city in order, chunks of 128 draws."""
        if not replay_cache:
            g = np.random.default_rng(4404)
            acc = {}
            for c in CITIES:
                m = Mc[c]
                pieces = []
                for start in range(0, REGISTERED_DRAWS, 128):
                    k = min(128, REGISTERED_DRAWS - start)
                    idx = g.integers(0, len(m), size=(k, len(m)))
                    avg = m[idx].mean(1)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        pieces.append(effects({n: avg[:, ix[n]] for n in names}))
                for kk in pieces[0]:
                    acc.setdefault(kk, []).append(np.concatenate([p[kk] for p in pieces]))
            for kk, v in acc.items():
                replay_cache[kk] = np.mean(np.stack(v), 0)
        return interval(replay_cache[key], LEVEL_S4)

    eff = s4["effects"]
    spec = {"H10 brake": ("PATCH_red", "PATCH_reduction", [("lo", ">", 0.0)], (">=", 0.30)),
            "H10 collision": ("PATCH_coll", "PATCH_collision_diff", [("hi", "<", 0.003)], None),
            "H11 gap": ("gap", "sham_gap", [("lo", ">", 0.0)], (">=", 0.20)),
            "H12 brake": ("TRIM_red", "TRIM_reduction", [("lo", ">", 0.0)], (">=", 0.30)),
            "H12 collision": ("TRIM_coll", "TRIM_collision_diff", [("hi", "<", 0.003)], None),
            "H13 brake": ("MIX_red", "MIX_reduction", [("lo", ">", 0.0)], (">=", 0.30)),
            "H13 collision": ("MIX_coll", "MIX_collision_diff", [("hi", "<", 0.003)], None),
            "H13 focal": ("focal", "focal_miss_diff", [("hi", "<", 0.02)], None)}
    done = {}
    for label, (k, rk, rules, bar) in spec.items():
        comp = Component(label, LEVEL_S4, pts[k], d[k], eff[rk]["point"], eff[rk]["ci"], rules, bar,
                         resolver=lambda n, g, k=k: boot(st, stat, n, g)[k], replay=lambda k=k: replay(k))
        comp.evaluate(args.resolve_draws, rng)
        comps.append(comp)
        done[label] = comp

    def status(passed):
        if not powered:
            return "underpowered"
        if not finite:
            return "inconclusive"
        if not (pc_static and pc_log):
            return "uninterpretable"
        return "supported" if passed else "killed"

    for h in ("H10", "H12", "H13"):
        parts = [c for lbl, c in done.items() if lbl.startswith(h + " ")]
        verdict(h, status(all(c.passes() for c in parts)), s4["claims"][h]["status"])
    v11 = status(done["H11 gap"].passes())
    if v11 in ("supported", "killed") and not dose_ok:
        v11 = "uninterpretable (SHAM2 realised dose below 99% of PATCH)"
    verdict("H11", v11, s4["claims"]["H11"]["status"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=REGISTERED_DRAWS, help="fresh bootstrap draws per component")
    ap.add_argument("--cf-draws", type=int, default=4000, help="fresh draws for the H2/H3 capture-fraction bootstrap")
    ap.add_argument("--resolve-draws", type=int, default=100000, help="draws used to resolve NEAR-BOUNDARY bounds")
    ap.add_argument("--seed", type=int, default=20260930)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    comps: list[Component] = []
    for fn in (stage1, stage2, stage3a, stage3, stage4):
        try:
            fn(args, rng, comps)
        except Exception as exc:  # a crash is a failed audit, never a pass
            fail(f"{fn.__name__} raised {type(exc).__name__}: {exc}")
            print(f"ERROR in {fn.__name__}: {type(exc).__name__}: {exc}")
    print("== registered gates (controls, floors, undefined rules, dose) ==")
    print("\n".join(GATES))
    print("== decision components ==")
    print("\n".join(c.line() for c in comps))
    print("== claim verdicts ==")
    print("\n".join(VERDICTS))
    near = [c.name for c in comps if c.near]
    print(f"NEAR-BOUNDARY components: {', '.join(near) if near else 'none'}")
    print(f"draws {args.draws}, cf-draws {args.cf_draws}, resolve-draws {args.resolve_draws}, seed {args.seed}, "
          f"{time.time() - t0:.0f} s")
    if len(VERDICTS) != 14:  # H1 to H13, with H6 registered as H6a and H6b
        fail(f"expected 14 verdicts (H1 to H13, H6 as H6a and H6b), got {len(VERDICTS)}")
    if FAIL:
        print("MISMATCH: " + "; ".join(FAIL))
        sys.exit(1)
    print("ALL REGISTERED DECISIONS REPRODUCED (H1 to H13: every gate, component and verdict)")


if __name__ == "__main__":
    main()

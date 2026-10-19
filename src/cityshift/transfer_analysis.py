"""Paired planning-transfer estimates, development controls, and bounded joint gates.

Scenario bootstraps condition on the observed cities and fixed checkpoints.
Contacts are observed nonreactive replay contacts, not physical safety outcomes.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


CITIES = ("austin", "dearborn", "miami", "palo-alto", "pittsburgh", "washington-dc")
OBJECTIVES = ("weighted", "risk_budget")
LEARNED = ("NATIVE", "PATCH", "CALIB", "SELECTIVE")
ANALYTIC = ("oracle", "cv", "blind")
POLICIES = (*LEARNED, *ANALYTIC)
LEVEL = .9875
TAIL = (1 - LEVEL) / 2
BASE_METRICS = ("hard_brake", "observed_contact", "observed_near_conflict", "min_clearance_m",
                "progress_fraction", "progress_m", "mean_abs_jerk_mps3")
OPTIONAL_METRICS = ("genuine_conflict", "excessive_brake", "complete_decisions", "eligible_decisions")
BINARY = ("hard_brake", "observed_contact", "observed_near_conflict", "genuine_conflict", "excessive_brake")


def clopper_pearson_upper(events: int, trials: int, alpha: float = TAIL) -> float:
    """One-sided exact binomial upper bound, with no scipy dependency."""
    if not isinstance(events, (int, np.integer)) or not isinstance(trials, (int, np.integer)):
        raise ValueError("binomial counts must be integers")
    if trials < 1 or not 0 <= events <= trials or not 0 < alpha < 1:
        raise ValueError("invalid binomial counts or tail probability")
    if events == trials:
        return 1.0
    if events == 0:
        return -math.expm1(math.log(alpha) / trials)
    k = np.arange(events + 1, dtype=float)
    log_combination = np.array([math.lgamma(trials + 1) - math.lgamma(int(i) + 1)
                                - math.lgamma(trials - int(i) + 1) for i in k])
    lo, hi = 0.0, 1.0
    for _ in range(64):
        p = (lo + hi) / 2
        terms = log_combination + k * math.log(p) + (trials - k) * math.log1p(-p)
        largest = float(terms.max())
        log_cdf = largest + math.log(float(np.exp(terms - largest).sum()))
        if log_cdf > math.log(alpha):
            lo = p
        else:
            hi = p
    return hi


def harmful_discordance_upper(harmful: np.ndarray, cities: np.ndarray) -> dict:
    """Conservative city-equal harm bound with simultaneous exact city coverage.

    Harm is any harmful paired seed within a scenario. Its probability bounds
    the seed-averaged event-rate increase from above, without assuming independent
    model seeds. Simultaneous city bounds use a Bonferroni allocation of TAIL.
    """
    harmful, cities = np.asarray(harmful), np.asarray(cities)
    if harmful.ndim != 1 or cities.shape != harmful.shape or not len(harmful):
        raise ValueError("paired harmful discordances and cities must be nonempty vectors")
    if not np.isin(harmful, [0, 1]).all():
        raise ValueError("harmful discordances must be binary")
    observed_cities = sorted(set(cities.tolist()))
    records = []
    for city in observed_cities:
        values = harmful[cities == city]
        records.append({"city": str(city), "trials": len(values), "harmful_scenarios": int(values.sum()),
                        "upper": clopper_pearson_upper(int(values.sum()), len(values), TAIL / len(observed_cities))})
    return {"upper": float(np.mean([value["upper"] for value in records])),
            "one_sided_confidence": 1 - TAIL, "city_bounds": records,
            "method": "city-equal mean of simultaneous exact harmful-scenario binomial bounds"}


def _prepare(rows: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    data = rows.copy()
    if "hard_brake" not in data and "planner_hard_brake" in data:
        data["hard_brake"] = data.planner_hard_brake
    required = {"scenario_id", "city", "objective", "policy", "seed", *BASE_METRICS}
    if not required <= set(data):
        raise ValueError(f"missing rollout columns: {sorted(required - set(data))}")
    if data.empty or data[list(required - set(BASE_METRICS))].isna().any().any():
        raise ValueError("rollout memberships cannot be empty or missing")
    if not set(data.objective) <= set(OBJECTIVES) or not set(data.policy) <= set(POLICIES):
        raise ValueError("unknown objective or policy")
    if not set(data.city) <= set(CITIES) or (data.groupby("scenario_id").city.nunique() != 1).any():
        raise ValueError("unknown city or inconsistent scenario city")
    seeds = data.seed.to_numpy(float)
    if not np.isfinite(seeds).all() or not np.equal(seeds, np.floor(seeds)).all():
        raise ValueError("seed identifiers must be integers")
    data["seed"] = seeds.astype(int)
    if data.duplicated(["scenario_id", "objective", "policy", "seed"]).any():
        raise ValueError("duplicate rollout membership")
    scene_ids = set(data.scenario_id)
    for objective, objective_rows in data.groupby("objective"):
        if set(objective_rows.policy) != set(POLICIES):
            raise ValueError(f"incomplete policy family for {objective}")
        for policy, policy_rows in objective_rows.groupby("policy"):
            if set(policy_rows.scenario_id) != scene_ids:
                raise ValueError("policy/objective scenario cohorts differ")
            expected = {0, 1, 2} if policy in LEARNED else {-1}
            if any(set(values) != expected for _, values in policy_rows.groupby("scenario_id").seed):
                raise ValueError("all three learned seeds or one analytic seed -1 are required")
    metrics = [*BASE_METRICS, *(name for name in OPTIONAL_METRICS if name in data)]
    if {"excessive_brake", "genuine_conflict"}.intersection(data) \
            and not {"complete_decisions", "eligible_decisions"} <= set(data):
        raise ValueError("complete-future outcomes require complete/eligible decision support")
    for metric in metrics:
        values = data[metric].to_numpy(float)
        missing_allowed = metric in ("min_clearance_m", "progress_fraction", "excessive_brake", "genuine_conflict")
        if np.isinf(values).any() or (not missing_allowed and not np.isfinite(values).all()):
            raise ValueError(f"nonfinite rollout metric: {metric}")
        if metric in BINARY and not np.isin(values[np.isfinite(values)], [0, 1]).all():
            raise ValueError(f"nonbinary rollout metric: {metric}")
        if missing_allowed:
            # Missing geometry/progress is allowed only on common scene support.
            support = data.assign(_finite=np.isfinite(values)).groupby(["scenario_id", "objective"])._finite.nunique()
            if (support != 1).any():
                raise ValueError(f"policy/seed missing support differs for {metric}")
    if "complete_decisions" in data:
        if "eligible_decisions" not in data:
            raise ValueError("complete decisions require eligible decisions")
        complete, eligible = data.complete_decisions.to_numpy(float), data.eligible_decisions.to_numpy(float)
        if not np.equal(complete, np.floor(complete)).all() or not np.equal(eligible, 3).all():
            raise ValueError("decision support must count the three complete-horizon replans")
        if (complete < 0).any() or (complete > eligible).any():
            raise ValueError("invalid complete decision count")
        if (data.groupby(["scenario_id", "objective"]).complete_decisions.nunique() != 1).any():
            raise ValueError("complete decision support must be common across policies and seeds")
        for metric in ("excessive_brake", "genuine_conflict"):
            if metric in data:
                if not np.array_equal(np.isfinite(data[metric].to_numpy(float)), complete > 0):
                    raise ValueError(f"{metric} must be unknown exactly when no complete decision exists")
    return data, metrics


def _bootstrap_means(matrix: np.ndarray, cities: np.ndarray, samples: int, rng) -> dict:
    common = np.isfinite(matrix).all(axis=1)
    point, draws = np.zeros(len(POLICIES)), np.zeros((samples, len(POLICIES)))
    observed_cities = sorted(set(cities.tolist()))
    counts = {str(city): int(((cities == city) & common).sum()) for city in observed_cities}
    if any(count == 0 for count in counts.values()):
        return {"point": np.full(len(POLICIES), np.nan),
                "draws": np.full((samples, len(POLICIES)), np.nan), "counts": counts}
    for city in observed_cities:
        values = matrix[(cities == city) & common]
        point += values.mean(0) / len(observed_cities)
        for start in range(0, samples, 128):
            stop = min(start + 128, samples)
            index = rng.integers(0, len(values), size=(stop - start, len(values)))
            draws[start:stop] += values[index].mean(1) / len(observed_cities)
    return {"point": point, "draws": draws, "counts": counts}


def _estimate(point: float, draws: np.ndarray) -> dict:
    finite = np.isfinite(draws)
    return {"point": float(point) if np.isfinite(point) else None,
            "ci98_75": [float(value) for value in np.quantile(draws, [TAIL, 1 - TAIL])] if finite.all() else [None, None],
            "undefined_share": float(1 - finite.mean())}


def _upper_below(estimate: dict, margin: float) -> bool:
    upper = estimate["ci98_75"][1]
    return upper is not None and upper < margin


def analyze_rollouts(rows: pd.DataFrame, bootstrap_samples: int = 10000, seed: int = 20261019,
                     departure_scenes: int | None = None, departure_summary: dict | None = None,
                     phase: str = "test") -> dict:
    """Analyze paired rolls without fitting or selecting a policy on their outcomes.

    ``departure_scenes`` counts independent scenes containing true departures on
    the same cohort. Optional ``departure_summary[policy]`` supplies the external
    heldout ``recall_loss_ci98_75`` versus NATIVE; missing evidence leaves the
    fourth joint component pending. SELECTIVE is primary; CALIB is exploratory.
    ``phase='pilot'`` reports feasibility, never a final supported claim.
    """
    if not isinstance(bootstrap_samples, int) or bootstrap_samples < 1 or phase not in ("pilot", "test"):
        raise ValueError("invalid bootstrap count or phase")
    if departure_scenes is not None and (not isinstance(departure_scenes, (int, np.integer)) or departure_scenes < 0):
        raise ValueError("departure scene count must be a nonnegative integer")
    departure_scenes = int(departure_scenes) if departure_scenes is not None else None
    data, metrics = _prepare(rows)
    rng = np.random.default_rng(seed)
    averages = data.groupby(["scenario_id", "city", "objective", "policy"], sort=True)[metrics].mean().reset_index()
    result = {"schema": 1, "phase": phase, "bootstrap_samples": bootstrap_samples, "bootstrap_seed": seed,
              "confidence_level": LEVEL, "primary_policy": "SELECTIVE", "departure_scenes": departure_scenes,
              "assumptions": ["fixed checkpoints and observed cities; learned seeds averaged within each scenario",
                              "paired scenario bootstrap within city; equal city weights",
                              "complete decision support is shared by all policies and seeds within each objective",
                              "complete-future conflict and excess-brake events are unknown when no decision is complete",
                              "observed contacts reflect logged nonreactive actors and incomplete observation",
                              "min clearance and short-route progress use reported common finite support",
                              "four components per primary joint decision; CALIB comparisons are exploratory"],
              "objectives": {}}
    objective_feasibility = []
    for objective in OBJECTIVES:
        group = averages[averages.objective == objective]
        if group.empty:
            objective_feasibility.append(False)
            result["objectives"][objective] = {"status": "missing", "feasibility": {"passed": False}}
            continue
        scene_order = sorted(group.scenario_id.unique())
        cities = group[group.policy == "NATIVE"].set_index("scenario_id").loc[scene_order, "city"].to_numpy()
        matrices, estimates = {}, {}
        for metric in metrics:
            matrix = group.pivot(index="scenario_id", columns="policy", values=metric).loc[scene_order, list(POLICIES)]
            matrices[metric] = matrix.to_numpy(float)
            estimates[metric] = _bootstrap_means(matrices[metric], cities, bootstrap_samples, rng)
        rates = {policy: {metric: {"city_equal_mean": _estimate(estimates[metric]["point"][j],
                                                              estimates[metric]["draws"][:, j])["point"],
                                   "finite_scenarios_by_city": estimates[metric]["counts"]}
                          for metric in metrics} for j, policy in enumerate(POLICIES)}
        brake = "excessive_brake" if "excessive_brake" in metrics else "hard_brake"
        native_index, oracle_index, blind_index = [POLICIES.index(policy) for policy in ("NATIVE", "oracle", "blind")]
        native_events = int((matrices[brake][:, native_index] > 0).sum())
        native_rate = estimates[brake]["point"][native_index]
        oracle_rate = estimates[brake]["point"][oracle_index]
        blind_harm = int(((matrices["observed_contact"][:, blind_index] > 0)
                         & (matrices["observed_contact"][:, oracle_index] == 0)).sum())
        gates = {"all_six_cities": set(cities) == set(CITIES), "complete_future_excessive_brake_available": brake == "excessive_brake",
                 "native_braking_scenes_at_least_20": native_events >= 20,
                 "departure_scenes_at_least_20": departure_scenes is not None and departure_scenes >= 20,
                 "native_oracle_brake_discrimination": bool(native_rate > 0 and native_rate >= 2 * oracle_rate),
                 "blind_oracle_harmful_contact_scenes_at_least_5": blind_harm >= 5}
        feasible = all(gates.values())
        objective_feasibility.append(feasible)
        final_power = native_events >= 100 and departure_scenes is not None and departure_scenes >= 100
        comparisons = {}
        raw_objective = data[data.objective == objective]
        for policy in ("PATCH", "CALIB", "SELECTIVE"):
            comparisons[policy] = {}
            for reference in ("NATIVE", "PATCH"):
                if policy == reference:
                    continue
                j, k = POLICIES.index(policy), POLICIES.index(reference)
                effects = {}
                for metric, summary in estimates.items():
                    effects[metric] = _estimate(summary["point"][j] - summary["point"][k],
                                                summary["draws"][:, j] - summary["draws"][:, k])
                effects["progress_loss"] = _estimate(
                    -effects["progress_fraction"]["point"] if effects["progress_fraction"]["point"] is not None else np.nan,
                    estimates["progress_fraction"]["draws"][:, k] - estimates["progress_fraction"]["draws"][:, j])
                pair = raw_objective[raw_objective.policy.isin([policy, reference])].pivot(
                    index=["scenario_id", "seed"], columns="policy", values="observed_contact")
                harmful = (pair[policy] > pair[reference]).groupby("scenario_id").any().loc[scene_order].to_numpy()
                exact = harmful_discordance_upper(harmful, cities)
                bootstrap_upper = effects["observed_contact"]["ci98_75"][1]
                effects["observed_contact"]["harmful_discordance_bound"] = exact
                effects["observed_contact"]["harm_upper"] = max(bootstrap_upper, exact["upper"]) \
                    if bootstrap_upper is not None else None
                comparisons[policy][reference] = effects
        joint = {}
        for policy in ("SELECTIVE", "CALIB"):
            effects = comparisons[policy]["NATIVE"]
            planning_gates = {"less_excessive_braking": _upper_below(effects[brake], 0),
                              "progress_loss_below_0_02": _upper_below(effects["progress_loss"], .02),
                              "contact_harm_below_0_003": effects["observed_contact"]["harm_upper"] is not None
                              and effects["observed_contact"]["harm_upper"] < .003}
            recall_ci = (departure_summary or {}).get(policy, {}).get("recall_loss_ci98_75")
            if recall_ci is not None:
                recall_ci = np.asarray(recall_ci, dtype=float)
                if recall_ci.shape != (2,) or not np.isfinite(recall_ci).all() or recall_ci[0] > recall_ci[1]:
                    raise ValueError("departure recall loss requires a finite ordered 98.75% interval")
            departure_pass = bool(recall_ci[1] < .02) if recall_ci is not None else None
            planning_pass = all(planning_gates.values())
            if phase == "pilot":
                status = "development_only"
            elif not feasible:
                status = "uninterpretable_controls_or_feasibility"
            elif not final_power or departure_pass is None:
                status = "inconclusive_power_or_missing_departure"
            else:
                status = "supported" if planning_pass and departure_pass else "not_supported"
            joint[policy] = {"scope": "primary" if policy == "SELECTIVE" else "exploratory",
                             "planning_gates": planning_gates, "planning_joint3": planning_pass,
                             "departure_recall_loss_ci98_75": recall_ci.tolist() if recall_ci is not None else None,
                             "departure_recall_loss_below_0_02": departure_pass,
                             "joint_pending_departure": departure_pass is None,
                             "joint4": bool(phase == "test" and feasible and final_power
                                            and planning_pass and departure_pass is True),
                             "status": status}
        result["objectives"][objective] = {"scenarios": len(scene_order), "brake_metric": brake, "rates": rates,
                                          "feasibility": {"passed": feasible, "gates": gates,
                                                          "native_braking_scenes": native_events,
                                                          "native_brake_scored_scenes": int(np.isfinite(
                                                              matrices[brake][:, native_index]).sum()),
                                                          "blind_oracle_harmful_contact_scenes": blind_harm},
                                          "final_power": {"passed": bool(final_power), "native_minimum": 100,
                                                          "departure_minimum": 100},
                                          "comparisons": comparisons, "joint": joint}
    result["feasibility_passed_both_objectives"] = all(objective_feasibility)
    result["full_test_gate"] = "proceed" if all(objective_feasibility) else "stop_insufficient_controls_or_events"
    return result

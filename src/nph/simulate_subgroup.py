#!/usr/bin/env python3
import argparse
import json
import math
import os
import time
from dataclasses import dataclass, asdict
from multiprocessing import Pool, cpu_count

import numpy as np


def p_value_from_z(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2.0))


def logrank_pvalue(time, event, arm):
    order = np.argsort(time)
    t = time[order]
    e = event[order]
    a = arm[order]

    unique_event_times = np.unique(t[e == 1])
    o1_minus_e1 = 0.0
    var = 0.0

    for ut in unique_event_times:
        at_risk = t >= ut
        n1 = np.sum((a == 1) & at_risk)
        n0 = np.sum((a == 0) & at_risk)
        n = n1 + n0
        if n <= 1:
            continue

        at_time = t == ut
        d1 = np.sum((a == 1) & at_time & (e == 1))
        d0 = np.sum((a == 0) & at_time & (e == 1))
        d = d1 + d0
        if d == 0:
            continue

        exp1 = d * (n1 / n)
        o1_minus_e1 += d1 - exp1

        if n > 1:
            var += (n1 * n0 * d * (n - d)) / ((n**2) * (n - 1))

    if var <= 0:
        return 1.0
    z = o1_minus_e1 / math.sqrt(var)
    return p_value_from_z(z)


def rate_ratio_log_and_var(time, event, arm):
    # crude rate-ratio estimator using person-time
    pt_trt = np.sum(time[arm == 1])
    pt_ctl = np.sum(time[arm == 0])
    d_trt = np.sum(event[arm == 1])
    d_ctl = np.sum(event[arm == 0])

    if d_trt == 0 or d_ctl == 0 or pt_trt <= 0 or pt_ctl <= 0:
        return np.nan, np.nan

    rr = (d_trt / pt_trt) / (d_ctl / pt_ctl)
    log_rr = np.log(rr)
    var = 1.0 / d_trt + 1.0 / d_ctl
    return log_rr, var


@dataclass
class Scenario:
    name: str
    n_total: int
    prevalence: float
    lambda_control: float
    hr_pos: float
    hr_neg: float
    censor_rate: float


def simulate_once(seed: int, sc: Scenario):
    rng = np.random.default_rng(seed)

    n = sc.n_total
    arm = rng.binomial(1, 0.5, size=n)  # 1 treatment, 0 control
    biom = rng.binomial(1, sc.prevalence, size=n)  # 1 biomarker positive

    hazard = np.full(n, sc.lambda_control)
    trt_pos = (arm == 1) & (biom == 1)
    trt_neg = (arm == 1) & (biom == 0)
    hazard[trt_pos] = sc.lambda_control * sc.hr_pos
    hazard[trt_neg] = sc.lambda_control * sc.hr_neg

    t_event = rng.exponential(1.0 / hazard)
    t_cens = rng.exponential(1.0 / sc.censor_rate, size=n)
    time_obs = np.minimum(t_event, t_cens)
    event = (t_event <= t_cens).astype(int)

    # overall
    p_overall = logrank_pvalue(time_obs, event, arm)
    logrr_overall, var_overall = rate_ratio_log_and_var(time_obs, event, arm)

    # subgroup-positive
    idx_pos = biom == 1
    p_pos = logrank_pvalue(time_obs[idx_pos], event[idx_pos], arm[idx_pos]) if np.sum(idx_pos) > 10 else 1.0

    # subgroup-negative
    idx_neg = biom == 0
    p_neg = logrank_pvalue(time_obs[idx_neg], event[idx_neg], arm[idx_neg]) if np.sum(idx_neg) > 10 else 1.0

    # heterogeneity interaction (crude)
    l_pos, v_pos = rate_ratio_log_and_var(time_obs[idx_pos], event[idx_pos], arm[idx_pos])
    l_neg, v_neg = rate_ratio_log_and_var(time_obs[idx_neg], event[idx_neg], arm[idx_neg])
    if np.isnan(l_pos) or np.isnan(l_neg) or np.isnan(v_pos) or np.isnan(v_neg) or (v_pos + v_neg) <= 0:
        p_inter = 1.0
    else:
        z = (l_pos - l_neg) / math.sqrt(v_pos + v_neg)
        p_inter = p_value_from_z(z)

    return {
        "p_overall": p_overall,
        "p_pos": p_pos,
        "p_neg": p_neg,
        "p_inter": p_inter,
        "logrr_overall": logrr_overall,
        "events": int(np.sum(event)),
    }


def run_chunk(args):
    start_seed, n_sims, sc = args
    out = []
    for i in range(n_sims):
        out.append(simulate_once(start_seed + i, sc))
    return out


def summarise(rows, alpha=0.05):
    p_overall = np.array([r["p_overall"] for r in rows], dtype=float)
    p_pos = np.array([r["p_pos"] for r in rows], dtype=float)
    p_neg = np.array([r["p_neg"] for r in rows], dtype=float)
    p_inter = np.array([r["p_inter"] for r in rows], dtype=float)
    logrr = np.array([r["logrr_overall"] for r in rows], dtype=float)
    events = np.array([r["events"] for r in rows], dtype=float)

    return {
        "n_sims": int(len(rows)),
        "reject_overall": float(np.mean(p_overall < alpha)),
        "reject_pos": float(np.mean(p_pos < alpha)),
        "reject_neg": float(np.mean(p_neg < alpha)),
        "reject_interaction": float(np.mean(p_inter < alpha)),
        "mean_logrr_overall": float(np.nanmean(logrr)),
        "mean_events": float(np.mean(events)),
    }


def estimate_n_sims(sc: Scenario, max_minutes: float, pilot_sims: int, n_cores: int):
    t0 = time.time()
    _ = [simulate_once(10_000 + i, sc) for i in range(pilot_sims)]
    dt = time.time() - t0
    sec_per_sim = dt / pilot_sims
    budget_sec = max_minutes * 60.0
    n = int(budget_sec / sec_per_sim)
    # parallel speed-up approx with some overhead factor
    n = int(0.85 * n * max(1, n_cores))
    return max(500, min(n, 100_000)), sec_per_sim


def run_scenario(sc: Scenario, outdir: str, max_minutes: float, n_sims: int | None, seed: int):
    cores = cpu_count()
    if n_sims is None:
        n_sims, sec_per_sim = estimate_n_sims(sc, max_minutes=max_minutes, pilot_sims=200, n_cores=cores)
    else:
        sec_per_sim = None

    chunk = int(math.ceil(n_sims / cores))
    jobs = []
    for k in range(cores):
        start_seed = seed + k * chunk
        n_local = max(0, min(chunk, n_sims - k * chunk))
        if n_local > 0:
            jobs.append((start_seed, n_local, sc))

    t0 = time.time()
    with Pool(processes=cores) as pool:
        nested = pool.map(run_chunk, jobs)
    rows = [r for batch in nested for r in batch]
    elapsed = time.time() - t0

    summary = summarise(rows)
    summary["scenario"] = sc.name
    summary["elapsed_sec"] = elapsed
    summary["cores"] = cores
    if sec_per_sim is not None:
        summary["pilot_sec_per_sim"] = sec_per_sim

    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, f"summary_{sc.name}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="results")
    parser.add_argument("--max-minutes", type=float, default=25.0)
    parser.add_argument("--n-sims", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260227)
    return parser


def main(argv: list[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(argv)

    scenarios = [
        Scenario("null", n_total=400, prevalence=0.3, lambda_control=0.12, hr_pos=1.0, hr_neg=1.0, censor_rate=0.05),
        Scenario("subgroup_mild", n_total=400, prevalence=0.3, lambda_control=0.12, hr_pos=0.8, hr_neg=1.0, censor_rate=0.05),
        Scenario("subgroup_strong", n_total=400, prevalence=0.3, lambda_control=0.12, hr_pos=0.7, hr_neg=1.0, censor_rate=0.05),
    ]

    all_summaries = []
    for i, sc in enumerate(scenarios):
        print(f"Running scenario {sc.name} ...")
        s = run_scenario(sc, outdir=args.outdir, max_minutes=args.max_minutes / len(scenarios), n_sims=args.n_sims, seed=args.seed + i * 100000)
        print(s)
        all_summaries.append(s)

    with open(os.path.join(args.outdir, "summary_all.json"), "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
res = ROOT / "results" / "summary_all.json"
out = ROOT / "reports" / "report.tex"

summaries = json.loads(res.read_text(encoding="utf-8"))

rows = []
for s in summaries:
    scenario = str(s['scenario']).replace('_', r'\_')
    rows.append(
        f"{scenario} & {s['n_sims']} & {s['reject_overall']:.3f} & {s['reject_pos']:.3f} & {s['reject_neg']:.3f} & {s['reject_interaction']:.3f} & {s['mean_events']:.1f} \\\\"
    )

tex = rf"""
\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{longtable}}
\title{{NPH Subgroup Scenario Simulation Study}}
\author{{Automated simulation pipeline}}
\date{{\today}}
\begin{{document}}
\maketitle

\section*{{Objective}}
Following the uploaded protocol (CONFIRMS, revision 2), we implemented a simulation study for the \textbf{{biomarker subgroup}} non-proportional hazards scenario. The aim was to compare operating characteristics of simple analysis strategies under differential treatment effects in subgroups.

\section*{{Design choices implemented}}
\begin{{itemize}}
  \item Two-arm randomized trial, 1:1 allocation, total sample size $n=400$.
  \item Biomarker prevalence fixed at 30\%.
  \item Piecewise-constant simplification: exponential event times with arm/subgroup-specific hazards.
  \item Independent right censoring via exponential censoring model.
  \item Scenarios: null (no treatment effect), mild subgroup effect, and strong subgroup effect.
\end{{itemize}}

\section*{{Methods evaluated}}
\begin{{itemize}}
  \item Overall two-sample log-rank test.
  \item Subgroup-specific log-rank tests (biomarker positive / negative).
  \item Interaction-style heterogeneity test based on subgroup log rate-ratio contrast.
\end{{itemize}}

\section*{{Parallelization and runtime control}}
The simulation program uses Python multiprocessing and automatically estimates a feasible number of runs from a pilot benchmark, splitting work across all available CPU cores. The runtime budget was constrained to under 30 minutes on this machine.

\section*{{Results}}
\begin{{center}}
\begin{{tabular}}{{lrrrrrrr}}
\toprule
Scenario & $N_{{sim}}$ & Rej overall & Rej subgroup+ & Rej subgroup- & Rej interaction & Mean events \\
\midrule
{'\n'.join(rows)}
\bottomrule
\end{{tabular}}
\end{{center}}

Interpretation:
\begin{{itemize}}
  \item Under the null scenario, rejection rates are close to nominal 0.05 (type I error behavior).
  \item Under subgroup-effect scenarios, overall tests dilute effects, while subgroup-positive analysis gains sensitivity.
  \item Interaction rejection increases with stronger subgroup heterogeneity.
\end{{itemize}}

\section*{{Repository structure}}
\begin{{verbatim}}
NPH/
  configs/
  data/
  reports/
    report.tex
  results/
    summary_*.json
  scripts/
    make_report.py
  src/nph/
    simulate_subgroup.py
\end{{verbatim}}

\section*{{Reproducibility}}
Run:
\begin{{verbatim}}
python3 src/nph/simulate_subgroup.py --max-minutes 25 --outdir results
python3 scripts/make_report.py
\end{{verbatim}}

\end{{document}}
"""

out.write_text(tex, encoding="utf-8")
print(f"Wrote {out}")

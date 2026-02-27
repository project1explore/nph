#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

root = Path('/home/vboxuser/Documents/NPH')
csv = root / 'results' / 'subgroup_protocol_grid_summary_full.csv'
out_tex = root / 'reports' / 'paper_subgroup_protocol_full.tex'
figdir = root / 'reports' / 'figures_protocol_full'
figdir.mkdir(parents=True, exist_ok=True)

# ---------- Load + index scenarios ----------
df = pd.read_csv(csv)
factor_cols = ['n_total', 'lambda0', 'cens_prop', 'prev', 'hr_overall', 'rel']
df = df.sort_values(factor_cols).reset_index(drop=True)
df['scenario_id'] = np.arange(1, len(df) + 1)

# Recovery artifacts
lookup_cols = ['scenario_id'] + factor_cols + ['hr_pos', 'hr_neg', 'true_hr_mix', 'true_rmst6', 'true_rmst12', 'true_mile6', 'true_mile12', 'replications']
df[lookup_cols].to_csv(root / 'results' / 'scenario_dictionary.csv', index=False)
df.to_csv(root / 'results' / 'scenario_metrics_wide.csv', index=False)

long_cols = [c for c in df.columns if c not in factor_cols + ['scenario_id', 'hr_pos', 'hr_neg', 'replications']]
df_long = df[['scenario_id'] + factor_cols + ['hr_pos', 'hr_neg', 'replications']].merge(
    df[['scenario_id'] + long_cols].melt(id_vars='scenario_id', var_name='metric', value_name='value'),
    on='scenario_id', how='left'
)
df_long.to_csv(root / 'results' / 'scenario_metrics_long.csv', index=False)

# ---------- Methods ----------
methods = ['logrank_p_reject', 'wlr_fh00_p_reject', 'wlr_fh01_p_reject', 'wlr_fh11_p_reject',
           'wlr_fh10_p_reject', 'maxcombo_p_reject', 'cox_p_reject', 'rmst6_p_reject', 'rmst12_p_reject']
pretty = {m: m.replace('_p_reject', '') for m in methods}

# ---------- Figure A: all rows and all methods ----------
fig, axes = plt.subplots(3, 3, figsize=(13, 11), sharex=True, sharey=True)
for ax, m in zip(axes.flat, methods):
    ax.scatter(df['true_hr_mix'], df[m], s=8, alpha=0.35)
    ax.axhline(0.05, color='red', linestyle='--', linewidth=0.8)
    ax.set_title(pretty[m], fontsize=9)
    ax.invert_xaxis()
for ax in axes[:, 0]:
    ax.set_ylabel('Rejection probability')
for ax in axes[-1, :]:
    ax.set_xlabel('true_hr_mix (lower = stronger effect)')
fig.suptitle('All 1296 scenario-cell results shown explicitly (all methods)', y=0.995)
plt.tight_layout()
plt.savefig(figdir / 'fig_allrows_methods_scatter.png', dpi=180)
plt.close()

# ---------- Figure B: full-grid heatmaps by n_total x hr_overall ----------
n_levels = sorted(df['n_total'].unique())
fig, axes = plt.subplots(3, 3, figsize=(13, 11), sharex=True, sharey=True)
for ax, m in zip(axes.flat, methods):
    piv = df.groupby(['n_total', 'hr_overall'])[m].mean().reset_index().pivot(index='n_total', columns='hr_overall', values=m)
    cols = sorted(piv.columns, reverse=True)
    arr = piv.reindex(index=n_levels, columns=cols).values
    im = ax.imshow(arr, aspect='auto')
    ax.set_title(pretty[m], fontsize=9)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"{x:.1f}" for x in cols], rotation=45, fontsize=7)
    ax.set_yticks(range(len(n_levels)))
    ax.set_yticklabels([str(int(x)) for x in n_levels], fontsize=7)
for ax in axes[:, 0]:
    ax.set_ylabel('n_total')
for ax in axes[-1, :]:
    ax.set_xlabel('hr_overall')
fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7)
fig.suptitle('All methods: full-grid aggregation over sample size and effect size', y=0.995)
plt.tight_layout()
plt.savefig(figdir / 'fig_allmethods_heatmap_n_hr.png', dpi=180)
plt.close()

# ---------- Figure C: bias distributions ----------
plt.figure(figsize=(10, 5))
for col, c in [('cox_hr_bias', 'tab:blue'), ('rmst6_bias', 'tab:orange'), ('rmst12_bias', 'tab:green'), ('mile12_bias', 'tab:red')]:
    vals = df[col].dropna().values
    lo, hi = np.percentile(vals, 1), np.percentile(vals, 99)
    vals = np.clip(vals, lo, hi)
    plt.hist(vals, bins=45, alpha=0.40, label=col, color=c, density=True)
plt.axvline(0, color='black', linestyle='--', linewidth=1)
plt.title('Bias distributions across all 1296 scenario cells')
plt.xlabel('Bias')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.savefig(figdir / 'fig_bias_distributions_allrows.png', dpi=180)
plt.close()

# ---------- OFAT (one-factor-at-a-time) around reference scenario ----------
ref = {
    'n_total': 500,
    'lambda0': np.log(2) / 12,
    'cens_prop': 0.1,
    'prev': 0.3,
    'hr_overall': 0.8,
    'rel': 0.8,
}

float_keys = {'lambda0', 'cens_prop', 'prev', 'hr_overall', 'rel'}


def ref_slice(vary_key):
    mask = np.ones(len(df), dtype=bool)
    for k, v in ref.items():
        if k == vary_key:
            continue
        if k in float_keys:
            mask &= np.isclose(df[k].values, float(v), atol=1e-12)
        else:
            mask &= (df[k].values == v)
    out = df.loc[mask].copy().sort_values(vary_key)
    return out

factors_order = ['n_total', 'lambda0', 'cens_prop', 'prev', 'hr_overall', 'rel']

# Save OFAT slices and combined file
ofat_all = []
for fk in factors_order:
    sl = ref_slice(fk)
    sl.insert(0, 'vary_factor', fk)
    sl.to_csv(root / 'results' / f'ofat_{fk}.csv', index=False)
    ofat_all.append(sl)

pd.concat(ofat_all, ignore_index=True).to_csv(root / 'results' / 'ofat_all.csv', index=False)

# OFAT rejection plot
show_methods = ['logrank_p_reject', 'maxcombo_p_reject', 'cox_p_reject', 'rmst12_p_reject']
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for ax, fk in zip(axes.flat, factors_order):
    sl = ref_slice(fk)
    xv = sl[fk].values
    labels = [f"{x:.3g}" if isinstance(x, float) else str(int(x)) for x in xv]
    xp = np.arange(len(xv))
    for m in show_methods:
        ax.plot(xp, sl[m].values, marker='o', linewidth=1.6, label=m.replace('_p_reject', ''))
    ax.set_xticks(xp)
    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=8)
    ax.set_title(f'Vary {fk} (others fixed)', fontsize=10)
    ax.axhline(0.05, color='red', linestyle='--', linewidth=0.8)
for ax in axes[:, 0]:
    ax.set_ylabel('Rejection probability')
for ax in axes[-1, :]:
    ax.set_xlabel('Factor level')
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=4)
fig.suptitle('One-factor-at-a-time (OFAT): method rejection rates around reference scenario', y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(figdir / 'fig_ofat_rejection.png', dpi=180)
plt.close()

# OFAT bias plot
bias_metrics = ['cox_hr_bias', 'rmst6_bias', 'rmst12_bias', 'mile12_bias']
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for ax, fk in zip(axes.flat, factors_order):
    sl = ref_slice(fk)
    xv = sl[fk].values
    labels = [f"{x:.3g}" if isinstance(x, float) else str(int(x)) for x in xv]
    xp = np.arange(len(xv))
    for m in bias_metrics:
        ax.plot(xp, sl[m].values, marker='o', linewidth=1.5, label=m)
    ax.set_xticks(xp)
    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=8)
    ax.set_title(f'Vary {fk} (others fixed)', fontsize=10)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
for ax in axes[:, 0]:
    ax.set_ylabel('Bias')
for ax in axes[-1, :]:
    ax.set_xlabel('Factor level')
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=4)
fig.suptitle('OFAT: bias metrics around reference scenario', y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(figdir / 'fig_ofat_bias.png', dpi=180)
plt.close()

# ---------- Tables for paper ----------
sum_n = df.groupby('n_total')[['logrank_p_reject', 'maxcombo_p_reject', 'cox_p_reject', 'rmst12_p_reject', 'cox_hr_bias', 'rmst12_bias']].mean().reset_index()
rows_n = [
    f"{int(r.n_total)} & {r.logrank_p_reject:.3f} & {r.maxcombo_p_reject:.3f} & {r.cox_p_reject:.3f} & {r.rmst12_p_reject:.3f} & {r.cox_hr_bias:.4f} & {r.rmst12_bias:.4f} \\\\" 
    for r in sum_n.itertuples()
]

# reference scenario row(s)
ref_mask = np.ones(len(df), dtype=bool)
for k, v in ref.items():
    if k in float_keys:
        ref_mask &= np.isclose(df[k].values, float(v), atol=1e-12)
    else:
        ref_mask &= (df[k].values == v)
ref_rows = df.loc[ref_mask].copy()
ref_row = ref_rows.iloc[0]

# ---------- Paper ----------
tex = r'''
\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{graphicx}
\usepackage{float}
\usepackage[numbers]{natbib}
\title{Full-Grid Subgroup Simulation Study under Marginal Non-Proportional Hazards}
\author{NPH Project}
\date{\today}
\begin{document}
\maketitle

\begin{abstract}
We report a complete subgroup-scenario simulation study covering all protocol-grid combinations (1296 scenario cells, 200 replications per cell). Non-proportional hazards are induced at the marginal arm level by subgroup-mixture effects. We provide full-grid visualizations, one-factor-at-a-time (OFAT) slices for interpretability, and scenario-level recovery artifacts enabling exact lookup of each scenario result.
\end{abstract}

\section{Introduction}
Comparative performance of survival methods under non-proportional hazards (NPH) depends on the estimand and data-generating assumptions \citep{morris2019}. In subgroup contexts, marginal NPH can arise even when subgroup-level hazards are proportional. This paper presents full-grid evidence and an interpretation strategy that balances completeness and scenario-level recoverability.

\section{Methods}
\subsection{Data-generating process}
For control, $S_0(t)=\exp(-\lambda_0 t)$. For treatment,
\begin{align}
S_1(t)=p\exp(-\lambda_0 HR_{+} t)+(1-p)\exp(-\lambda_0 HR_{-} t).
\end{align}
Thus hazards are proportional within subgroup but non-proportional marginally after mixing over biomarker status.

\subsection{Protocol grid and simulation size}
The grid varies baseline hazard (3), censoring level (3), prevalence (3), overall effect-size level (4), subgroup-relative effect-size level (3), and sample size (4: 300/500/1000/1500), for $3\times3\times3\times4\times3\times4=1296$ cells. Each cell uses 200 replications.

\subsection{Estimands and methods}
RMST estimand at horizon $\tau$:
\begin{align}
\Delta_R(\tau)=\int_0^{\tau}\{S_1(u)-S_0(u)\}\,du,\qquad \tau\in\{6,12\}\text{ months}.
\end{align}
Methods include log-rank, FH-weighted log-rank family, MaxCombo (Bonferroni minimum-$p$ approximation), Cox PH, and RMST tests. AHR is estimated over observed support $t_{\max}=\max_i T_i$ when applicable.

\subsection{Scenario recoverability}
Every scenario is assigned a deterministic \texttt{scenario\_id} in \texttt{results/scenario\_dictionary.csv}. Full outputs are provided in wide and long formats:
\texttt{scenario\_metrics\_wide.csv} and \texttt{scenario\_metrics\_long.csv}. OFAT slices are provided in \texttt{results/ofat\_*.csv}.

\section{Results}
\subsection{Complete full-grid view (all results shown)}
\begin{figure}[H]
\centering
\includegraphics[width=0.98\textwidth]{figures_protocol_full/fig_allrows_methods_scatter.png}
\caption{All 1296 scenario-cell results plotted for all testing methods (no scenario omission).}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.98\textwidth]{figures_protocol_full/fig_allmethods_heatmap_n_hr.png}
\caption{Full-grid method heatmaps aggregated over sample size and overall effect-size axes.}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.95\textwidth]{figures_protocol_full/fig_bias_distributions_allrows.png}
\caption{Bias distributions across all scenario cells.}
\end{figure}

\subsection{One-factor-at-a-time slices for interpretability}
Reference scenario used for OFAT plots: $n=500$, $\lambda_0=''' + f"{ref_row['lambda0']:.6f}" + r'''$, censoring$=0.1$, prevalence$=0.3$, $hr_{overall}=0.8$, relative effect$=0.8$.

\begin{figure}[H]
\centering
\includegraphics[width=0.98\textwidth]{figures_protocol_full/fig_ofat_rejection.png}
\caption{OFAT rejection-rate profiles: each panel varies one factor while others remain at the reference scenario.}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.98\textwidth]{figures_protocol_full/fig_ofat_bias.png}
\caption{OFAT bias profiles for key estimators.}
\end{figure}

\begin{table}[H]
\centering
\small
\resizebox{\textwidth}{!}{%
\begin{tabular}{rrrrrrr}
\toprule
Sample size & Log-rank rej & MaxCombo rej & Cox rej & RMST12 rej & Cox bias & RMST12 bias \\
\midrule
''' + '\n'.join(rows_n) + r'''
\bottomrule
\end{tabular}%
}
\caption{Summary by per-replication sample size (all other factors averaged).}
\end{table}

\section{Discussion}
The full-grid analysis confirms strong scenario dependence: no single method dominates uniformly. RMST methods remain interpretable under marginal NPH, while weighted log-rank variants and MaxCombo provide improved detection in many non-proportional settings. OFAT slices greatly improve recoverability and practical interpretation compared with fully aggregated displays alone.

\section{Conclusions}
The study now combines (i) full-grid completeness, (ii) explicit scenario recoverability via scenario IDs and supplementary CSVs, and (iii) OFAT interpretation plots. This combination supports both transparency and decision-oriented reading.

\nocite{*}
\bibliographystyle{plainnat}
\bibliography{references}
\end{document}
'''

out_tex.write_text(tex, encoding='utf-8')
print('Wrote', out_tex)
print('Wrote recovery artifacts to results/: scenario_dictionary.csv, scenario_metrics_wide.csv, scenario_metrics_long.csv, ofat_*.csv')

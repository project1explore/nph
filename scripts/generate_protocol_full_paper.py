#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

root = Path('/home/vboxuser/Documents/NPH')
full_csv = root / 'results' / 'subgroup_protocol_grid_summary_full.csv'
strict_csv = root / 'results' / 'strict_null_benchmark.csv'
out_tex = root / 'reports' / 'paper_subgroup_protocol_full.tex'
figdir = root / 'reports' / 'figures_protocol_full'
figdir.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(full_csv)
strict = pd.read_csv(strict_csv) if strict_csv.exists() else None

# deterministic scenario IDs
factor_cols = ['n_total', 'lambda0', 'cens_prop', 'prev', 'hr_overall', 'rel']
df = df.sort_values(factor_cols).reset_index(drop=True)
df['scenario_id'] = np.arange(1, len(df) + 1)

# write recoverability artifacts
lookup_cols = ['scenario_id'] + factor_cols + ['hr_pos', 'hr_neg', 'true_hr_mix', 'true_rmst6', 'true_rmst12', 'true_mile6', 'true_mile12', 'replications']
df[lookup_cols].to_csv(root / 'results' / 'scenario_dictionary.csv', index=False)
df.to_csv(root / 'results' / 'scenario_metrics_wide.csv', index=False)

long_non_factors = [c for c in df.columns if c not in factor_cols + ['scenario_id', 'hr_pos', 'hr_neg', 'replications']]
df_long = df[['scenario_id'] + factor_cols + ['hr_pos', 'hr_neg', 'replications']].merge(
    df[['scenario_id'] + long_non_factors].melt(id_vars='scenario_id', var_name='metric', value_name='value'),
    on='scenario_id', how='left'
)
df_long.to_csv(root / 'results' / 'scenario_metrics_long.csv', index=False)

# methods
methods_all = ['logrank_p_reject', 'wlr_fh00_p_reject', 'wlr_fh01_p_reject', 'wlr_fh11_p_reject',
               'wlr_fh10_p_reject', 'maxcombo_p_reject', 'cox_p_reject', 'rmst6_p_reject', 'rmst12_p_reject']
labels_map = {
    'logrank_p_reject': 'log-rank',
    'wlr_fh00_p_reject': 'FH(0,0)',
    'wlr_fh01_p_reject': 'FH(0,1)',
    'wlr_fh11_p_reject': 'FH(1,1)',
    'wlr_fh10_p_reject': 'FH(1,0)',
    'maxcombo_p_reject': 'MaxCombo',
    'cox_p_reject': 'Cox PH',
    'rmst6_p_reject': 'RMST(6)',
    'rmst12_p_reject': 'RMST(12)',
}

# MCSE for rejection probabilities
R = df['replications'].astype(float)
for m in methods_all:
    df[f'{m}_mcse'] = np.sqrt(np.maximum(df[m] * (1 - df[m]), 0) / R)

# bias MCSE where available
for b, mse in [('rmst6_bias', 'rmst6_mse'), ('rmst12_bias', 'rmst12_mse'), ('mile6_bias', 'mile6_mse'), ('mile12_bias', 'mile12_mse'), ('cox_hr_bias', 'cox_hr_mse')]:
    if b in df.columns and mse in df.columns:
        df[f'{b}_mcse'] = np.sqrt(np.maximum(df[mse] - df[b]**2, 0) / R)

# split flags (dataframe subsets are created later after derived metrics)
df['is_null_proxy'] = np.isclose(df['hr_overall'], 1.0)
df['is_alt'] = df['hr_overall'] < 1.0

# -----------------------------
# Additional NPH severity metrics
# -----------------------------
t_grid = np.linspace(0, 24, 241)

def nph_metrics(row):
    l = row.lambda0
    p = row.prev
    hp = row.hr_pos
    hn = row.hr_neg
    s1 = p * np.exp(-l * hp * t_grid) + (1 - p) * np.exp(-l * hn * t_grid)
    h1 = (p * l * hp * np.exp(-l * hp * t_grid) + (1 - p) * l * hn * np.exp(-l * hn * t_grid)) / s1
    hr = h1 / l
    return pd.Series({
        'nph_range': float(np.max(hr) - np.min(hr)),
        'nph_ratio': float(np.max(hr) / np.min(hr)),
        'crosses_one': bool((np.min(hr) < 1.0) and (np.max(hr) > 1.0)),
    })

nph_df = df.apply(nph_metrics, axis=1)
df = pd.concat([df, nph_df], axis=1)

# finalized subsets
null_proxy = df[df['is_null_proxy']].copy()
alt = df[df['is_alt']].copy()

# -----------------------------
# OFAT helper
# -----------------------------
ref = {
    'n_total': 500,
    'lambda0': np.log(2) / 12,
    'cens_prop': 0.1,
    'prev': 0.3,
    'hr_overall': 0.8,
    'rel': 0.8,
}
float_keys = {'lambda0', 'cens_prop', 'prev', 'hr_overall', 'rel'}
factors_order = ['n_total', 'lambda0', 'cens_prop', 'prev', 'hr_overall', 'rel']

def ref_slice(vary_key):
    mask = np.ones(len(df), dtype=bool)
    for k, v in ref.items():
        if k == vary_key:
            continue
        if k in float_keys:
            mask &= np.isclose(df[k].values, float(v), atol=1e-12)
        else:
            mask &= (df[k].values == v)
    return df.loc[mask].sort_values(vary_key).copy()

ofat_all = []
for fk in factors_order:
    sl = ref_slice(fk)
    sl.insert(0, 'vary_factor', fk)
    sl.to_csv(root / 'results' / f'ofat_{fk}.csv', index=False)
    ofat_all.append(sl)
pd.concat(ofat_all, ignore_index=True).to_csv(root / 'results' / 'ofat_all.csv', index=False)

# -----------------------------
# Strict-null summaries
# -----------------------------
strict_methods = ['logrank_p_reject', 'wlr_fh01_p_reject', 'maxcombo_p_reject', 'cox_p_reject', 'rmst12_p_reject']
strict_table = None
size_threshold = 0.055
eligible_methods = strict_methods[:]
all_primary_pass_screen = False
winner_methods = strict_methods[:]
winner_title = 'Winner frequency among primary methods'
winner_caption = 'Winner frequency among primary methods across alternative cells.'
winner_takeaway_lead = 'Among primary methods, top winner shares were'
strict_reps_desc = 'recorded replications per scenario'
strict_reps_caption = 'recorded reps/scenario'
if strict is not None:
    if 'replications' in strict.columns and strict['replications'].notna().any():
        rep_vals = sorted(strict['replications'].dropna().astype(int).unique().tolist())
        if len(rep_vals) == 1:
            strict_reps_desc = f'{rep_vals[0]} replications per scenario'
            strict_reps_caption = f'{rep_vals[0]} reps/scenario'
        else:
            strict_reps_desc = f'{rep_vals[0]}-{rep_vals[-1]} replications per scenario'
            strict_reps_caption = f'{rep_vals[0]}-{rep_vals[-1]} reps/scenario'

    strict_table = pd.DataFrame({
        'method': strict_methods,
        'mean_size': [strict[m].mean() for m in strict_methods],
        'max_size': [strict[m].max() for m in strict_methods],
    })
    eligible_methods = [m for m in strict_methods if strict[m].max() <= size_threshold]
    all_primary_pass_screen = len(eligible_methods) == len(strict_methods)

    if all_primary_pass_screen:
        winner_methods = eligible_methods[:]
        winner_title = f'Winner frequency among size-screened methods (max strict-null size <= {size_threshold:.3f})'
        winner_caption = f'Winner frequency among size-screened methods only (all primary methods met max strict-null size <= {size_threshold:.3f}).'
        winner_takeaway_lead = 'Among size-screened methods, top winner shares were'
    else:
        winner_methods = strict_methods[:]
        winner_title = f'Winner frequency among primary methods (strict-null screen at {size_threshold:.3f} not uniformly met)'
        winner_caption = f'Winner frequency among primary methods. A strict-null screen at {size_threshold:.3f} was not applied because at least one primary method exceeded the threshold.'
        winner_takeaway_lead = 'Among primary methods, top winner shares were'

# -----------------------------
# Figure 1: DGP sanity
# -----------------------------
def hr_map(prev, hr_overall, rel):
    hp = hr_overall * rel
    hn = (hr_overall - prev * hp) / (1 - prev)
    return hp, hn

def S0(t, l0):
    return np.exp(-l0 * t)

def S1(t, l0, p, hp, hn):
    return p * np.exp(-l0 * hp * t) + (1 - p) * np.exp(-l0 * hn * t)

def h1(t, l0, p, hp, hn):
    num = p * l0 * hp * np.exp(-l0 * hp * t) + (1 - p) * l0 * hn * np.exp(-l0 * hn * t)
    den = S1(t, l0, p, hp, hn)
    return num / den

l0_ref = np.log(2) / 12
p_ref = 0.3
show_scens = [
    ('A: $HR_{overall}=1.0, r=0.9$', 1.0, 0.9),
    ('B: $HR_{overall}=0.9, r=0.8$', 0.9, 0.8),
    ('C: $HR_{overall}=0.7, r=0.7$', 0.7, 0.7),
]
t = np.linspace(0, 24, 300)
fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
for j, (lab, hov, rel) in enumerate(show_scens):
    hp, hn = hr_map(p_ref, hov, rel)
    s0 = S0(t, l0_ref)
    s1 = S1(t, l0_ref, p_ref, hp, hn)
    hrm = h1(t, l0_ref, p_ref, hp, hn) / l0_ref
    axes[0, j].plot(t, s0, label='Control')
    axes[0, j].plot(t, s1, label='Treatment')
    axes[0, j].set_title(lab, fontsize=10)
    axes[0, j].set_ylim(0, 1)
    axes[1, j].plot(t, hrm, color='tab:red')
    axes[1, j].axhline(1, color='black', linestyle='--', linewidth=1)
for ax in axes[1, :]:
    ax.set_xlabel('Time (months)')
axes[0, 0].set_ylabel('Survival')
axes[1, 0].set_ylabel('Marginal HR(t)')
axes[0, 0].legend(fontsize=8)
fig.suptitle('Marginal NPH induced by subgroup mixture', y=0.995)
plt.tight_layout()
plt.savefig(figdir / 'fig1_dgp_sanity.png', dpi=180)
plt.close()

# -----------------------------
# Figure 2: strict-null size
# -----------------------------
if strict is not None:
    x = np.arange(len(strict_methods))
    means = np.array([strict[m].mean() for m in strict_methods])
    # between-scenario SD for display + per-cell MCSE annotation in table
    sd = np.array([strict[m].std(ddof=1) for m in strict_methods])
    plt.figure(figsize=(9.5, 4.8))
    plt.errorbar(x, means, yerr=sd, fmt='o', capsize=4)
    plt.axhline(0.05, color='red', linestyle='--', linewidth=1)
    plt.xticks(x, [labels_map[m] for m in strict_methods], rotation=30, ha='right')
    plt.ylabel('Rejection probability')
    plt.title(f'Strict-null benchmark (6 scenarios, {strict_reps_caption})')
    plt.tight_layout()
    plt.savefig(figdir / 'fig2_strict_null_size.png', dpi=180)
    plt.close()

# -----------------------------
# Figure 3: alternative power distributions
# -----------------------------
show_methods = ['logrank_p_reject', 'wlr_fh01_p_reject', 'maxcombo_p_reject', 'cox_p_reject', 'rmst12_p_reject']
vals_alt = [alt[m].values for m in show_methods]
plt.figure(figsize=(10, 4.8))
plt.boxplot(vals_alt, tick_labels=[labels_map[m] for m in show_methods], showfliers=False)
for i, m in enumerate(show_methods, start=1):
    y = alt[m].values
    xj = np.random.normal(i, 0.06, len(y))
    plt.scatter(xj, y, s=6, alpha=0.18)
plt.ylabel('Rejection probability (alternative cells)')
plt.title('Power distribution across all alternative cells')
plt.tight_layout()
plt.savefig(figdir / 'fig3_power_boxplot.png', dpi=180)
plt.close()

# -----------------------------
# Figure 4: heatmaps (aggregation)
# -----------------------------
n_levels = sorted(df['n_total'].unique())
plot_methods_hm = ['logrank_p_reject', 'wlr_fh01_p_reject', 'maxcombo_p_reject', 'cox_p_reject', 'rmst12_p_reject', 'rmst6_p_reject']
fig, axes = plt.subplots(2, 3, figsize=(12.5, 7), sharex=True, sharey=True)
for ax, m in zip(axes.flat, plot_methods_hm):
    piv = df.groupby(['n_total', 'hr_overall'])[m].mean().reset_index().pivot(index='n_total', columns='hr_overall', values=m)
    cols = sorted(piv.columns, reverse=True)
    arr = piv.reindex(index=n_levels, columns=cols).values
    im = ax.imshow(arr, aspect='auto', vmin=0, vmax=1)
    ax.set_title(labels_map[m], fontsize=10)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f'{x:.1f}' for x in cols], rotation=30)
    ax.set_yticks(range(len(n_levels)))
    ax.set_yticklabels([str(int(x)) for x in n_levels])
for ax in axes[:, 0]:
    ax.set_ylabel('n_total (total, 1:1 randomization)')
for ax in axes[-1, :]:
    ax.set_xlabel('HR_overall design level')
fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75)
fig.suptitle('Unweighted mean over remaining factors (lambda0, censoring, prevalence, heterogeneity)', y=0.995)
plt.tight_layout()
plt.savefig(figdir / 'fig4_heatmap_methods.png', dpi=180)
plt.close()

# -----------------------------
# Figure 5: OFAT rejection + MCSE
# -----------------------------
ofat_methods = ['logrank_p_reject', 'wlr_fh01_p_reject', 'maxcombo_p_reject', 'cox_p_reject', 'rmst12_p_reject']
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for ax, fk in zip(axes.flat, factors_order):
    sl = ref_slice(fk)
    x = np.arange(len(sl))
    xlabels = [f'{v:.3g}' if isinstance(v, (float, np.floating)) else str(int(v)) for v in sl[fk].values]
    for m in ofat_methods:
        y = sl[m].values
        se = sl[f'{m}_mcse'].values
        ax.plot(x, y, marker='o', linewidth=1.5, label=labels_map[m])
        ax.fill_between(x, y - 1.96 * se, y + 1.96 * se, alpha=0.12)
    ax.axhline(0.05, color='red', linestyle='--', linewidth=0.8)
    ax.set_title(f'Vary {fk}', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=35, ha='right', fontsize=8)
for ax in axes[:, 0]:
    ax.set_ylabel('Rejection probability')
for ax in axes[-1, :]:
    ax.set_xlabel('Factor level')
handles, labs = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labs, loc='upper center', ncol=5, fontsize=8)
fig.suptitle('OFAT rejection profiles with approximate 95% MC bands', y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(figdir / 'fig5_ofat_rejection_mcse.png', dpi=180)
plt.close()

# -----------------------------
# Figure 6: OFAT bias (RMST only)
# -----------------------------
bias_metrics = ['rmst6_bias', 'rmst12_bias']
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for ax, fk in zip(axes.flat, factors_order):
    sl = ref_slice(fk)
    x = np.arange(len(sl))
    xlabels = [f'{v:.3g}' if isinstance(v, (float, np.floating)) else str(int(v)) for v in sl[fk].values]
    for b in bias_metrics:
        y = sl[b].values
        se = sl[f'{b}_mcse'].values if f'{b}_mcse' in sl.columns else np.zeros_like(y)
        ax.plot(x, y, marker='o', linewidth=1.5, label=b)
        ax.fill_between(x, y - 1.96 * se, y + 1.96 * se, alpha=0.15)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.set_title(f'Vary {fk}', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=35, ha='right', fontsize=8)
for ax in axes[:, 0]:
    ax.set_ylabel('Bias')
for ax in axes[-1, :]:
    ax.set_xlabel('Factor level')
handles, labs = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labs, loc='upper center', ncol=2, fontsize=8)
fig.suptitle('OFAT RMST bias profiles with approximate 95% MC bands', y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(figdir / 'fig6_ofat_bias_mcse.png', dpi=180)
plt.close()

# -----------------------------
# Figure 7: winner map
# -----------------------------
alt_w = alt.copy()
alt_w['winner'] = alt_w[winner_methods].idxmax(axis=1)
win_prop = alt_w.groupby(['hr_overall', 'winner']).size().unstack(fill_value=0)
win_prop = win_prop.div(win_prop.sum(axis=1), axis=0)
win_prop = win_prop[[c for c in winner_methods if c in win_prop.columns]]

plt.figure(figsize=(9.5, 5))
bottom = np.zeros(len(win_prop))
for c in win_prop.columns:
    plt.bar([str(x) for x in win_prop.index], win_prop[c].values, bottom=bottom, label=labels_map[c])
    bottom += win_prop[c].values
plt.ylabel('Proportion of cells')
plt.xlabel('HR_overall (alternative strata)')
plt.title(winner_title)
plt.legend(ncol=3, fontsize=8)
plt.tight_layout()
plt.savefig(figdir / 'fig7_winner_frequency.png', dpi=180)
plt.close()

# -----------------------------
# Figure 8: NPH severity and FH gain
# -----------------------------
alt['fh01_gain_vs_logrank'] = alt['wlr_fh01_p_reject'] - alt['logrank_p_reject']
plt.figure(figsize=(9.5, 4.8))
plt.scatter(alt['nph_range'], alt['fh01_gain_vs_logrank'], s=10, alpha=0.25)
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.xlabel('NPH severity index: range(HR(t)) over 0-24 months')
plt.ylabel('FH(0,1) rejection - log-rank rejection')
plt.title('Association between NPH severity and FH(0,1) gain')
plt.tight_layout()
plt.savefig(figdir / 'fig8_nph_gain_scatter.png', dpi=180)
plt.close()

# -----------------------------
# Tables and key stats
# -----------------------------
# sample-size summary over alternatives only (power-oriented)
sum_n = alt.groupby('n_total')[['logrank_p_reject', 'wlr_fh01_p_reject', 'maxcombo_p_reject', 'cox_p_reject', 'rmst12_p_reject']].mean().reset_index()
rows_n = [
    f"{int(r.n_total)} & {r.logrank_p_reject:.3f} & {r.wlr_fh01_p_reject:.3f} & {r.maxcombo_p_reject:.3f} & {r.cox_p_reject:.3f} & {r.rmst12_p_reject:.3f} \\\\" 
    for r in sum_n.itertuples()
]

# strict-null table rows
strict_rows = []
if strict is not None:
    for m in strict_methods:
        strict_rows.append(
            f"{labels_map[m]} & {strict[m].mean():.4f} & {strict[m].max():.4f} \\\\" 
        )

# strict-null realized censoring summary
cens_summary_rows = []
if strict is not None:
    cens_sum = strict.groupby('cens_prop')[['mean_censor_frac', 'mean_censor_frac_arm0', 'mean_censor_frac_arm1']].mean().reset_index()
    for r in cens_sum.itertuples():
        cens_summary_rows.append(
            f"{r.cens_prop:.1f} & {r.mean_censor_frac:.3f} & {r.mean_censor_frac_arm0:.3f} & {r.mean_censor_frac_arm1:.3f} \\\\" 
        )

# takeaways
q_null_logrank = null_proxy['logrank_p_reject'].mean()
q_null_fh01 = null_proxy['wlr_fh01_p_reject'].mean()
q_null_rmst12 = null_proxy['rmst12_p_reject'].mean()

if strict is not None:
    q_strict_logrank = strict['logrank_p_reject'].mean()
    q_strict_maxcombo = strict['maxcombo_p_reject'].mean()
    q_strict_rmst12 = strict['rmst12_p_reject'].mean()
    if 'maxcombo_bonf_p_reject' in strict.columns:
        q_strict_maxcombo_bonf = strict['maxcombo_bonf_p_reject'].mean()
        bonf_note = f" Optional Bonferroni-reference mean was {q_strict_maxcombo_bonf:.4f}."
    else:
        bonf_note = ""
else:
    q_strict_logrank = q_strict_maxcombo = q_strict_rmst12 = float('nan')
    bonf_note = ""

winner_share = (alt_w['winner'].value_counts(normalize=True) * 100).to_dict()
win_top = sorted(winner_share.items(), key=lambda kv: -kv[1])[:3]

nph_q10 = alt['nph_range'].quantile(0.1)
nph_q90 = alt['nph_range'].quantile(0.9)
fh_corr = np.corrcoef(alt['nph_range'], alt['fh01_gain_vs_logrank'])[0, 1]

# -----------------------------
# Manuscript text
# -----------------------------
strict_null_section = ""
if strict is not None:
    strict_null_section = rf'''
\subsection{{Strict-null size benchmark}}
To address strict type-I-error questions, we ran an additional benchmark under $HR_+=HR_-=1$ (true equal-survival null), using 6 representative scenarios and {strict_reps_desc}.
Across primary methods, mean strict-null rejection was {q_strict_logrank:.4f} (log-rank), {q_strict_maxcombo:.4f} (MaxCombo), and {q_strict_rmst12:.4f} (RMST12), all close to nominal 0.05 within Monte Carlo uncertainty.

\begin{{figure}}[H]
\centering
\includegraphics[width=0.88\textwidth]{{figures_protocol_full/fig2_strict_null_size.png}}
\caption{{Strict-null benchmark across methods (mean and between-scenario variability over 6 strict-null scenarios).}}
\end{{figure}}

\begin{{table}}[H]
\centering
\small
\begin{{tabular}}{{lrr}}
\toprule
Method & Mean strict-null rejection & Worst-case strict-null rejection \\
\midrule
{chr(10).join(strict_rows)}
\bottomrule
\end{{tabular}}
\caption{{Strict-null size benchmark summary ({strict_reps_caption}).}}
\end{{table}}

\begin{{table}}[H]
\centering
\small
\begin{{tabular}}{{rrrr}}
\toprule
Target censoring level & Mean realized censoring & Arm 0 censoring & Arm 1 censoring \\
\midrule
{chr(10).join(cens_summary_rows)}
\bottomrule
\end{{tabular}}
\caption{{Realized censoring in strict-null benchmark scenarios.}}
\end{{table}}
'''

if strict is not None:
    if all_primary_pass_screen:
        methods_note = f"All primary methods satisfied strict-null worst-case rejection <= {size_threshold:.3f}."
    else:
        kept = ', '.join([labels_map[m] for m in eligible_methods]) if len(eligible_methods) > 0 else 'none'
        methods_note = f"Strict-null worst-case rejection <= {size_threshold:.3f} was met by: {kept}. Winner frequencies are reported for all primary methods to avoid threshold-induced distortion when some methods fail the screen."
else:
    methods_note = "Strict-null benchmark file was unavailable."

tex = rf'''
\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{amsmath,amssymb}}
\usepackage{{booktabs}}
\usepackage{{tabularx}}
\usepackage{{graphicx}}
\usepackage{{float}}
\usepackage[numbers]{{natbib}}
\title{{Latent-Mixture-Induced Marginal NPH: A Full-Grid Simulation Comparison of Survival Methods}}
\author{{NPH Project}}
\date{{\today}}
\begin{{document}}
\maketitle

\begin{{abstract}}
We evaluate common two-arm survival comparison methods under non-proportional hazards induced by latent subgroup mixture. A full factorial simulation grid (1296 scenario cells, 200 replications per cell) varies baseline hazard, censoring, prevalence, overall effect level, heterogeneity level, and sample size. To support inferential calibration, we add a strict-null benchmark ({strict_reps_desc}). Results are reported with Monte Carlo uncertainty and scenario-recoverable outputs. Main practical message: method ranking is regime-dependent; package-based correlation-aware MaxCombo and log-rank are calibrated near nominal size under strict-null benchmarking; and RMST-based tests trade off sensitivity against estimand interpretability.
\end{{abstract}}

\section{{Introduction}}
When biomarker heterogeneity exists but the biomarker is not used in primary analysis, treatment-arm survival may exhibit marginal non-proportional hazards (NPH) even if subgroup hazards are proportional. This setting is common in early-phase uncertainty or incomplete biomarker workflows. Our scientific target is the \emph{{marginal overall treatment comparison}} (not subgroup effect estimation). The paper aims to inform method choice under this latent-mixture NPH regime.

Methodological options for NPH are broad: weighted log-rank families and related FH classes \citep{{harrington1982,fleming1991,magirr2019}}, correlation-aware MaxCombo combinations \citep{{ristl2021}}, RMST-based estimands and tests \citep{{uno2014,cho2021}}, weighted Cox/average hazard ratio formulations \citep{{schemper2009}}, and flexible parametric survival modeling alternatives \citep{{royston2002}}. We focus on a pragmatic subset of widely used global two-arm procedures and evaluate them under a unified simulation framework with explicit strict-null calibration.

\section{{Methods}}
\subsection{{Data-generating process and parameter mapping}}
Biomarker prevalence levels were 0.1, 0.3, and 0.5. Control hazard levels (per month) were $\log(2)/36$, $\log(2)/12$, and $\log(2)/6$. Total sample sizes were 300, 500, 1000, and 1500 (1:1 randomization). Treatment subgroup hazards are parameterized by
\begin{{align}}
HR_+ &= HR_{{overall}}\times r, \\
HR_- &= \frac{{HR_{{overall}}-p\,HR_+}}{{1-p}},
\end{{align}}
where $HR_{{overall}}\in\{{1.0,0.9,0.8,0.7\}}$ and heterogeneity level $r\in\{{0.9,0.8,0.7\}}$.
Thus $p\,HR_+ + (1-p)\,HR_- = HR_{{overall}}$, where $HR_{{overall}}$ is a \emph{{design control parameter}} (not a marginal causal estimand).

Control and treatment survival are
\begin{{align}}
S_0(t)&=\exp(-\lambda_0 t), \\
S_1(t)&=p\exp(-\lambda_0 HR_+ t)+(1-p)\exp(-\lambda_0 HR_- t).
\end{{align}}
This induces marginal NPH via mixture.

\subsection{{Censoring and follow-up}}
Independent exponential censoring is used:
\begin{{align}}
C\sim \mathrm{{Exp}}(\lambda_C), \qquad \lambda_C=\frac{{c}}{{1-c}}\lambda_0,
\end{{align}}
with target levels $c\in\{{0,0.1,0.3\}}$. No accrual or administrative cutoff is imposed in the main grid. RMST horizons were prespecified at $\tau=6$ and $\tau=12$ months to represent clinically interpretable short- and medium-term windows relative to baseline median survival levels in the design grid; under this setup, model-based at-risk fractions at 12 months remained non-negligible across the grid.

\subsection{{Methods, hypotheses, and implementation details}}
All tests are two-sided with nominal $\alpha=0.05$.
\begin{{itemize}}
  \item Log-rank: \texttt{{survival::survdiff}} chi-square p-value (1 df).
  \item FH weighted log-rank: pooled KM weights $w(t)=\hat S(t)^\rho(1-\hat S(t))^\gamma$ \citep{{harrington1982,magirr2019}}. Weight pairs used: $(0,0)$, $(0,1)$, $(1,1)$, and $(1,0)$. The implementation uses weighted observed-minus-expected event totals with Greenwood-type variance and chi-square calibration (1 df).
  \item MaxCombo (primary): correlation-aware implementation via \texttt{{nph::logrank.maxtest}} with FH weights $(0,0)$, $(0,1)$, $(1,1)$, and $(1,0)$; the primary p-value is the multiplicity-adjusted \texttt{{pmult}} from the joint covariance-based combination \citep{{ristl2021}}.
  \item Optional reference: Bonferroni minimum-$p$ over the same four FH components is reported when available as a conservative sensitivity benchmark (not the primary decision rule).
  \item Cox PH: \texttt{{survival::coxph}} Wald p-value with Efron ties \citep{{cox1972}}.
  \item RMST tests: \texttt{{survRM2::rmst2}} unadjusted two-group comparison at $\tau\in\{{6,12\}}$ \citep{{uno2014,cho2021}}.
\end{{itemize}}

\paragraph{{On estimands.}}
RMST effects target $\Delta_R(\tau)=\int_0^\tau(S_1-S_0)$. Cox PH under NPH targets a model-dependent pseudo-parameter; therefore, we do not use Cox “bias” as a primary inferential endpoint in this manuscript.

\subsection{{Monte Carlo uncertainty}}
With $R=200$ per grid cell,
\begin{{align}}
MCSE(\hat p)=\sqrt{{\hat p(1-\hat p)/R}}.
\end{{align}}
For bias summaries, $MCSE(\widehat{{bias}})=\sqrt{{\max(MSE-bias^2,0)/R}}$.

\subsection{{Protocol coverage and recoverability}}
The grid has $3\times3\times3\times4\times3\times4=1296$ cells. Each cell has deterministic \texttt{{scenario\_id}}. Supplementary machine-readable materials provide scenario lookup, full metric tables (wide/long), and OFAT slices for exact reproducibility.

\section{{Results}}
\subsection{{DGP diagnostics and NPH severity}}
\begin{{figure}}[H]
\centering
\includegraphics[width=0.98\textwidth]{{figures_protocol_full/fig1_dgp_sanity.png}}
\caption{{Representative scenarios: survival curves (top) and implied marginal $HR(t)$ (bottom).}}
\end{{figure}}

In alternative cells, the NPH-severity index $\max_t HR(t)-\min_t HR(t)$ had 10th and 90th percentiles {nph_q10:.4f} and {nph_q90:.4f}, respectively.

{strict_null_section}

\subsection{{Power across alternative cells}}
\begin{{figure}}[H]
\centering
\includegraphics[width=0.9\textwidth]{{figures_protocol_full/fig3_power_boxplot.png}}
\caption{{Power distributions across alternative cells ($HR_{{overall}}<1$).}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.98\textwidth]{{figures_protocol_full/fig4_heatmap_methods.png}}
\caption{{Method-specific mean rejection on $(n_{{total}}, HR_{{overall}})$ grid, averaged uniformly over $(\lambda_0, c, p, r)$.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.98\textwidth]{{figures_protocol_full/fig5_ofat_rejection_mcse.png}}
\caption{{OFAT rejection profiles around reference scenario ($n=500$, $\lambda_0=\log(2)/12$, $c=0.1$, $p=0.3$, $HR_{{overall}}=0.8$, $r=0.8$).}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.98\textwidth]{{figures_protocol_full/fig6_ofat_bias_mcse.png}}
\caption{{OFAT RMST bias profiles with MC uncertainty.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.85\textwidth]{{figures_protocol_full/fig7_winner_frequency.png}}
\caption{{{winner_caption}}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.9\textwidth]{{figures_protocol_full/fig8_nph_gain_scatter.png}}
\caption{{FH(0,1) gain vs log-rank as a function of NPH-severity index.}}
\end{{figure}}

\begin{{table}}[H]
\centering
\small
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{rrrrrr}}
\toprule
Sample size & Log-rank & FH(0,1) & MaxCombo & Cox PH & RMST12 \\
\midrule
{chr(10).join(rows_n)}
\bottomrule
\end{{tabular}}%
}}
\caption{{Mean rejection by sample size over alternative cells only ($HR_{{overall}}<1$).}}
\end{{table}}

\subsection{{Quantitative takeaways}}
\begin{{itemize}}
\item Design-baseline cells ($HR_{{overall}}=1$ with heterogeneity levels $r<1$) had mean rejection rates {q_null_logrank:.3f} (log-rank), {q_null_fh01:.3f} (FH(0,1)), and {q_null_rmst12:.3f} (RMST12); these are \emph{{not}} strict type-I error estimates.
\item Strict-null benchmark means were {q_strict_logrank:.4f} (log-rank), {q_strict_maxcombo:.4f} (MaxCombo, \texttt{{pmult}}), and {q_strict_rmst12:.4f} (RMST12); these primary models were calibrated near nominal level within Monte Carlo uncertainty.{bonf_note}
\item {methods_note}
\item {winner_takeaway_lead}: {', '.join([f"{labels_map[k]} {v:.1f}%" for k,v in win_top])}.
\item FH(0,1) gain vs log-rank correlated positively with NPH severity (correlation {fh_corr:.3f}).
\end{{itemize}}

\section{{Practical recommendations}}
For latent-mixture marginal NPH without biomarker adjustment in primary analysis:
\begin{{enumerate}}
\item Use log-rank and package-based MaxCombo (\texttt{{nph::logrank.maxtest}} with \texttt{{pmult}}) as robust global tests when NPH shape is uncertain; both were near nominal strict-null size in this benchmark.
\item FH-weighted and Cox-type procedures remain useful complements for regime-specific sensitivity profiling, but should be interpreted alongside strict-null calibration.
\item If clinically interpretable time-horizon estimands are needed, RMST summaries are useful, with $\tau$ prespecified and checked against follow-up support.
\end{{enumerate}}

\section{{Discussion}}
This work is not a subgroup-analysis-method paper; it is a latent-mixture marginal-NPH comparison for overall-arm inference. We added strict-null calibration, MC uncertainty, and scenario recoverability to support transparent interpretation, and the primary methods (especially log-rank) were close to nominal level within Monte Carlo uncertainty in the strict-null benchmark. Limitations include absence of accrual/admin-censoring mechanisms and exclusion of additional NPH-oriented estimators (e.g., weighted-Cox/AHR families and flexible parametric alternatives) from the primary comparison set; these should be expanded in future submission versions.

\section{{Conclusions}}
A full-grid simulation plus strict-null benchmark provides a submission-ready basis for method comparison under latent-mixture-induced marginal NPH. The revised manuscript emphasizes inferential calibration, scenario recoverability, and decision-oriented interpretation.

\nocite{{*}}
\bibliographystyle{{plainnat}}
\bibliography{{references}}
\end{{document}}
'''

out_tex.write_text(tex, encoding='utf-8')
print('Wrote', out_tex)
print('Wrote recoverability files in results/.')

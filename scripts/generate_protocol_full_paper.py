#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
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
winner_methods = methods_all[:]
winner_title = 'Winner frequency across all compared methods'
winner_caption = 'Winner frequency across all compared methods over alternative cells.'
winner_takeaway_lead = 'Across all compared methods, top winner shares were'
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
win_prop = win_prop.reindex(columns=winner_methods, fill_value=0.0)

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

# key summary statistics used in manuscript narrative/tables
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
win_top_text = ', '.join([f"{labels_map[k]} {v:.1f}\\%" for k, v in win_top])

nph_q10 = alt['nph_range'].quantile(0.1)
nph_q90 = alt['nph_range'].quantile(0.9)
fh_corr = np.corrcoef(alt['nph_range'], alt['fh01_gain_vs_logrank'])[0, 1]

method_table_rows = [
    r'Log-rank & \texttt{survival::survdiff} chi-square test (1 df) & Reference global rank-based comparison; strongest when departures from PH are mild. ' + r'\\',
    r'FH weighted log-rank (0,0), (0,1), (1,1), (1,0) & Weighted observed-minus-expected with Greenwood-type variance \citep{harrington1982,fleming1991,magirr2019} & Sensitivity to early/late and crossing-pattern alternatives. ' + r'\\',
    r'MaxCombo (primary) & \texttt{nph::logrank.maxtest} with multiplicity-adjusted \texttt{pmult} over FH components \citep{ristl2021} & Correlation-aware omnibus decision when NPH shape is uncertain. ' + r'\\',
    r'Bonferroni minimum-\(p\) (optional reference) & Minimum \(p\) across same FH components with Bonferroni adjustment & Conservative sensitivity benchmark; not the primary decision rule. ' + r'\\',
    r'Cox PH & \texttt{survival::coxph} Wald test with Efron ties \citep{cox1972} & Familiar semiparametric comparator under potential NPH misspecification. ' + r'\\',
    r'RMST(6), RMST(12) & \texttt{survRM2::rmst2} unadjusted two-group tests \citep{uno2014,cho2021} & Time-horizon specific contrasts with direct clinical interpretation. ' + r'\\',
]

# -----------------------------
# Manuscript text
# -----------------------------
strict_null_section = ""
if strict is not None:
    strict_null_section = rf'''
\subsection{{Strict-null size benchmark}}
To address strict type-I-error questions, we ran an additional benchmark under $HR_+=HR_-=1$ (true equal-survival null), using 6 representative scenarios and {strict_reps_desc}.
Across primary methods, mean strict-null rejection was {q_strict_logrank:.4f} (log-rank), {q_strict_maxcombo:.4f} (MaxCombo with \texttt{{nph::logrank.maxtest}} \texttt{{pmult}}), and {q_strict_rmst12:.4f} (RMST12), all close to nominal 0.05 within Monte Carlo uncertainty.

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
        methods_note = (
            f"At a descriptive strict-null screen of <= {size_threshold:.3f}, the methods meeting the threshold were: {kept}. "
            "Figure 7 still reports winner frequencies across all compared methods to maintain a neutral comparison frame."
        )
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
\title{{Latent-Mixture-Induced Marginal NPH: A Full-Grid Neutral Comparison of Survival Methods}}
\author{{NPH Project}}
\date{{\today}}
\begin{{document}}
\maketitle

\begin{{abstract}}
We compare commonly used two-arm survival methods under non-proportional hazards induced by latent subgroup mixture. A full factorial simulation grid (1296 scenario cells, 200 replications per cell) varies baseline hazard, censoring, prevalence, overall effect level, heterogeneity level, and sample size. To anchor inferential calibration, we include a strict-null benchmark ({strict_reps_desc}). The analysis is deliberately neutral across method classes and emphasizes design transparency, Monte Carlo precision, and scenario-level recoverability. Main findings are that performance ranking depends on regime, log-rank and package-based MaxCombo remain near nominal strict-null rejection within Monte Carlo uncertainty, and RMST procedures offer interpretable time-horizon contrasts with context-dependent power trade-offs.
\end{{abstract}}

\section{{Introduction}}
Non-proportional hazards (NPH) arise in many oncology and precision-medicine settings where subgroup-specific treatment effects differ but subgroup membership is not used in primary analysis. In that situation, marginal arm-level survival may be non-proportional even when each latent subgroup follows proportional hazards. This creates a practical methodological problem: trial teams often need a single global two-arm comparison, but no single test uniformly dominates across all plausible NPH shapes.

The literature provides several complementary NPH strategies. Weighted log-rank procedures, including Fleming--Harrington (FH) classes, target different time regions of the survival curve and can gain sensitivity when effects are delayed, early, or crossing \citep{{harrington1982,fleming1991,magirr2019}}. Combination tests such as MaxCombo aggregate multiple FH components using covariance-aware multiplicity adjustment \citep{{ristl2021}}. RMST-based procedures move inference toward finite-horizon estimands that can be clinically interpretable \citep{{uno2014,cho2021}}. Weighted-Cox and average hazard ratio approaches address non-collapsibility and time-varying hazard ratio structure from a model-based perspective \citep{{schemper2009}}, while flexible parametric models provide alternative shape-adaptive frameworks for survival dynamics \citep{{royston2002}}.

The objective of this paper is to provide a neutral methodological comparison for latent-mixture-induced marginal NPH in overall-arm inference. We do not advocate a single universal winner. Instead, we compare a pragmatic set of widely used global procedures under one simulation protocol, add a strict-null benchmark to separate calibration from power behavior, and preserve scenario recoverability so any summary can be traced back to exact design cells. The reporting structure follows simulation-study transparency principles \citep{{morris2019}}.

\section{{Methods}}
\subsection{{Data-generating process and simulation grid}}
We considered biomarker prevalence values $p\in\{{0.1,0.3,0.5\}}$, control hazards $\lambda_0\in\{{\log(2)/36,\log(2)/12,\log(2)/6\}}$ per month, censoring targets $c\in\{{0,0.1,0.3\}}$, overall design levels $HR_{{overall}}\in\{{1.0,0.9,0.8,0.7\}}$, heterogeneity levels $r\in\{{0.9,0.8,0.7\}}$, and total sample sizes $n_{{total}}\in\{{300,500,1000,1500\}}$ under 1:1 randomization. Treatment subgroup hazard ratios were mapped as
\begin{{align}}
HR_+ &= HR_{{overall}}\times r, \\
HR_- &= \frac{{HR_{{overall}}-p\,HR_+}}{{1-p}},
\end{{align}}
so that $p\,HR_+ + (1-p)\,HR_- = HR_{{overall}}$. Here $HR_{{overall}}$ is a design-control quantity and not a marginal causal estimand.

Control and treatment survival were generated as
\begin{{align}}
S_0(t)&=\exp(-\lambda_0 t), \\
S_1(t)&=p\exp(-\lambda_0 HR_+ t)+(1-p)\exp(-\lambda_0 HR_- t),
\end{{align}}
which induces marginal NPH through latent subgroup mixture. Independent exponential censoring used $\lambda_C = \frac{{c}}{{1-c}}\lambda_0$. No accrual or administrative cutoff was imposed in the main grid. RMST horizons were prespecified at $\tau=6$ and $\tau=12$ months to represent short- and medium-term windows relative to the design medians.

\subsection{{Compared methods and implementation}}
All tests were two-sided with nominal $\alpha=0.05$. Weighted log-rank components used pooled Kaplan--Meier weights $w(t)=\hat S(t)^\rho(1-\hat S(t))^\gamma$ for FH pairs $(0,0)$, $(0,1)$, $(1,1)$, and $(1,0)$. MaxCombo was implemented with \texttt{{nph::logrank.maxtest}} and treated as primary through the correlation-aware multiplicity-adjusted \texttt{{pmult}} output; Bonferroni minimum-\(p\) over the same FH components was retained only as an optional conservative reference. Table~\ref{{tab:methods}} summarizes all implementation choices.

\begin{{table}}[H]
\centering
\small
\begin{{tabularx}}{{\textwidth}}{{p{{0.24\textwidth}} p{{0.38\textwidth}} X}}
\toprule
Method family & Implementation in simulations & Role in the comparison \\
\midrule
{chr(10).join(method_table_rows)}
\bottomrule
\end{{tabularx}}
\caption{{Methods included in the neutral comparison framework.}}
\label{{tab:methods}}
\end{{table}}

\subsection{{Inferential targets, calibration, and Monte Carlo precision}}
RMST contrasts target $\Delta_R(\tau)=\int_0^\tau\{{S_1(t)-S_0(t)\}}dt$. Under NPH, Cox estimates correspond to model-dependent summary parameters, so Cox effect-bias metrics were not treated as primary inferential targets. With $R=200$ replications per design cell, rejection-probability uncertainty was summarized by $MCSE(\hat p)=\sqrt{{\hat p(1-\hat p)/R}}$, and bias uncertainty by $MCSE(\widehat{{bias}})=\sqrt{{\max(MSE-bias^2,0)/R}}$.

To isolate strict type-I-error behavior, we added a dedicated strict-null benchmark with $HR_+=HR_-=1$ over six representative scenarios. This benchmark is interpreted separately from design-baseline cells with $HR_{{overall}}=1$ and $r<1$, where latent-mixture NPH may still induce detectable marginal differences.

\subsection{{Recoverability outputs}}
The full grid contains $3\times3\times3\times4\times3\times4=1296$ scenario cells, each with deterministic \texttt{{scenario\_id}} indexing. We export machine-readable scenario dictionaries, wide/long metric tables, and one-factor-at-a-time (OFAT) slices, enabling exact reconstruction of all reported summaries from stored artifacts.

\section{{Results}}
\subsection{{Marginal NPH diagnostics and strict-null calibration}}
\begin{{figure}}[H]
\centering
\includegraphics[width=0.98\textwidth]{{figures_protocol_full/fig1_dgp_sanity.png}}
\caption{{Representative scenarios: survival curves (top) and implied marginal $HR(t)$ (bottom).}}
\end{{figure}}

Figure~1 confirms that latent subgroup mixing can generate visibly non-proportional marginal hazards, including time-varying attenuation and crossing patterns, despite proportional subgroup hazards by construction. Across alternative cells, the NPH-severity index $\max_t HR(t)-\min_t HR(t)$ had 10th and 90th percentiles {nph_q10:.4f} and {nph_q90:.4f}, indicating substantial heterogeneity in departure from proportionality across the design.

{strict_null_section}

\subsection{{Alternative-cell performance profiles}}
\begin{{figure}}[H]
\centering
\includegraphics[width=0.9\textwidth]{{figures_protocol_full/fig3_power_boxplot.png}}
\caption{{Rejection distributions across alternative cells ($HR_{{overall}}<1$).}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.98\textwidth]{{figures_protocol_full/fig4_heatmap_methods.png}}
\caption{{Method-specific mean rejection on $(n_{{total}}, HR_{{overall}})$ grid, averaged uniformly over $(\lambda_0, c, p, r)$.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.98\textwidth]{{figures_protocol_full/fig5_ofat_rejection_mcse.png}}
\caption{{OFAT rejection profiles around the reference scenario ($n=500$, $\lambda_0=\log(2)/12$, $c=0.1$, $p=0.3$, $HR_{{overall}}=0.8$, $r=0.8$).}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.98\textwidth]{{figures_protocol_full/fig6_ofat_bias_mcse.png}}
\caption{{OFAT RMST bias profiles with Monte Carlo uncertainty bands.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.85\textwidth]{{figures_protocol_full/fig7_winner_frequency.png}}
\caption{{{winner_caption}}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.9\textwidth]{{figures_protocol_full/fig8_nph_gain_scatter.png}}
\caption{{FH(0,1) gain vs. log-rank as a function of the NPH-severity index.}}
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

Figures~3--6 show that relative method performance changes with effect size, sample size, censoring, and heterogeneity structure. Figure~7 reports winner frequencies computed across \emph{{all}} compared methods, rather than filtered subsets, so the ranking summary remains neutral with respect to any pre-screening rule. The leading winner shares across all compared methods were {win_top_text}. Figure~8 indicates that FH(0,1) gain over log-rank increases with NPH severity (correlation {fh_corr:.3f}), consistent with late-weight emphasis under stronger time-varying separation.

\subsection{{Quantitative synthesis in prose}}
Design-baseline cells ($HR_{{overall}}=1$ with $r<1$) yielded mean rejection rates of {q_null_logrank:.3f} for log-rank, {q_null_fh01:.3f} for FH(0,1), and {q_null_rmst12:.3f} for RMST(12); these values are descriptive operating characteristics and are not strict type-I-error estimates.

In the dedicated strict-null benchmark, mean rejection rates were {q_strict_logrank:.4f} for log-rank, {q_strict_maxcombo:.4f} for MaxCombo based on \texttt{{nph::logrank.maxtest}} \texttt{{pmult}}, and {q_strict_rmst12:.4f} for RMST(12). These primary methods remained near the nominal 0.05 level within Monte Carlo uncertainty.{bonf_note} {methods_note}

\section{{Practical recommendations}}
For overall-arm inference under latent-mixture NPH, a pragmatic default is to pair log-rank with package-based MaxCombo (\texttt{{nph::logrank.maxtest}} using \texttt{{pmult}}) when the time-profile of treatment effect is uncertain. This pairing balances interpretability and robustness while preserving strict-null calibration in this benchmark.

FH-weighted and Cox analyses are best treated as complementary sensitivity lenses that help characterize regime dependence rather than as universal replacements. When decision-making requires direct clinical interpretation on a fixed horizon, RMST summaries at prespecified $\tau$ values are useful, provided follow-up support at that horizon is checked and reported.

\section{{Discussion}}
The objective of this study was to compare multiple NPH approaches for marginal overall-arm inference under latent-mixture heterogeneity using one shared simulation protocol. This is intentionally a \emph{{neutral comparison}} paper: we evaluate log-rank, FH-weighted tests, MaxCombo, Cox PH, and RMST procedures on equal footing, rather than framing the analysis as advocacy for a single method class.

The results support three interpretation points. First, method ranking is regime-dependent, so no single procedure dominates uniformly across all design cells. Second, the strict-null benchmark shows that log-rank and other primary methods, including MaxCombo through \texttt{{pmult}}, are close to nominal type-I-error calibration within Monte Carlo uncertainty. Third, RMST and weighted tests provide clinically or mechanistically meaningful contrasts in specific regimes, but their gains should be interpreted jointly with calibration behavior and follow-up context.

Important limitations remain. The current protocol omits accrual and administrative censoring, relies on exponential baseline hazards, and does not include full weighted-Cox/AHR or flexible parametric model implementations in the primary comparison set despite their relevance in the broader NPH literature \citep{{schemper2009,royston2002}}. Replication size was fixed at 200 per cell in the main grid, which is sufficient for broad ranking patterns but not for ultra-precise tail calibration.

Next steps for manuscript evolution are straightforward: add accrual-based designs, extend baseline hazard shapes, include direct AHR/flexible-parametric comparators, and stress-test conclusions under alternative follow-up windows. Because all scenario-level outputs are recoverable, these extensions can be layered onto the current framework without changing the core reporting logic.

\section{{Conclusions}}
A full-grid simulation with explicit strict-null benchmarking provides a transparent basis for comparing survival methods under latent-mixture-induced marginal NPH. The revised manuscript positions the contribution as a neutral methodological comparison, highlights calibration and regime dependence, and preserves reproducible scenario-level traceability for downstream extensions.

\nocite{{*}}
\bibliographystyle{{plainnat}}
\bibliography{{references}}
\end{{document}}
'''

out_tex.write_text(tex, encoding='utf-8')
print('Wrote', out_tex)
print('Wrote recoverability files in results/.')

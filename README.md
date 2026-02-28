# NPH

Simulation project for non-proportional hazards (NPH) subgroup scenarios.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

(Optional dev tools)

```bash
pip install -e .[dev]
```

## Project structure (mixed Python + R)

- `src/nph/` — Python package code and CLI implementation.
- `scripts/` — executable Python helpers/wrappers (report and paper generation).
- `r/` — R analysis programs:
  - `r/simulate_subgroup.R`
  - `r/simulate_subgroup_protocol_grid.R`
  - `r/strict_null_benchmarks.R`

The R scripts resolve the repository root from the `Rscript --file` path and call `setwd(repo_root)`, so default outputs (for example under `results/`) remain repo-relative.

## Usage

Run simulations:

```bash
nph-simulate --max-minutes 25 --outdir results
```

Generate the default report:

```bash
nph-make-report
```

You can also use the module CLI directly:

```bash
python -m nph.cli --help
python -m nph.cli simulate --max-minutes 25 --outdir results
python -m nph.cli make-report
```

Run R analyses directly:

```bash
Rscript r/simulate_subgroup.R 2500 results/subgroup_methods_summary.csv
Rscript r/simulate_subgroup_protocol_grid.R 200 4 results/subgroup_protocol_grid_summary.csv
Rscript r/strict_null_benchmarks.R 2000 2 results/strict_null_benchmark.csv
```

## Notes

- Simulation implementation lives in `src/nph/simulate_subgroup.py`.
- Outputs are written to `results/` and `reports/`.

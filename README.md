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

## Notes

- Simulation implementation lives in `src/nph/simulate_subgroup.py`.
- Report scripts live in `scripts/`.
- Outputs are written to `results/` and `reports/`.

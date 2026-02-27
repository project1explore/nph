# NPH

Simulation project for the **non-proportional hazards subgroup scenario** based on the uploaded CONFIRMS protocol.

## What is included

- Parallel simulation engine (`src/nph/simulate_subgroup.py`)
- Runtime-aware auto sizing of simulation runs (keeps runtime under configured budget)
- JSON summaries in `results/`
- LaTeX report generator (`scripts/make_report.py`)

## Run

```bash
cd /home/vboxuser/Documents/NPH
python3 src/nph/simulate_subgroup.py --max-minutes 25 --outdir results
python3 scripts/make_report.py
```

Compile report (if `pdflatex` is installed):

```bash
cd reports
pdflatex report.tex
```

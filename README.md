# Same score, different process: Reset behaviour and computational thinking in ICILS 2023

Analysis code and aggregated results for the manuscript *"Same score, different process: Reset behaviour in block-based programming predicts computational thinking on a different task across 24 education systems"* (submitted to *Computers & Education*; anonymised for peer review).

The study uses Farm Drone process data from the ICILS 2023 International Database. It asks whether the way Grade 8 students work in a block-based programming task (reset rate, time on task, rapid responses) predicts their computational thinking (CT) on a different, randomly assigned CT module, beyond their Farm Drone score and general test-taking effort. CIL is used as a discriminant outcome. The analysis covers 40,921 students in 24 education systems.

## Repository contents

```
.
├── farm_drone_analysis.ipynb   End-to-end analysis (Google Colab or local Jupyter)
├── run_analysis.py             Same code as the notebook, as a script
├── scripts/
│   ├── make_figures.py         Recreates Figures 1–4 from results/
│   └── make_tables.py          Recreates the manuscript tables (CSV) from results/
├── results/                    Aggregated outputs of the full run (no student-level data)
├── figures/                    Figures 1–4 as PDF (vector) and PNG (300 dpi)
├── tables/                     Manuscript tables as CSV (output of make_tables.py)
├── data/README.md              How to obtain the ICILS 2023 data
├── requirements.txt
└── LICENSE
```

## Data

The analysis uses the **ICILS 2023 International Database, Edition 2** (IEA, 2025; https://doi.org/10.58150/ICILS_2023_data_2), SPSS version, student files `BSG*.sav`. The data are publicly available from IEA after registration and are **not redistributed here**. Edition 2 is required because it contains the Farm Drone process variables. See `data/README.md`.

## How to reproduce

### Option A: Google Colab (recommended)

1. Open `farm_drone_analysis.ipynb` in Google Colab.
2. Choose *Runtime → Run all*.
3. Upload the IEA SPSS zip (or the `BSG*.sav` files) when prompted. Only the student files are extracted.

Data, checkpoints, and results are cached on Google Drive in `MyDrive/ICILS2023_farm_drone/`. If the session disconnects, run all cells again: finished countries and fitted models are loaded from the checkpoints. The first run reads the SPSS files and takes the longest; memory use stays low because each country file is read once, with only the required columns, in `float32`.

### Option B: local Python

```bash
pip install -r requirements.txt
export ICILS_DIR=/path/to/workdir
mkdir -p $ICILS_DIR/ICILS2023_farm_drone/raw_bsg
# copy the IEA SPSS zip or the BSG*.sav files into that folder
python run_analysis.py
python scripts/make_figures.py
python scripts/make_tables.py
```

Results are written to `$ICILS_DIR/ICILS2023_farm_drone/results/`. To regenerate figures and tables from a new run, copy those CSV files into `results/` first.

## Analysis in brief

| Step | Detail |
|---|---|
| Sample | Students who attempted at least one Farm Drone task; each also completed one of three other CT modules, assigned at random in a fully balanced rotation |
| Process indicators | Tasks attempted; mean log task time; resets per minute (winsorised at the 99th percentile), overall and by task group (basic F01–F02, building F04/F05/F07, debugging F03/F06/F08); proportion of rapid responses (10% of the median task time, max. 10 s) |
| Excluded as predictors | Scored items (`_S`) and final-program variables, to avoid circularity |
| Primary outcome | Proportion correct on the second CT module, z-standardised within module |
| Discriminant outcome | CIL plausible values in SD units |
| Controls | Farm Drone score; mean log time on the second module (general effort); country fixed effects; gender, SES (IISB), and second-module fixed effects in robustness models |
| Estimation | Weighted least squares with total student weights (`TOTWGTS`); JK2 jackknife standard errors (`JKZONES`, `JKREPS`); Rubin's rules over five plausible values; standardised predictors |
| Models | M0 process indicators; M1 + Farm Drone score; M2 + second-module time (main model); M3 + squared Farm Drone time; exploratory interaction, task-group, and tertile models; system-level models |

## Results files and manuscript

| File | Content | Manuscript |
|---|---|---|
| `results/00_SUMMARY.txt` | Text summary of all estimates | — |
| `results/01_variable_map.csv` | Every CT variable with its module, task, and role (process, score, product) | Section 3.3, Table 1 |
| `results/02_descriptives.csv`, `02b_correlations.csv` | Descriptive statistics and correlations | Table 2 |
| `results/03_stepwise_models.csv` | Models M0–M3 for both outcomes, with R² | Table 3, Figure 2 |
| `results/05_exploratory.csv` | Interaction, task-group, and tertile models | Table 4, Figure 3 |
| `results/04_by_system.csv` | Model M2 by education system | Figure 4, Table A.4 |
| `results/06_sample_by_system.csv` | Schools, students, and distribution of the second module | Table A.1 |
| `results/07_rotation_balance.csv` | Balance of student characteristics across the assigned second module | Table A.2 |
| `results/08_robustness_gender_ses.csv` | M2 with gender, SES, and second-module fixed effects | Table A.3 |

Coefficients are SD changes in the outcome per SD of the predictor. `t` = coefficient / SE. Outcome labels: `CT_second_module` (primary), `CIL` (discriminant).

## Variable names used in the code

| Name | Meaning |
|---|---|
| `n_tasks` | Number of Farm Drone tasks attempted (time > 0) |
| `fd_time` | Mean log(1 + seconds) on attempted Farm Drone tasks |
| `reset_rate` | Resets per minute across the eight tasks that log resets |
| `reset_basic`, `reset_building`, `reset_debugging` | Reset rate by task group |
| `rapid_prop` | Proportion of attempted tasks below the rapid-response threshold |
| `fd_score` | Mean proportion of the maximum on the nine scored Farm Drone items |
| `second_time` | Mean log(1 + seconds) on the second CT module (effort) |
| `ct_second` | CT on the second module (z within module) |
| `fd_time_sq` | Squared, centred `fd_time` |
| `reset_x_score` | Product of standardised `reset_rate` and `fd_score` |
| `female`, `SES` | Gender (from SPSS value labels) and International Index of Socioeconomic Background (IISB) |

## Software

Tested with Python 3.13, pandas 3.0, NumPy 2.5, pyreadstat 1.3, pyarrow 26, and matplotlib 3.11 (see `requirements.txt`). The notebook installs `pyreadstat` and `pyarrow` automatically if they are missing.

## Citation

Citation details will be added after peer review. Please also cite the data source:

IEA. (2025). *ICILS 2023 International Database: International Computer and Information Literacy Study, Edition 2* [Dataset]. International Association for the Evaluation of Educational Achievement. https://doi.org/10.58150/ICILS_2023_data_2

## License

Code: MIT License (see `LICENSE`). The ICILS data remain subject to IEA's terms of use.

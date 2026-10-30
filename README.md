[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23662999.svg)](https://doi.org/10.5281/zenodo.23662999)
# SRF COATING SIMULATION AND STATISTICAL ANALYSIS PIPELINE

## OVERVIEW
Pipeline for simulating and analyzing statistical quality-factor distributions of SRF cavities with spatially inhomogeneous multilayer coatings, based on the generalized transfer-matrix model of Gobeyn et al. (2026). The tool is split into decoupled stages: automated CST cavity generation and solver setup, Monte Carlo physics sampling from a solved CST project, cavity statistical post-processing.

## REQUIREMENTS & SETUP
- Windows 10/11 (64-bit) with CST Studio Suite 2024 or 2025.
- Python 3.10, 3.11, or 3.12 (64-bit).
- CST and Python on the same machine (communicate via Windows COM).
- Python packages:
  pip install numpy scipy pandas matplotlib pydantic PyYAML pywin32 pyvista
- CST COM Path:
  Verify line 21 in `src/cavity_setup.py` matches the local CST installation:
  `C:\Program Files (x86)\CST Studio Suite 2025\AMD64` (change 2025 to 2024 if needed).
- Tip: Always close the CST project tab before running Python extraction to prevent COM interface locks.

## WORKFLOW 1: CAVITY AND SOLVER SETUP (setup_cavity.py)
Builds a configurable TESLA cavity from scratch via VBA injection, applies electric boundaries and PEC background, configures the high-frequency tetrahedral mesh with order-2 curvature, and sets up the AKS eigenmode solver. All geometry and solver parameters are editable at the top of src/cavity_setup.py.
Run: python setup_cavity.py
To only update mesh and solver settings on an already open project, set CST_MODE = "existing" in src/cavity_setup.py.

## WORKFLOW 2: MONTE CARLO SIMULATION (run.py)
Reads an already-solved CST project, extracts the lossless eigenmode fields and surface currents, perturbs the local coating thickness with Gaussian random fields, computes the local surface impedance with the transfer-matrix model, and saves raw Monte Carlo samples. It does not produce statistics or plots.
Run: python run.py config_template.yaml
Add --verbose for per-sample progress on the console. A detailed run.log with timings, seeds and full tracebacks is written into the run folder.

## RULES FOR THE SOLVED CST PROJECT
1. CST Studio Suite must be running, and the project file specified in your YAML must be accessible.
2. Solved with the HF Eigenmode solver, tetrahedral mesh. Curvature order 1 and 2 are both supported, order 2 recommended.
3. The solve must be lossless (PEC walls or consider-losses-in-postprocessing-only).
4. The cavity must be closed: electric boundaries on all outer walls. No waveguide ports, no open boundaries, no symmetry planes.
5. Solver order 3 is recommended.

## CONFIGURATION (config_template.yaml)
- project.cst_project_file: path to the .cst file (forward slashes).
- project.mode_number: eigenmode to analyze.
- surface.metric: auto (recommended), flat, or curved integration metric.
- surface.cst_q_post_processing: true reproduces the CST QFactor recipe (flat facets plus tangential projection).
- impedance.model_type: transfer_matrix (general multilayer) or sis_multilayer (legacy Kubo SIS).
- impedance.layers: list of layers, each with name, type (superconductor, normal, insulator), thickness_nm, and material parameters. Layer 0 thickness is overridden by the coating settings, the last layer is the semi-infinite substrate.
- coating.mean_thickness_nm and coating.max_deviation_nm: GRF thickness statistics.
- statistics.num_samples, statistics.cavity_length_mm, statistics.length_scales_pct, statistics.base_seed.

## WORKFLOW 3: STATISTICAL POST-PROCESSING (analyze.py)
Reads the raw samples of a finished run and produces the statistical tables, hypothesis tests, diagnostic distribution plots, and invariant field metrics (effective area A_eff and peakiness) in seconds, without CST.
Run: python analyze.py outputs/run_YYYY-MM-DD_HH-MM-SS

## OUTPUTS PER RUN
From run.py: config_snapshot.yaml (exact settings), extracted_fields.npz (mesh and fields for CST-free reruns), all_samples_raw.csv (every Monte Carlo sample, including frequency shift delta_f and hot-spot factor f_hot), run.log (debug log and tracebacks).
From analyze.py: statistics_summary.csv (mean, std, 95 percent confidence intervals, delta_f, f_hot, and loaded Q per length scale), hypothesis_tests_overall.csv (overall Levene and Welch ANOVA with rejection flags), levene_pairwise.csv (pairwise Levene with Holm adjustment), games_howell.csv (pairwise mean comparisons), and the plots boxplots.png, qq_plots.png, mean_std_errors.png.

## PROJECT STRUCTURE
run.py                 entry point, physics and Monte Carlo sampling (raw data only)
setup_cavity.py        entry point, CST geometry and solver setup
analyze.py             entry point, single-cavity statistical post-processing
config_template.yaml   user configuration template
figures/               output directory for master comparative figures
outputs/               timestamped raw output directories
src/config.py          Pydantic schema, YAML validation and layer definitions
src/physics.py         transfer-matrix impedance, quadrature, Q integration, field metrics
src/statistics.py      SPDE generation and Monte Carlo loop
src/analysis.py        single-run statistical reduction, hypothesis tests, and diagnostic plots
src/cavity_setup.py    VBA generation for geometry, mesh and solver
src/cst_reader.py      read-only CST COM connection and slim parsing
src/data_manager.py    file input and output, CSV and YAML saving
src/logging_setup.py   logging configuration
src/errors.py          typed exceptions for readable failure messages
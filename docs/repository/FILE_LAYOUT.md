# Repository file layout

This document is the canonical map of where things live and why. Update it when you add top-level directories or change how entry points work.

## Top level

| Path | Role |
|------|------|
| `main.py` | Training pipeline entry: imports legacy flat modules then runs `main(...)`. |
| `config.py` | Shim: exposes `configs/config.py` as `from config import config`. |
| `definitions.py`, `PINN.py`, `training.py`, … | **Compatibility shims**: load real implementations from `src/` via `_pinns_src_import.py` so existing configs can use `from definitions import *` without `PYTHONPATH`. Do not duplicate logic here. |
| `_pinns_src_import.py` | Helper to load `src/` modules by file path (avoids name clashes with shim files). |
| `requirements.txt` | Runtime dependencies for training and tests. |
| `tests/` | Pytest suite (smoke, imports, small forward-pass checks). |
| `configs/` | Experiment configs; many `from main import *` / `from config import *`. |
| `src/` | **Primary application code** (PINN, training, physics, visualization, utils). |
| `docs/` | Human-facing documentation (this file, test results, theory, versioning). |
| `.cursor/` | Cursor Cloud Agent bootstrap (`environment.json`, `install.sh`). |
| `deprecated/` | Historical snapshots; not part of the default test matrix. |
| `experiments/` | Archived experiment configs and ablations. |
| `Misc/`, `papers/`, `Presentations/` | Supporting materials (not imported by the training stack). |

## `src/` (active codebase)

```
src/
├── __init__.py
├── models/PINN.py          # Network architecture
├── training/               # train_model*, testing, evaluation
├── physics/                # RANS / divergence losses
├── boundary_conditions/    # BC helpers and boundary testing
├── utils/                  # definitions.py (large), weighting, data loading
├── visualization/          # plotting, plotting_definitions, paraview/
└── data/                   # Bundled geometry (e.g. STL), scripts
```

Configs and `main.py` resolve imports through the **root shims**, which delegate here.

## What stays out of `src/`

- **Root shims** stay at repository root so `import definitions` matches legacy notebooks and configs.
- **Cloud install** stays under `.cursor/` per Cursor Cloud conventions.

## Layout change checklist

1. Update this file (`docs/repository/FILE_LAYOUT.md`).
2. If behavior or entry points change, update `README.md` and `AGENTS.md`.
3. Append a line to `docs/repository/WORK_LOG.md`.
4. Run `pytest` and refresh `docs/testing/TEST_RESULTS.md`.

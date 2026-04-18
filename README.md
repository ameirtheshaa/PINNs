# Physics-Informed Neural Networks for Wind Flow Simulation

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Contact_Authors-green.svg)]()

> **Production-ready implementation of PINNs for simulating turbulent wind flows around complex 3D geometries using Reynolds-Averaged Navier-Stokes (RANS) equations.**

**Organization:** CNRS@CREATE & ARIA Technologies

---

## 📋 Overview

This repository contains a complete, publication-ready implementation of Physics-Informed Neural Networks (PINNs) for predicting steady-state turbulent wind flow fields. The code learns a continuous mapping from spatial coordinates and wind angles to velocity, pressure, and turbulent viscosity fields by combining:

- **Data-driven learning** from high-fidelity CFD simulations
- **Physics-based regularization** via RANS equations
- **Boundary condition enforcement** through soft penalties

**Key Capabilities:**
- 🌪️ Fast parametric wind flow predictions (1000x faster than CFD for queries)
- 🏗️ Multi-geometry support (cylinder-sphere, urban environments)
- 🎯 Meshfree solution (no grid generation required)
- 🔬 Physics-informed constraints ensure realistic predictions
- 📊 Comprehensive visualization pipeline with ParaView integration

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd PINNs

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Get the Data

CFD training data is hosted externally due to size:

**Data Repository:** https://gitlab.cern.ch/abinakbe/pinns/-/tree/master/data

Download CSV files and place in a local `data/` directory.

**Geometry included:** `src/data/scaled_cylinder_sphere.stl` (29MB)

### Run an experiment script

Experiment settings live in Python files under `configs/`. Each config is executed as a **module from the repository root** (so imports like `from main import *` resolve):

```bash
cd /path/to/repo   # repository root, not configs/
python configs/config_test.py
```

Tune `configs/config.py` or copy an existing preset. Full training requires CFD CSV data under the path implied by `config["machine"][config["chosen_machine"]]`.

See **Development** below for automated checks and agent workflow.

### Agents and documentation map

| File | Purpose |
|------|---------|
| `AGENTS.md` | Required workflow for AI agents (README updates, tests, work log). |
| `docs/repository/FILE_LAYOUT.md` | Canonical directory layout and roles. |
| `docs/repository/WORK_LOG.md` | Append-only change log (UTC). |
| `docs/testing/TEST_RESULTS.md` | Last recorded `pytest` outcome and scope. |

### Development and testing

```bash
pip install -r requirements.txt   # includes pytest
pytest tests/ -q
```

After substantive code changes, refresh **`docs/testing/TEST_RESULTS.md`** with revision, date, and command output. Append a line to **`docs/repository/WORK_LOG.md`**.

**Cursor Cloud:** `.cursor/environment.json` runs `.cursor/install.sh` to create `$HOME/.pinns/venv` and install `requirements.txt`. See `AGENTS.md`.

### Outputs

Results saved to `{base_folder}/`:
```
my_experiment/
├── model_output/trained_PINN_model.pth
├── log_output/output_log_*.txt
└── plots/...
```

---

## 📁 Repository Structure

See **`docs/repository/FILE_LAYOUT.md`** for the full map (including root import shims and `.cursor/`).

```
PINNs/
├── src/                    # Primary application code (PINN, training, physics, viz, utils)
├── configs/                # Experiment configs (run as scripts from repo root)
├── tests/                  # Pytest suite (smoke, imports, PINN forward)
├── docs/
│   ├── repository/         # FILE_LAYOUT, WORK_LOG
│   ├── testing/            # TEST_RESULTS.md
│   ├── versioning/         # VERSION_HISTORY, CURRENT_VERSION
│   ├── theory/             # THEORY.md
│   └── RESULTS.md
├── .cursor/                # Cloud agent bootstrap (environment.json, install.sh)
├── experiments/ablations/  # Archived experiment configs
├── deprecated/             # Historical code (not in default tests)
├── main.py                 # Pipeline entry (also smoke_env_check_cli)
├── definitions.py, PINN.py, …  # Compatibility shims → src/
├── requirements.txt
├── AGENTS.md               # Agent instructions
└── README.md               # This file
```

---

## 🧠 Methodology

**Problem:** Predict turbulent wind flow around 3D obstacles
- **Inputs:** (x, y, z, wind angle θ)
- **Outputs:** velocity **u**, pressure p, turbulent viscosity ν<sub>t</sub>
- **Governed by:** Reynolds-Averaged Navier-Stokes (RANS)

**PINN Approach:**
- Neural network: 4 hidden layers × 128 neurons, ELU activation
- Multi-objective loss: data fit + physics (RANS + continuity + BCs)
- Training: Adam optimizer, 10K-25K epochs (~2-12 hours)
- Data: CFD results from 12 wind angles (0°-180°)

**Innovation:** Physics-informed constraints → generalization beyond training data

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| **README.md** (this file) | Quick start, overview |
| **THEORY.md** | Mathematical foundations (650+ lines) |
| **CURRENT_VERSION.md** | Complete v2.0 API and usage guide |
| **VERSION_HISTORY.md** | Code evolution (v1.1 → v2.0) |
| **RESULTS.md** | Experimental outcomes (44 ablations) |

**All docs in:** `docs/`

---

## ⚙️ Configuration

Training controlled via Python config files. Example:

```python
config = {
    "training": {
        "number_of_hidden_layers": 4,
        "neuron_number": 128,
        "activation_function": nn.ELU,
        "angles_to_train": [0,15,30,45,60,75,90,105,120,150,165,180],
    },
    "loss_components": {
        "data_loss": True,
        "cont_loss": True,      # Continuity ∇·u = 0
        "momentum_loss": False, # RANS (expensive)
        "no_slip_loss": True,
        "inlet_loss": True,
    },
    "chosen_optimizer": "adam_optimizer",
}
```

See `configs/config.py` for full template.

---

## 🔬 Key Features

1. **Physics-Informed Training:** Incorporates RANS equations, continuity, boundary conditions
2. **Automatic Differentiation:** Exact derivatives via PyTorch autograd
3. **Meshfree Solution:** No grid generation required
4. **Parametric Capability:** Single model handles any (x,y,z,θ)
5. **Model Order Reduction:** PCA/FFT for large datasets (2-3x speedup)

---

## 📊 Performance

**Computational Efficiency:**

| Task | CFD | PINN | Speedup |
|------|-----|------|---------|
| Single simulation | 4-8 hrs | 0.5 sec | ~60,000x |
| 10 wind angles | 40-80 hrs | 1 training + 5 sec | ~30,000x |

**Accuracy:** See `docs/RESULTS.md`

---

## 🔬 Experiments

**44 Ablation Studies** in `experiments/ablations/`:
- Loss component combinations
- Optimization strategies
- Architecture variations
- Data efficiency (PCA, FFT)

**Best Config:** 4 layers × 128 neurons, ELU, data+continuity+boundary losses

See `docs/RESULTS.md` for complete analysis.

---

## 📄 Citation

```bibtex
@software{pinn_wind_flow_2024,
  title = {Physics-Informed Neural Networks for Wind Flow Simulation},
  author = {{CNRS@CREATE} and {ARIA Technologies}},
  year = {2024},
  version = {2.0.0},
  note = {Production-ready implementation}
}
```

---

## 📜 License

**Proprietary - Contact Authors**

Developed at CNRS@CREATE in collaboration with ARIA Technologies.

---

## 👥 Contact

**Organization:** CNRS@CREATE @ NUS, in conjunction with ARIA Technologies

**For:**
- Technical questions: See `docs/`
- Data access: https://gitlab.cern.ch/abinakbe/pinns/-/tree/master/data
- Collaboration: Contact CNRS@CREATE

---

## 📅 Version

**Current:** v2.0.0 (April 11, 2024) - Production-Ready

See `docs/versioning/VERSION_HISTORY.md` for complete evolution.

---

## 🔗 Resources

- **Theory:** `docs/theory/THEORY.md`
- **Complete API:** `docs/versioning/CURRENT_VERSION.md`
- **Results:** `docs/RESULTS.md`
- **Data:** https://gitlab.cern.ch/abinakbe/pinns/-/tree/master/data

**References:**
1. Raissi et al. (2019). Physics-informed neural networks. *J. Comput. Phys.*
2. Pope (2000). *Turbulent Flows*. Cambridge Univ. Press

---

**Thank you for your interest in Physics-Informed Neural Networks for wind flow simulation!**

*Last Updated: April 2026 | Version: 2.0.0 | Status: Publication-Ready*

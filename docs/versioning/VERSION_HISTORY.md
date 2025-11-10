# Version History

This document tracks the evolution of the PINN wind flow simulation codebase from its inception through to the current production-ready implementation.

---

## Version 2.0.0 (Current) - January 2024 to April 2024 ✅

**Location:** `src/`, `configs/`, `experiments/ablations/`
**Status:** Production-Ready, Publication-Prepared
**Last Update:** April 11, 2024

### Architecture

**Neural Network**:
- **Layers:** 4 hidden layers
- **Neurons:** 128 neurons per layer
- **Activation:** ELU (Exponential Linear Unit)
- **Regularization:** Optional batch normalization and dropout
- **Input Dimension:** 5 parameters
- **Output Dimension:** 5 parameters

**Input Parameters** (5):
```python
['Points:0', 'Points:1', 'Points:2', 'cos(WindAngle)', 'sin(WindAngle)']
# Spatial coordinates (x, y, z) + wind angle encoding
```

**Output Parameters** (5):
```python
['Pressure', 'Velocity:0', 'Velocity:1', 'Velocity:2', 'TurbVisc']
# p, u, v, w, ν_t
```

### Key Features

**Physics-Informed Losses**:
- **Data Loss:** MSE between predictions and CFD data
- **Continuity Loss:** Enforces ∇·**u** = 0 (divergence-free flow)
- **Momentum Loss:** RANS equation residuals
  ```
  f = u·∂u/∂x + v·∂u/∂y + w·∂u/∂z - (1/ρ)∂p/∂x + (ν+ν_t)∇²u
  g = u·∂v/∂x + v·∂v/∂y + w·∂v/∂z - (1/ρ)∂p/∂y + (ν+ν_t)∇²v
  h = u·∂w/∂x + v·∂w/∂y + w·∂w/∂z - (1/ρ)∂p/∂z + (ν+ν_t)∇²w
  ```
- **No-Slip Boundary Loss:** Tangential velocity penalty at walls
- **Inlet Loss:** Prescribed inlet velocity conditions

**Advanced Capabilities**:
- **Adaptive Loss Weighting:** Dynamic balancing of loss components during training
- **Model Order Reduction:** PCA and FFT-based data compression experiments
- **Multi-Optimizer Support:** LBFGS and Adam with configurable switching
- **Batch Training:** Configurable batch sizes and balanced wind angle sampling
- **ParaView Integration:** Full export pipeline for professional visualization

**Training Configuration**:
- Wind angles: [0°, 15°, 30°, 45°, 60°, 75°, 90°, 105°, 120°, 150°, 165°, 180°]
- Excluded for testing: [135°]
- Optimizers: Adam (lr=0.001) → optional switch to LBFGS
- Early stopping: Loss convergence monitoring with SMA (Simple Moving Average)
- Normalization: StandardScaler for both features and targets

### Module Structure

```
src/
├── models/
│   └── PINN.py (14KB) - Neural network with physics-informed methods
├── training/
│   ├── training.py (24KB) - Main training loops
│   ├── testing.py (17KB) - Evaluation and validation
│   └── training_definitions.py (8KB) - Training utilities
├── physics/
│   └── physics.py (22KB) - RANS equations and derivative scaling
├── boundary_conditions/
│   ├── boundary.py (10KB) - BC enforcement
│   └── boundary_testing.py (9KB) - BC validation
├── utils/
│   ├── definitions.py (119KB) - Core utilities and data loading
│   └── weighting.py (5KB) - Adaptive loss weighting
├── visualization/
│   ├── plotting.py (30KB) - Main plotting functions
│   ├── plotting_definitions.py (11KB) - Plotting utilities
│   └── paraview/ - ParaView export scripts
└── data/
    ├── scaled_cylinder_sphere.stl (29MB) - Geometry file
    └── Data processing scripts
```

### Experimental Results

**Completed Ablation Studies** (44 configurations in `experiments/ablations/`):
- Loss component combinations (data-only, data+continuity, all physics)
- Boundary condition variants (no-slip, inlet, combined)
- Adaptive vs fixed weighting schemes
- PCA compression levels (various n_components)
- FFT-based reduction experiments
- ParaView reduced-data training

### Improvements Over Previous Versions

1. **Modular Architecture:** Clear separation into packages
2. **Full Physics:** Complete RANS + continuity enforcement
3. **Production Quality:** Comprehensive logging, checkpointing, metrics
4. **Scalability:** Batch training, memory-efficient data handling
5. **Visualization:** Professional ParaView integration
6. **Experimentation:** Extensive ablation study framework

---

## Version 1.5 (December 2023)

**Location:** `deprecated/v5_modular_plotting/`
**Status:** Deprecated
**Date Range:** December 2023

### Changes from v1.4

- **Modularized Plotting:** Separated plotting into `plotting.py` and `plotting_definitions.py`
- **Separated Testing:** Introduced `testing.py` for model evaluation
- **Training Utilities:** Created `training_definitions.py` for reusable training functions
- **Simplified Architecture:** Cleaner code organization

### Architecture

- **Inputs:** 5 parameters (x, y, z, cos(θ), sin(θ))
- **Outputs:** 5 parameters (p, u, v, w, ν_t)
- **Layers:** Configurable hidden layers
- **Physics:** Basic boundary conditions

### Deprecated Reason

Superseded by v2.0 with comprehensive physics-informed losses, modular package structure, and production-ready features.

### Files

- `PINN.py` (12KB)
- `boundary.py` (9.4KB)
- `config.py` (4.0KB)
- `definitions.py` (26KB)
- `divergence.py` (17KB) - Divergence calculation utilities
- `main.py` (2.9KB)
- `plotting.py` (15KB)
- `plotting_definitions.py` (7.2KB)
- `testing.py` (5.4KB)
- `training.py` (7.4KB)
- `training_definitions.py` (5.2KB)
- `weighting.py` (4.7KB)

---

## Version 1.4 (November 2023)

**Location:** `deprecated/v4_enhanced_boundary/`
**Status:** Deprecated
**Date Range:** November 2023

### Changes from v1.3

- **Enhanced Boundary Implementation:** Improved no-slip boundary condition enforcement
- **Refined Penalty Terms:** Better tangential velocity penalty calculation
- **Config Experiments:** 5 configuration files testing different boundary approaches

### Architecture

- **Inputs:** 5 parameters (x, y, z, cos(θ), sin(θ))
- **Outputs:** 5 parameters (p, u, v, w, ν_t)
- **Physics:** Enhanced boundary conditions with improved penalty terms

### Deprecated Reason

Superseded by v1.5 with modular plotting architecture.

### Files

- `PINN.py` (12KB)
- `boundary.py` (8.6KB) - Enhanced implementation
- `config.py` (7.6KB)
- `definitions.py` (38KB)
- `main.py` (8.5KB)
- `plotting.py` (87KB) - Monolithic plotting
- `training.py` (20KB)
- `weighting.py` (482B)

---

## Version 1.3 (October-November 2023)

**Location:** `deprecated/v3_boundary_conditions/`
**Status:** Deprecated
**Date Range:** October 21 - November 26, 2023

### Changes from v1.2

- **Boundary Conditions Added:** First implementation of no-slip and inlet BCs
- **Weighting System:** Introduced adaptive loss weighting (`weighting.py`)
- **Multiple Experiments:** 4 active configs + 3 completed in `done/` subfolder

### Architecture

- **Inputs:** 5 parameters (x, y, z, cos(θ), sin(θ))
- **Outputs:** 5 parameters (p, u, v, w, ν_t)
- **Physics:** Data loss + continuity loss + boundary conditions

### Key Configurations

1. `config_21102023_adam_datalossonly_infinite.py` - Data loss baseline
2. `config_26112023_adam_datalossonly_infinite.py` - Refined data-only
3. `config_29102023_both_datalosscontloss_infinite.py` - Data + continuity
4. `config_11112023_adam_datalosscontloss_infinite.py` - Adam optimizer variant

**Completed Experiments:**
- Min-max normalization test
- Tanh activation function test
- 64-neuron architecture test

### Deprecated Reason

Enhanced boundary implementation in v1.4 provided better BC enforcement.

### Files

- `PINN.py` (11KB)
- `boundary.py` (7.3KB) - Initial BC implementation
- `config.py` (7.4KB)
- `definitions.py` (37KB)
- `main.py` (8.2KB)
- `plotting.py` (87KB)
- `training.py` (19KB)
- `weighting.py` (482B) - Initial adaptive weighting

---

## Version 1.2 (October 2023)

**Location:** `deprecated/v2_turbvisc_output/`
**Status:** Deprecated
**Date Range:** October 2023

### Changes from v1.1

- **Corrected I/O:** Turbulent viscosity (ν_t) moved from input to output
- **Removed RANS Loss:** Simplified to data-only training
- **5 Outputs:** Now predicting ν_t instead of providing it

### Architecture

**Inputs** (5): x, y, z, cos(θ), sin(θ)
**Outputs** (5): u, v, w, p, ν_t

**Rationale:** The neural network should learn to predict turbulent viscosity from flow conditions, not receive it as input. This makes the model more general and applicable to new scenarios.

### Physics

- Data loss only (MSE between predictions and CFD data)
- No physics-informed losses
- No boundary conditions

### Deprecated Reason

Missing physics-informed losses limited accuracy. Added in v1.3.

### Files

- `PINN.py` (5.1KB)
- `config.py` (3.9KB)
- `definitions.py` (46KB)
- `main.py` (13KB)
- `plotting.py` (1.8KB)
- `training.py` (13KB)
- `README` - Documents the change from v1.1

---

## Version 1.1 (October 2023)

**Location:** `deprecated/v1_initial/`
**Status:** Deprecated
**Date Range:** Early October 2023

### Initial Implementation

First version of the PINN implementation for wind flow simulation.

### Architecture

**Inputs** (9): x, y, z, θ, cos(θ), sin(θ), ν_t, ρ, ν
**Outputs** (4): u, v, w, p

### Physics

- RANS momentum loss included
- Physics-informed training with momentum equations

### Critical Flaw

**Problem:** Turbulent viscosity (ν_t) was provided as an input parameter.
**Impact:** Model couldn't generalize to new flow conditions since ν_t is typically unknown a priori.

### Deprecated Reason

Fundamental architectural flaw - ν_t should be predicted by the network, not provided as input. This made the model impractical for real-world applications where ν_t is unknown.

### Files

- `PINN.py` (5.0KB)
- `PINN_modf.py` (8.0KB) - Modified version
- `config.py` (3.9KB)
- `definitions.py` (71KB)
- `main.py` (13KB)
- `plotting.py` (14KB)
- `training.py` (13KB)
- `README` - Documents the initial approach

---

## Specialized Implementations (Various Dates)

### TPU Implementation

**Location:** `deprecated/TPU_implementation/`
**Status:** Abandoned
**Purpose:** Experimental Google Cloud TPU training

**Why Abandoned:**
- Limited TPU support for PyTorch compared to TensorFlow
- Standard GPU implementation proved sufficient for project needs
- Maintenance overhead not justified by performance gains

**Files:** Simplified versions of main codebase adapted for TPU compatibility

### Geometry-Specific Implementations

**Location:** `deprecated/geometry_specific_implementations/`
**Status:** Deprecated
**Purpose:** Separate implementations for different geometries

**Subdirectories:**
- `old_cylindersphere/` - Multiple dated versions for cylinder-sphere geometry
- `old_ladefense/` - La Défense urban geometry implementation
- `old_tpu/` - Additional TPU experiments
- `old_both/` - Unified implementation attempt (March 2024)
- `find_diff/` - Scripts for comparing code versions

**Why Deprecated:**
Unified implementation in v2.0 handles multiple geometries through configuration, eliminating need for separate codebases.

---

## Predecessor Research (Pre-October 2023)

**Location:** `deprecated/predecessor_researchers/`
**Status:** Reference Only

### Boyuan's Work

- **File:** `u_composite_model_p_nut_pure_NN_RANS_div_refinement_with_CFD.ipynb/py`
- **Approach:** Composite model combining NN predictions with CFD refinement
- **Contribution:** Demonstrated feasibility of hybrid NN-CFD approaches

### Wangzhe's Work

- **File:** `Parameterization_sphere_uvw-nut_self-consistent-nut_ref-last_trial.py`
- **Approach:** Sphere parameterization with self-consistent turbulent viscosity
- **Contribution:** Explored self-consistency conditions for ν_t prediction

**Impact on Current Work:**
These predecessor implementations informed the architecture decisions in v1.1+, particularly around:
- How to handle turbulent viscosity prediction
- Integration of physics-informed losses
- CFD data utilization strategies

---

## Version Timeline Summary

| Version | Date | Inputs | Outputs | Key Feature | Status |
|---------|------|--------|---------|-------------|--------|
| v1.1 | Oct 2023 | 9 (inc. ν_t) | 4 | RANS loss, flawed architecture | Deprecated |
| v1.2 | Oct 2023 | 5 | 5 (inc. ν_t) | Corrected I/O, data-only | Deprecated |
| v1.3 | Oct-Nov 2023 | 5 | 5 | Added boundary conditions | Deprecated |
| v1.4 | Nov 2023 | 5 | 5 | Enhanced boundaries | Deprecated |
| v1.5 | Dec 2023 | 5 | 5 | Modular plotting | Deprecated |
| **v2.0** | **Jan-Apr 2024** | **5** | **5** | **Full physics + production features** | **Current** |

---

## Migration Path

To migrate from any previous version to v2.0:

1. **Update imports:**
   ```python
   # Old
   from definitions import *
   from PINN import PINN
   from training import train_model

   # New
   from src.utils.definitions import *
   from src.models import PINN
   from src.training import train_model
   ```

2. **Update configuration:**
   - Move config files to `configs/` directory
   - Use standard `config.py` as template
   - Adjust paths to use new structure

3. **Update data paths:**
   - Point to data location (GitLab or local)
   - Ensure geometry file is in `src/data/`

4. **Run from repository root:**
   ```bash
   python main.py --config configs/your_config.py
   ```

---

## Lessons Learned

### Architecture Evolution
1. **ν_t as Output:** Critical correction in v1.2 - turbulent viscosity must be predicted
2. **Physics Matters:** Pure data-driven (v1.2) < physics-informed (v1.3+)
3. **Modular Design:** Separated concerns improved maintainability (v1.5, v2.0)

### Training Insights
4. **Adaptive Weighting:** Dynamic loss balancing outperformed fixed weights
5. **Multiple Optimizers:** Adam (fast exploration) + LBFGS (precise refinement) = best results
6. **Batch Training:** Essential for large datasets, balanced sampling critical

### Boundary Conditions
7. **Soft Enforcement:** Penalty-based BCs more stable than hard constraints
8. **Normal+Tangential:** Decomposing velocity at boundaries improved convergence
9. **Multiple BCs:** Combining no-slip + inlet + physics losses gave best accuracy

### Data Efficiency
10. **Model Order Reduction:** PCA preconditioning reduced training time
11. **Strategic Sampling:** Sparse wind angle training (12/13) achieved good interpolation
12. **Normalization:** StandardScaler outperformed MinMaxScaler

---

*Last Updated: November 2025*
*Current Version: 2.0.0*
*Repository: PINNs for Wind Flow Simulation*
*Organization: CNRS@CREATE & ARIA*

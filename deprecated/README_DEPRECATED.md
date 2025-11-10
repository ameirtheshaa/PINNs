# Deprecated Code Archive

This directory contains historical versions of the codebase that have been superseded by the current implementation in `src/`. These files are preserved for reference and to maintain development history.

## Directory Structure

### Version History (Oct 2023 - Dec 2023)

#### `v1_initial/` - October 2023
**First Version of PINN Implementation**

- **Input Parameters** (9): x, y, z, θ, cos(θ), sin(θ), ν_t, ρ, ν
- **Output Parameters** (4): u, v, w, p
- **Key Features**:
  - Included RANS momentum loss in physics-informed training
  - Turbulent viscosity (ν_t) provided as input parameter
- **Deprecated Reason**: Turbulent viscosity should be predicted by the network, not provided as input
- **Date Range**: October 2023
- **README**: See `v1_initial/README`

#### `v2_turbvisc_output/` - October 2023
**Turbulent Viscosity as Output**

- **Input Parameters** (5): x, y, z, cos(θ), sin(θ)
- **Output Parameters** (5): u, v, w, p, ν_t
- **Key Features**:
  - Corrected architecture with ν_t as prediction
  - Removed RANS momentum loss
- **Deprecated Reason**: Missing physics-informed losses
- **Date Range**: October 2023
- **README**: See `v2_turbvisc_output/README`

#### `v3_boundary_conditions/` - October-November 2023
**Added Boundary Condition Enforcement**

- **Input Parameters** (5): x, y, z, cos(θ), sin(θ)
- **Output Parameters** (5): u, v, w, p, ν_t
- **Key Features**:
  - Implemented boundary condition handling (no-slip, inlet)
  - Multiple experimental configurations testing different loss combinations
  - Introduced adaptive weighting schemes
- **Deprecated Reason**: Superseded by v4 with enhanced boundary implementation
- **Date Range**: October-November 2023
- **Files**:
  - Core: PINN.py, boundary.py, config.py, definitions.py, main.py, plotting.py, training.py, weighting.py
  - Configs: 4 experimental configurations
  - Done subfolder: 3 completed experiments

#### `v4_enhanced_boundary/` - November 2023
**Enhanced Boundary Conditions**

- **Input Parameters** (5): x, y, z, cos(θ), sin(θ)
- **Output Parameters** (5): u, v, w, p, ν_t
- **Key Features**:
  - Improved boundary condition implementation
  - Enhanced no-slip boundary enforcement
- **Deprecated Reason**: Superseded by v5 with modular architecture
- **Date Range**: November 2023
- **Files**: Similar structure to v3 with enhanced boundary.py (8.6KB)

#### `v5_modular_plotting/` - December 2023
**Modularized Visualization**

- **Input Parameters** (5): x, y, z, cos(θ), sin(θ)
- **Output Parameters** (5): u, v, w, p, ν_t
- **Key Features**:
  - Simplified architecture
  - Modularized plotting functions into separate files
  - Introduced plotting_definitions.py and testing.py
  - Separated training_definitions.py from main training loop
- **Deprecated Reason**: Superseded by main/current implementation with full feature set
- **Date Range**: December 2023
- **Files**: First version with modular structure (plotting.py, plotting_definitions.py, testing.py, training_definitions.py)

---

### Specialized Implementations

#### `TPU_implementation/` - Date Unknown
**Google TPU-Specific Implementation**

- **Purpose**: Experimental implementation for Google Cloud TPU training
- **Key Differences**:
  - Simplified loss functions (losses.py)
  - TPU-compatible training loop
  - Reduced feature set
- **Status**: Abandoned - standard GPU implementation preferred
- **Files**: PINN.py (1.8KB), config.py, definitions.py, losses.py, main.py, plotting.py, testing.py (20KB), training.py

#### `geometry_specific_implementations/` - Various Dates
**Old Geometry-Specific Code**

Contains implementations separated by geometry type and date:

- **old_cylindersphere/**: Multiple dated versions for cylinder-sphere geometry
- **old_ladefense/**: Implementations specific to La Défense urban geometry
- **old_tpu/**: Additional TPU experiments
- **old_both/**: Combined geometry implementations (March 2024)
- **find_diff/**: Diff scripts for comparing code versions

**Deprecated Reason**: Unified implementation in current codebase handles multiple geometries

---

### Predecessor Research Code

#### `predecessor_researchers/` - Pre-October 2023
**Previous Researchers' Work**

Contains work from previous team members that informed this project:

- **boyuan/**:
  - `u_composite_model_p_nut_pure_NN_RANS_div_refinement_with_CFD.ipynb` (20MB)
  - `u_composite_model_p_nut_pure_NN_RANS_div_refinement_with_CFD.py` (38KB)
  - Composite model approach with CFD refinement

- **wangzhe/**:
  - `Parameterization_sphere_uvw-nut_self-consistent-nut_ref-last_trial.py` (86KB)
  - Sphere parameterization with self-consistent turbulent viscosity

**Purpose**: Reference implementations showing evolution of approaches

---

## Migration Notes

### Current Implementation Location
The production-ready code is now located in:
- `src/` - Modular source code
- `configs/` - Configuration files
- `experiments/ablations/` - Completed experiments
- `main.py` - Main execution script

### Import Path Changes
If you need to reference old code, note that import paths have changed:

**Old structure:**
```python
from definitions import *
from PINN import PINN
from training import train_model
```

**New structure:**
```python
from src.utils.definitions import *
from src.models import PINN
from src.training import train_model
```

### Key Improvements in Current Version
1. **Modular package structure** - Clear separation of concerns
2. **Physics-informed losses** - Full RANS equation enforcement
3. **Model Order Reduction** - PCA and FFT experiments
4. **ParaView integration** - Professional visualization pipeline
5. **Adaptive weighting** - Dynamic loss component balancing
6. **Multi-geometry support** - Unified implementation for different cases

---

## File Preservation Policy

**Why Keep Deprecated Code?**
1. **Development History**: Shows evolution of methodology
2. **Reproducibility**: Allows recreation of intermediate results
3. **Learning Resource**: Documents what worked and what didn't
4. **Audit Trail**: Maintains complete research record

**Do Not Use For:**
- New development (use `src/` instead)
- Production runs (use `main.py` with configs from `configs/`)
- Publication results (use current implementation)

---

## Version Timeline Summary

| Version | Date | Key Feature | Status |
|---------|------|-------------|--------|
| v1 | Oct 2023 | Initial implementation, ν_t as input | Deprecated |
| v2 | Oct 2023 | ν_t as output | Deprecated |
| v3 | Oct-Nov 2023 | Boundary conditions added | Deprecated |
| v4 | Nov 2023 | Enhanced boundaries | Deprecated |
| v5 | Dec 2023 | Modular plotting | Deprecated |
| **Current** | **Jan-Apr 2024** | **Full feature set, production-ready** | **Active** |

---

## Contact

For questions about deprecated code or migration assistance, refer to:
- `docs/versioning/VERSION_HISTORY.md` - Detailed version changelog
- `docs/versioning/CURRENT_VERSION.md` - Current implementation details
- Main README.md - Project overview and current usage

---

*Last Updated: November 2025*
*Repository: PINNs for Wind Flow Simulation*
*Organization: CNRS@CREATE & ARIA*

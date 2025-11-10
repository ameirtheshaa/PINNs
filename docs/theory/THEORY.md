# Mathematical Theory and Methodology

This document provides the mathematical foundations for Physics-Informed Neural Networks (PINNs) applied to turbulent wind flow simulation using Reynolds-Averaged Navier-Stokes (RANS) equations.

---

## Table of Contents

1. [Problem Formulation](#1-problem-formulation)
2. [Governing Equations](#2-governing-equations)
3. [PINN Methodology](#3-pinn-methodology)
4. [Loss Function Derivation](#4-loss-function-derivation)
5. [Boundary Conditions](#5-boundary-conditions)
6. [Optimization Strategy](#6-optimization-strategy)
7. [Theoretical Justification](#7-theoretical-justification)

---

## 1. Problem Formulation

### 1.1 Physical System

We consider steady-state, incompressible turbulent wind flow around 3D obstacles (cylinder-sphere composite or urban geometries like La Défense).

**Domain:** $\Omega \subset \mathbb{R}^3$ with boundary $\partial\Omega = \Gamma_{inlet} \cup \Gamma_{outlet} \cup \Gamma_{wall}$

**Physical Parameters:**
- Fluid density: $\rho = 1 \text{ kg/m}^3$ (air)
- Kinematic viscosity: $\nu = 10^{-5} \text{ m}^2\text{/s}$
- Wind angle: $\theta \in [0°, 180°]$

**Unknown Fields:**
- Velocity: $\mathbf{u}(x,y,z,\theta) = [u, v, w]^T : \Omega \times [0,2\pi) \to \mathbb{R}^3$
- Pressure: $p(x,y,z,\theta) : \Omega \times [0,2\pi) \to \mathbb{R}$
- Turbulent viscosity: $\nu_t(x,y,z,\theta) : \Omega \times [0,2\pi) \to \mathbb{R}^+$

### 1.2 Computational Challenge

**Traditional CFD:**
- Requires mesh generation for each geometry and wind angle
- Computationally expensive (hours to days per simulation)
- Limited parametric exploration

**PINN Approach:**
- Learn continuous mapping: $(\mathbf{x}, \theta) \mapsto (\mathbf{u}, p, \nu_t)$
- Meshfree, data-efficient
- Fast inference for new parameters
- Physics-informed regularization reduces overfitting

---

## 2. Governing Equations

### 2.1 Reynolds-Averaged Navier-Stokes (RANS)

For incompressible turbulent flow, applying Reynolds decomposition and time-averaging yields:

**Continuity Equation:**
$$
\nabla \cdot \mathbf{u} = \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} + \frac{\partial w}{\partial z} = 0
\quad \tag{1}
$$

**Momentum Equations:**
$$
\begin{aligned}
u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} + w \frac{\partial u}{\partial z} &= -\frac{1}{\rho}\frac{\partial p}{\partial x} + \nu_{eff} \nabla^2 u \quad \tag{2a} \\[8pt]
u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} + w \frac{\partial v}{\partial z} &= -\frac{1}{\rho}\frac{\partial p}{\partial y} + \nu_{eff} \nabla^2 v \quad \tag{2b} \\[8pt]
u \frac{\partial w}{\partial x} + v \frac{\partial w}{\partial y} + w \frac{\partial w}{\partial z} &= -\frac{1}{\rho}\frac{\partial p}{\partial z} + \nu_{eff} \nabla^2 w \quad \tag{2c}
\end{aligned}
$$

where the **effective viscosity** is:
$$
\nu_{eff} = \nu + \nu_t
\quad \tag{3}
$$

### 2.2 Expanded Form

The Laplacian terms in equation (2) can be written as:

$$
\nabla^2 u = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} + \frac{\partial^2 u}{\partial z^2}
\quad \tag{4}
$$

In our PINN implementation, we use a symmetric form exploiting the continuity equation:

$$
\nu_{eff} \nabla^2 \mathbf{u} = \nu_{eff} \nabla(\nabla \cdot \mathbf{u}) + \nu_{eff} \nabla^2 \mathbf{u}
\quad \tag{5}
$$

Since $\nabla \cdot \mathbf{u} = 0$, we can write:

$$
\nu_{eff} \nabla^2 u = \nu_{eff}\left(2\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} + \frac{\partial^2 v}{\partial x \partial y} + \frac{\partial^2 u}{\partial z^2} + \frac{\partial^2 w}{\partial x \partial z}\right)
\quad \tag{6a}
$$

and similarly for $v$ and $w$ components.

**Implemented Form in `src/models/PINN.py:255-257`:**

```python
f = (u * u_x + v * u_y + w * u_z - (1 / rho) * p_x
     + nu_eff * (2 * u_xx) + nu_eff * (u_yy + v_xy) + nu_eff * (u_zz + w_xz))
g = (u * v_x + v * v_y + w * v_z - (1 / rho) * p_y
     + nu_eff * (v_xx + u_xy) + nu_eff * (2 * v_yy) + nu_eff * (v_zz + w_yz))
h = (u * w_x + v * w_y + w * w_z - (1 / rho) * p_z
     + nu_eff * (w_xx + u_xz) + nu_eff * (w_yy + v_yz) + nu_eff * (2 * w_zz))
```

Equations (2) are satisfied when $f = g = h = 0$ throughout $\Omega$.

### 2.3 Boundary Conditions

**Inlet (Γ<sub>inlet</sub>):**
$$
\mathbf{u}|_{\Gamma_{inlet}} = \mathbf{u}_{inlet}(\mathbf{x}, \theta)
\quad \tag{7}
$$

**No-Slip Walls (Γ<sub>wall</sub>):**
$$
\mathbf{u}|_{\Gamma_{wall}} = \mathbf{0}
\quad \tag{8}
$$

**Outlet (Γ<sub>outlet</sub>):**
$$
\frac{\partial \mathbf{u}}{\partial n}\bigg|_{\Gamma_{outlet}} = \mathbf{0}
\quad \quad \text{(natural BC, not enforced)}
\quad \tag{9}
$$

---

## 3. PINN Methodology

### 3.1 Neural Network Architecture

We approximate the solution fields using a deep neural network $\mathcal{N}_\theta$ with parameters $\theta$:

$$
\mathcal{N}_\theta : \mathbb{R}^5 \to \mathbb{R}^5
\quad \tag{10}
$$

**Input:** $\mathbf{z} = [x, y, z, \cos\theta, \sin\theta]^T \in \mathbb{R}^5$

**Output:** $\mathbf{y} = [p, u, v, w, \nu_t]^T \in \mathbb{R}^5$

**Architecture:**
$$
\mathbf{y} = \mathcal{N}_\theta(\mathbf{z}) = \mathbf{W}_L \sigma(\mathbf{W}_{L-1} \sigma(\cdots \sigma(\mathbf{W}_1 \mathbf{z} + \mathbf{b}_1) \cdots) + \mathbf{b}_{L-1}) + \mathbf{b}_L
\quad \tag{11}
$$

where:
- $L = 5$ layers (1 input + 4 hidden + 1 output)
- $\sigma(\cdot) = \text{ELU}(\cdot)$ = Exponential Linear Unit activation
- $\mathbf{W}_i \in \mathbb{R}^{n_i \times n_{i-1}}$, $\mathbf{b}_i \in \mathbb{R}^{n_i}$ are weights and biases
- Hidden layer sizes: $n_1 = n_2 = n_3 = n_4 = 128$

**ELU Activation:**
$$
\text{ELU}(x) = \begin{cases}
x & \text{if } x > 0 \\
\alpha(e^x - 1) & \text{if } x \leq 0
\end{cases}
\quad \text{with } \alpha = 1
\quad \tag{12}
$$

**Rationale for ELU:**
- Smooth everywhere (unlike ReLU)
- Non-zero gradient for negative inputs → mitigates vanishing gradients
- Reduces bias shift in hidden units
- Empirically better convergence for this problem

### 3.2 Encoding Wind Angle

**Issue:** Wind angle $\theta \in [0°, 360°)$ is periodic: $\theta = 0° \equiv 360°$

**Naive Approach:** Direct input of $\theta$ creates discontinuity at $0°/360°$ boundary.

**Solution:** Encode as $[\cos\theta, \sin\theta]^T$:
$$
\theta \mapsto [\cos\theta, \sin\theta]^T \in \mathbb{R}^2
\quad \tag{13}
$$

This preserves periodicity and smoothness: $\cos(0°) = \cos(360°)$ and $\sin(0°) = \sin(360°)$.

### 3.3 Automatic Differentiation

The key enabling technology for PINNs is automatic differentiation (autodiff) via PyTorch's computational graph.

For any scalar output $y_i$ from the network:
$$
\frac{\partial y_i}{\partial x} = \frac{\partial \mathcal{N}_\theta^{(i)}(\mathbf{z})}{\partial x}
\quad \tag{14}
$$

is computed exactly using the chain rule through the network layers.

**Example (continuity equation):**
```python
x.requires_grad_(True)  # Enable gradient tracking
y.requires_grad_(True)
z.requires_grad_(True)

output = model(torch.cat([x, y, z, cos_theta, sin_theta], dim=1))
u, v, w = output[:, 1], output[:, 2], output[:, 3]

u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
v_y = torch.autograd.grad(v.sum(), y, create_graph=True)[0]
w_z = torch.autograd.grad(w.sum(), z, create_graph=True)[0]

div = u_x + v_y + w_z  # Should be ≈ 0
```

**Second Derivatives (for Laplacian):**
```python
u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
```

The `create_graph=True` flag maintains the computation graph, enabling second-order derivatives required for the momentum equations.

---

## 4. Loss Function Derivation

### 4.1 Total Loss

The network parameters $\theta$ are optimized to minimize:

$$
\mathcal{L}_{total}(\theta) = \sum_{i} w_i \mathcal{L}_i(\theta)
\quad \tag{15}
$$

where $w_i$ are weighting coefficients (possibly adaptive) and $\mathcal{L}_i$ are individual loss components.

### 4.2 Data Loss

Let $\mathcal{D} = \{(\mathbf{z}_j, \mathbf{y}_j^{CFD})\}_{j=1}^N$ be CFD training data.

$$
\mathcal{L}_{data}(\theta) = \frac{1}{N}\sum_{j=1}^N \|\mathcal{N}_\theta(\mathbf{z}_j) - \mathbf{y}_j^{CFD}\|_2^2
\quad \tag{16}
$$

This is standard supervised learning: fit the network to observed data.

**Normalization:** Both inputs $\mathbf{z}$ and outputs $\mathbf{y}$ are standardized:
$$
\tilde{\mathbf{z}} = \frac{\mathbf{z} - \mu_z}{\sigma_z}, \quad \tilde{\mathbf{y}} = \frac{\mathbf{y} - \mu_y}{\sigma_y}
\quad \tag{17}
$$

using `sklearn.preprocessing.StandardScaler`.

### 4.3 Continuity Loss

Sample collocation points $\{\mathbf{z}_k^{col}\}_{k=1}^{N_{col}}$ in the domain $\Omega$.

$$
\mathcal{L}_{cont}(\theta) = \frac{1}{N_{col}}\sum_{k=1}^{N_{col}} \left|\frac{\partial u_\theta}{\partial x} + \frac{\partial v_\theta}{\partial y} + \frac{\partial w_\theta}{\partial z}\right|^2 \bigg|_{\mathbf{z}_k^{col}}
\quad \tag{18}
$$

where $u_\theta, v_\theta, w_\theta$ are the velocity components from $\mathcal{N}_\theta$.

**Implementation Note:** Collocation points can be:
1. **CFD data points:** Reuse training data locations
2. **Random sampling:** Uniformly sample $\Omega$
3. **Boundary-focused:** More points near walls

Currently, we use option 1 for memory efficiency.

### 4.4 Momentum Loss

Define RANS residuals at collocation points:

$$
\begin{aligned}
f_\theta(\mathbf{z}) &= u_\theta \frac{\partial u_\theta}{\partial x} + v_\theta \frac{\partial u_\theta}{\partial y} + w_\theta \frac{\partial u_\theta}{\partial z} - \frac{1}{\rho}\frac{\partial p_\theta}{\partial x} + \nu_{eff,\theta} \nabla^2 u_\theta \\[6pt]
g_\theta(\mathbf{z}) &= u_\theta \frac{\partial v_\theta}{\partial x} + v_\theta \frac{\partial v_\theta}{\partial y} + w_\theta \frac{\partial v_\theta}{\partial z} - \frac{1}{\rho}\frac{\partial p_\theta}{\partial y} + \nu_{eff,\theta} \nabla^2 v_\theta \\[6pt]
h_\theta(\mathbf{z}) &= u_\theta \frac{\partial w_\theta}{\partial x} + v_\theta \frac{\partial w_\theta}{\partial y} + w_\theta \frac{\partial w_\theta}{\partial z} - \frac{1}{\rho}\frac{\partial p_\theta}{\partial z} + \nu_{eff,\theta} \nabla^2 w_\theta
\end{aligned}
\quad \tag{19}
$$

with $\nu_{eff,\theta} = \nu + \nu_{t,\theta}$.

Then:
$$
\mathcal{L}_{mom}(\theta) = \frac{1}{N_{col}}\sum_{k=1}^{N_{col}} \left(|f_\theta(\mathbf{z}_k)|^2 + |g_\theta(\mathbf{z}_k)|^2 + |h_\theta(\mathbf{z}_k)|^2\right)
\quad \tag{20}
$$

### 4.5 No-Slip Boundary Loss

For points $\{\mathbf{z}_m^{wall}\}_{m=1}^{N_{wall}}$ on solid walls with outward normals $\{\mathbf{n}_m\}$:

**Decompose velocity into normal and tangential components:**
$$
\mathbf{u}_\theta(\mathbf{z}_m) = (\mathbf{u}_\theta \cdot \mathbf{n}_m)\mathbf{n}_m + \mathbf{u}_\theta^{\parallel}
\quad \tag{21}
$$

where $\mathbf{u}_\theta^{\parallel}$ is the tangential component:
$$
\mathbf{u}_\theta^{\parallel} = \mathbf{u}_\theta - (\mathbf{u}_\theta \cdot \mathbf{n}_m)\mathbf{n}_m
\quad \tag{22}
$$

**Relaxed No-Slip Loss (Nitsche-type penalty):**
$$
\mathcal{L}_{no-slip}(\theta) = \frac{1}{2\epsilon}\frac{1}{N_{wall}}\sum_{m=1}^{N_{wall}} \|\mathbf{u}_\theta^{\parallel}(\mathbf{z}_m)\|_2^2 + \frac{1}{N_{wall}}\sum_{m=1}^{N_{wall}} |(\mathbf{u}_\theta \cdot \mathbf{n}_m)|^2
\quad \tag{23}
$$

where $\epsilon > 0$ is a relaxation parameter (typically $\epsilon = 0.01$).

**Rationale:**
- First term: Penalty for non-zero tangential velocity
- Second term: Enforce zero normal velocity (impermeability)
- $\epsilon$ controls stiffness of tangential penalty

**Implementation (`src/models/PINN.py:57-70`):**
```python
def compute_tangential_velocity(velocities, normals):
    normal_velocities = torch.sum(velocities * normals, dim=1, keepdim=True) * normals
    tangential_velocities = velocities - normal_velocities
    return tangential_velocities, normal_velocities

penalty = torch.mean(tangential_velocities**2) / (2*epsilon)
normal_loss = MSELoss()(normal_velocities, torch.zeros_like(normal_velocities))
no_slip_loss = penalty + normal_loss
```

### 4.6 Inlet Boundary Loss

For inlet points $\{\mathbf{z}_l^{inlet}\}_{l=1}^{N_{inlet}}$ with prescribed velocity $\mathbf{u}_{inlet}(\mathbf{z}_l, \theta)$:

$$
\mathcal{L}_{inlet}(\theta) = \frac{1}{N_{inlet}}\sum_{l=1}^{N_{inlet}} \|\mathbf{u}_\theta(\mathbf{z}_l^{inlet}) - \mathbf{u}_{inlet}(\mathbf{z}_l)\|_2^2
\quad \tag{24}
$$

### 4.7 Adaptive Weighting

The weights $w_i$ in equation (15) can be:

**1. Fixed:** Set manually based on physical intuition
```python
w_data = 1.0
w_cont = 0.1
w_mom = 0.01
```

**2. Adaptive (time-based):**
$$
w_{data}(t) = w_{init} + (w_{final} - w_{init})\frac{t}{T}
\quad \tag{25}
$$

where $t$ is current epoch, $T$ is total epochs.

**Rationale:** Start with strong data supervision, gradually increase physics weight.

**3. Gradient-based:**
$$
w_i \propto \frac{1}{\|\nabla_\theta \mathcal{L}_i\|}
\quad \tag{26}
$$

Balance gradients from different loss components.

**Implementation (`src/utils/weighting.py`):**
```python
def adaptive_weighting(current_epoch, total_epochs, init_weight, final_weight):
    return init_weight + (final_weight - init_weight) * (current_epoch / total_epochs)
```

---

## 5. Boundary Conditions

### 5.1 Boundary Extraction from STL Mesh

The geometry is provided as an STL (stereolithography) file: `src/data/scaled_cylinder_sphere.stl`.

**Algorithm:**
1. Load STL mesh: `mesh = stl.mesh.Mesh.from_file('geometry.stl')`
2. Extract surface triangles: Each triangle has 3 vertices $\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3$
3. Compute triangle normal: $\mathbf{n} = \frac{(\mathbf{v}_2 - \mathbf{v}_1) \times (\mathbf{v}_3 - \mathbf{v}_1)}{\|(\mathbf{v}_2 - \mathbf{v}_1) \times (\mathbf{v}_3 - \mathbf{v}_1)\|}$
4. Sample points on triangles (uniform or stratified)
5. Interpolate normals at sample points

**Implementation (`src/boundary_conditions/boundary.py`):**
```python
from stl import mesh

def extract_boundary_points(geometry_file, n_samples=1000):
    mesh_data = mesh.Mesh.from_file(geometry_file)
    # ... sampling and normal computation
    return boundary_points, normals
```

### 5.2 Inlet Profile

For wind flow, a logarithmic profile is often prescribed:

$$
u_{inlet}(z) = \frac{u_*}{\kappa}\ln\left(\frac{z}{z_0}\right)
\quad \tag{27}
$$

where:
- $u_*$ = friction velocity
- $\kappa = 0.41$ = von Kármán constant
- $z_0$ = roughness length

In practice, we use CFD data at the inlet plane as $\mathbf{u}_{inlet}$.

---

## 6. Optimization Strategy

### 6.1 Two-Stage Training

**Stage 1: Adam Optimizer**
- **Objective:** Fast exploration of parameter space
- **Learning rate:** $\alpha = 0.001$
- **Update rule:**
  $$
  \theta_{t+1} = \theta_t - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
  \quad \tag{28}
  $$
  where $\hat{m}_t$ and $\hat{v}_t$ are bias-corrected first and second moment estimates.

**Stage 2: L-BFGS Optimizer (Optional)**
- **Objective:** Fine-tune to local minimum
- **Method:** Limited-memory Broyden-Fletcher-Goldfarb-Shanno
- **Quasi-Newton:** Approximates Hessian using gradient history
- **Line search:** Strong Wolfe conditions

**Rationale:**
- Adam handles high-dimensional, non-convex landscapes well
- L-BFGS provides precise convergence near minima
- Two-stage approach balances speed and accuracy

### 6.2 Early Stopping

Monitor loss convergence using Simple Moving Average (SMA):

$$
\text{SMA}_t = \frac{1}{W}\sum_{i=t-W+1}^t \mathcal{L}_{total}^{(i)}
\quad \tag{29}
$$

where $W = 1000$ is the window size.

**Stop if:**
$$
|\text{SMA}_t - \text{SMA}_{t-1}| < \epsilon_{tol} \quad \text{for } C \text{ consecutive iterations}
\quad \tag{30}
$$

with $\epsilon_{tol} = 10^{-5}$ and $C = 10$.

**Implementation (`src/training/training.py:36-39`):**
```python
recent_losses = collections.deque(maxlen=1000)
if abs(current_sma - previous_sma) < threshold:
    consecutive_count += 1
    if consecutive_count >= 10:
        break  # Early stop
```

### 6.3 Learning Rate Scheduling

For Adam, we optionally use:

**Exponential decay:**
$$
\alpha_t = \alpha_0 \cdot \gamma^{t/T}
\quad \tag{31}
$$

**Cosine annealing:**
$$
\alpha_t = \alpha_{min} + \frac{1}{2}(\alpha_{max} - \alpha_{min})\left(1 + \cos\left(\frac{t\pi}{T}\right)\right)
\quad \tag{32}
$$

Currently not implemented but planned for future versions.

---

## 7. Theoretical Justification

### 7.1 Universal Approximation Theorem

**Theorem (Cybenko, 1989; Hornik et al., 1989):**
Let $\sigma$ be a continuous, non-polynomial activation function (e.g., ELU). Then, for any continuous function $f : \mathbb{R}^n \to \mathbb{R}^m$ on a compact set $K \subset \mathbb{R}^n$ and any $\epsilon > 0$, there exists a neural network $\mathcal{N}$ with one hidden layer such that:

$$
\sup_{\mathbf{x} \in K} \|f(\mathbf{x}) - \mathcal{N}(\mathbf{x})\| < \epsilon
\quad \tag{33}
$$

**Implication:** Our 4-layer network can approximate the solution $(\mathbf{u}, p, \nu_t)$ arbitrarily well, given sufficient width.

**Practical Note:** Deep networks (multiple layers) often require fewer neurons than shallow networks for the same accuracy, especially for problems with hierarchical structure.

### 7.2 Physics-Informed Regularization

**Problem:** Pure data-driven learning (minimizing only $\mathcal{L}_{data}$) may overfit and produce non-physical solutions.

**Solution:** Adding physics losses $\mathcal{L}_{cont}$, $\mathcal{L}_{mom}$, $\mathcal{L}_{BC}$ acts as **regularization**.

**Bayesian Interpretation:**
Minimizing $\mathcal{L}_{total}$ is equivalent to maximum a posteriori (MAP) estimation:

$$
\theta^* = \arg\max_\theta p(\theta | \mathcal{D}) = \arg\max_\theta p(\mathcal{D} | \theta) p(\theta)
\quad \tag{34}
$$

where:
- $p(\mathcal{D} | \theta) \propto \exp(-\mathcal{L}_{data})$ (likelihood)
- $p(\theta) \propto \exp(-\mathcal{L}_{physics})$ (prior encoding physical laws)

Thus, physics losses encode our prior belief that solutions must satisfy RANS equations.

### 7.3 Generalization to Unseen Parameters

**Key Advantage of PINNs:** By learning the continuous mapping $(\mathbf{x}, \theta) \mapsto (\mathbf{u}, p, \nu_t)$, the network can interpolate/extrapolate to unseen wind angles.

**Empirical Observation:**
- Training on angles: [0°, 15°, 30°, ..., 180°] excluding 135°
- Testing on 135° yields accurate predictions (see RESULTS.md)

**Theoretical Basis:**
The smoothness of the solution with respect to parameters is captured if:
1. The network is smooth (✓ ELU is $C^1$)
2. Sufficient sampling density in parameter space (✓ 12/13 angles)
3. Physics constraints prevent unphysical extrapolation (✓ RANS regularization)

### 7.4 Convergence Analysis

**No General Convergence Guarantee:** PINNs solve a non-convex optimization problem. Global convergence to the true PDE solution is not guaranteed.

**Practical Convergence:** Empirically, PINNs converge to accurate solutions when:
1. **Network capacity is sufficient:** $128$ neurons × $4$ layers ≈ $67K$ parameters > data points
2. **Loss weighting is balanced:** No single loss dominates
3. **Optimizer is robust:** Two-stage Adam+L-BFGS
4. **Data quality is high:** CFD data is accurate and well-distributed

**Failure Modes:**
- **Gradient pathologies:** Vanishing/exploding gradients (mitigated by ELU, careful initialization)
- **Spurious minima:** Local minima that satisfy losses but not true solution (mitigated by physics regularization)
- **Stiff PDEs:** High-frequency phenomena require specialized architectures (not an issue for steady RANS)

### 7.5 Comparison with Traditional CFD

| Aspect | Traditional CFD | PINN |
|--------|----------------|------|
| **Mesh** | Required | Meshfree |
| **Time per simulation** | Hours to days | Minutes (inference) |
| **Parametric studies** | Rerun for each parameter | Single training, fast queries |
| **Physical consistency** | Guaranteed (if converged) | Soft constraint via losses |
| **Data efficiency** | N/A (generates own data) | Needs CFD data for training |
| **Uncertainty quantification** | Limited | Possible (Bayesian PINNs) |

**Complementary Roles:**
- CFD: Generate high-fidelity training data
- PINN: Learn surrogate model for fast parametric exploration

---

## 8. Computational Complexity

### 8.1 Training Cost

**Forward Pass:**
- Input: $N \times 5$ (N points, 5 features)
- Layer 1: $5 \times 128$ + bias → $128N$ operations
- Layers 2-4: $(128 \times 128) \times 3 = 49,152$ operations per point
- Output: $128 \times 5$ → $640N$ operations
- **Total:** $\mathcal{O}(N)$ with large constant ~$50K$

**Backward Pass (Autodiff):**
- Same order as forward pass: $\mathcal{O}(N)$
- **Physics losses add:** Second-order derivatives → $\times 3$ computational cost
- **Total:** $\mathcal{O}(3N)$ for full physics-informed training

**Per Epoch:**
- For $N = 100,000$ points: ~0.5 seconds on NVIDIA A100
- Full training (20,000 epochs): ~3-4 hours

### 8.2 Memory Complexity

**Model Parameters:**
- Weights: $(5 \times 128) + (128 \times 128) \times 3 + (128 \times 5) = 50,560$ parameters
- At float32: $50,560 \times 4 = 202$ KB (negligible)

**Activation Storage (for backprop):**
- Need to store all layer activations: $N \times 128 \times 4 \text{ layers}$
- For $N = 100,000$: $100K \times 128 \times 4 \times 4 \text{ bytes} \approx 205$ MB

**Gradient Computation:**
- PyTorch autodiff creates computation graph
- Memory scales with depth and batch size
- **Typical:** 2-3 GB for $N = 100,000$ with physics losses

**Optimization:**
- Use gradient checkpointing for very large $N$
- Batch processing if memory-constrained

---

## 9. Numerical Considerations

### 9.1 Derivative Scaling

**Issue:** After normalizing inputs/outputs using StandardScaler, derivatives must be rescaled.

**Chain Rule:**
If $\tilde{u} = (u - \mu_u)/\sigma_u$ and $\tilde{x} = (x - \mu_x)/\sigma_x$, then:

$$
\frac{\partial u}{\partial x} = \frac{\sigma_u}{\sigma_x} \frac{\partial \tilde{u}}{\partial \tilde{x}}
\quad \tag{35}
$$

**Second Derivative:**
$$
\frac{\partial^2 u}{\partial x^2} = \frac{\sigma_u}{\sigma_x^2} \frac{\partial^2 \tilde{u}}{\partial \tilde{x}^2}
\quad \tag{36}
$$

**Implementation (`src/physics/physics.py:17-50`):**
```python
der_dict["u_x"] = der_dict["u_x"] * (stds_means_dict['Velocity_X_std'] /
                                      stds_means_dict['X_std'])
der_dict["u_xx"] = der_dict["u_xx"] * (stds_means_dict['Velocity_X_std'] /
                                        (stds_means_dict['X_std']**2))
```

**Critical:** Failure to rescale derivatives leads to incorrect physics residuals and poor convergence.

### 9.2 Numerical Stability

**Finite Precision:**
- All computations in float32 (or float64 for critical cases)
- Check for NaN/Inf in loss values
- Gradient clipping if gradients explode

**Condition Numbers:**
- Normalization keeps all variables $\mathcal{O}(1)$ → well-conditioned
- RANS residuals naturally scaled $\mathcal{O}(1)$ after normalization

### 9.3 Verification

**Method of Manufactured Solutions (MMS):**
1. Choose analytical solution $\mathbf{u}_{exact}(\mathbf{x}, \theta)$, $p_{exact}$
2. Compute source terms $\mathbf{f}$ by substituting into RANS
3. Train PINN with source terms
4. Compare $\mathcal{N}_\theta$ vs. $(\mathbf{u}_{exact}, p_{exact})$

**Currently:** Verification done by comparing to CFD data (which is itself verified).

---

## 10. Extensions and Future Work

### 10.1 Turbulence Modeling

**Current:** Turbulent viscosity $\nu_t$ is predicted by the network (data-driven).

**Alternative:** Incorporate algebraic turbulence models:
$$
\nu_t = C_\mu \frac{k^2}{\epsilon}
\quad \tag{37}
$$

where $k$ (turbulent kinetic energy) and $\epsilon$ (dissipation rate) are additional unknowns.

### 10.2 Unsteady Flows

**Current:** Steady-state RANS.

**Extension:** Time-dependent RANS or LES (Large Eddy Simulation):
$$
\frac{\partial \mathbf{u}}{\partial t} + \mathbf{u} \cdot \nabla \mathbf{u} = -\frac{1}{\rho}\nabla p + \nu_{eff}\nabla^2 \mathbf{u}
\quad \tag{38}
$$

Add time $t$ as an additional input: $\mathcal{N}_\theta(\mathbf{x}, \theta, t)$.

### 10.3 Multi-Fidelity PINNs

Combine low-fidelity (coarse CFD) and high-fidelity (fine CFD) data:
$$
\mathcal{L}_{total} = w_{HF}\mathcal{L}_{HF} + w_{LF}\mathcal{L}_{LF} + \mathcal{L}_{physics}
\quad \tag{39}
$$

### 10.4 Uncertainty Quantification

**Bayesian PINNs:**
Place prior distribution on weights: $p(\theta)$.
Posterior: $p(\theta | \mathcal{D}) \propto p(\mathcal{D} | \theta)p(\theta)$.
Sample using MCMC or variational inference.

**Output:** Predictive distribution $p(\mathbf{u}, p, \nu_t | \mathbf{x}, \theta, \mathcal{D})$ with uncertainty bounds.

---

## 11. References

### Theoretical Foundations

1. **Raissi, M., Perdikaris, P., & Karniadakis, G. E.** (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686-707.

2. **Cybenko, G.** (1989). Approximation by superpositions of a sigmoidal function. *Mathematics of Control, Signals and Systems*, 2(4), 303-314.

3. **Hornik, K., Stinchcombe, M., & White, H.** (1989). Multilayer feedforward networks are universal approximators. *Neural Networks*, 2(5), 359-366.

### RANS and Turbulence

4. **Pope, S. B.** (2000). *Turbulent Flows*. Cambridge University Press.

5. **Wilcox, D. C.** (2006). *Turbulence Modeling for CFD* (3rd ed.). DCW Industries.

### PINNs for Fluid Mechanics

6. **Jin, X., Cai, S., Li, H., & Karniadakis, G. E.** (2021). NSFnets (Navier-Stokes flow nets): Physics-informed neural networks for the incompressible Navier-Stokes equations. *Journal of Computational Physics*, 426, 109951.

7. **Mao, Z., Jagtap, A. D., & Karniadakis, G. E.** (2020). Physics-informed neural networks for high-speed flows. *Computer Methods in Applied Mechanics and Engineering*, 360, 112789.

### Optimization and Training

8. **Kingma, D. P., & Ba, J.** (2014). Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980*.

9. **Liu, D. C., & Nocedal, J.** (1989). On the limited memory BFGS method for large scale optimization. *Mathematical Programming*, 45(1-3), 503-528.

### Boundary Conditions

10. **Nitsche, J.** (1971). Über ein Variationsprinzip zur Lösung von Dirichlet-Problemen bei Verwendung von Teilräumen, die keinen Randbedingungen unterworfen sind. *Abhandlungen aus dem mathematischen Seminar der Universität Hamburg*, 36(1), 9-15.

---

*Last Updated: November 2025*
*Document Version: 1.0*
*Part of: PINNs for Wind Flow Simulation - Publication-Ready Codebase*

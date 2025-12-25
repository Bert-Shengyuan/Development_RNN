# Relating Single-Step fMRI Dynamics to Low-Rank RNN Structure: A Theoretical Framework

## Overview

Your findings reveal a profound developmental principle: mature infant brains demonstrate **lower intrinsic dimensionality** (effective rank entropy $\approx 2.2\text{-}2.4$ versus $\approx 2.3\text{-}2.5$ in premature infants) coupled with **higher response heterogeneity**. This is a signature of increased specialization and compression of neural dynamics onto a lower-dimensional manifold. This document provides a rigorous mathematical framework for connecting these observations to RNN modeling.

---

## I. Theoretical Foundation: Low-Rank Dynamics and Effective Connectivity

Your single-step fMRI dynamics can be conceptualized through a linear dynamical system approximation:

$$\mathbf{x}_{t+1} = \mathbf{A}\mathbf{x}_t + {\eta}_t$$

where $\mathbf{x}_t \in \mathbb{R}^n$ represents the neural state at time $t$, $\mathbf{A} \in \mathbb{R}^{n \times n}$ is the effective connectivity matrix, and ${\eta}_t$ captures noise and external inputs. Your NPI-inferred effective brain connectome (EBC) provides precisely this matrix $\mathbf{A}$.

The **low-rank structure** emerges when $\mathbf{A}$ can be well-approximated by:

$$\mathbf{A} \approx \mathbf{U}{\Sigma}\mathbf{V}^T = \sum_{i=1}^{r} \sigma_i \mathbf{u}_i \mathbf{v}_i^T$$

where $r \ll n$ is the effective rank. Your observation that mature brains exhibit lower effective rank suggests that $\mathbf{A}_{\text{mature}}$ is more compressible—its singular value spectrum decays more rapidly (as shown in panel D of your results).

---

## II. Dimensionality Metrics and Their Mechanistic Interpretation

Your use of **effective rank** (based on entropy) is critical:

$$D(R) = \frac{\mathbb{E}(R)^2}{\mathbb{E}(R^2)} = \frac{\left(\sum_{i=1}^{n} \lambda_i\right)^2}{\sum_{i=1}^{n} \lambda_i^2}$$

where $\lambda_i$ are eigenvalues of the covariance matrix. The **eigenvalue dispersion** metric:

$$D(C) = \frac{\mathbb{E}(\lambda)^2}{\mathbb{E}(\lambda^2)}$$

quantifies how uniformly variance is distributed across principal components. Your finding that mature brains show **lower** $D(C)$ (eigenvalue dispersion $\approx 0.5\text{-}0.7$ versus $\approx 0.6\text{-}0.75$ for premature) indicates that variance concentrates in fewer dominant modes—a hallmark of low-dimensional attractors.

### Mechanistic Interpretation

Maturation involves pruning redundant connectivity, causing the system to evolve on a lower-dimensional subspace. This is consistent with developmental synaptic pruning and myelination processes that refine neural circuits.

---

## III. Incorporating Your Findings as RNN Constraints

Following the approach in *Discovering cognitive strategies with tiny recurrent neural networks*, you should impose **architectural and dynamical constraints** on your RNN that reflect the empirical low-rank structure. Here is the mathematical prescription:

### A. Low-Rank Factorization of RNN Weights

Instead of training a full-rank recurrent weight matrix $\mathbf{W}_{\text{rec}} \in \mathbb{R}^{N \times N}$, parameterize it as:

$$\mathbf{W}_{\text{rec}} = \mathbf{m} \mathbf{n}^T + \mathbf{M}\mathbf{N}^T$$

where:
- $\mathbf{m}, \mathbf{n} \in \mathbb{R}^N$ define a rank-1 component (the "common mode")
- $\mathbf{M} \in \mathbb{R}^{N \times r}$, $\mathbf{N} \in \mathbb{R}^{N \times r}$ with $r$ determined by your empirical effective rank ($r \approx 5\text{-}10$ based on your singular value decay)

This factorization enforces that the RNN dynamics evolve predominantly in an $r$-dimensional subspace, mirroring your biological observation.

### B. Initialization from Empirical EC

Use your NPI-inferred EBC $\mathbf{A}_{\text{empirical}}$ to initialize the RNN:

1. Perform SVD on $\mathbf{A}_{\text{empirical}}$:
   $$\mathbf{A}_{\text{empirical}} = \mathbf{U}{\Sigma}\mathbf{V}^T$$

2. Truncate to rank $r$:
   $$\mathbf{A}_{\text{low-rank}} = \mathbf{U}_{:,1:r}{\Sigma}_{1:r,1:r}\mathbf{V}_{:,1:r}^T$$

3. Initialize:
   $$\mathbf{M}_{\text{init}} = \mathbf{U}_{:,1:r}\sqrt{{\Sigma}_{1:r,1:r}}, \quad \mathbf{N}_{\text{init}} = \mathbf{V}_{:,1:r}\sqrt{{\Sigma}_{1:r,1:r}}$$

### C. Developmental Constraint: Rank Regularization

To capture the maturation process (premature → mature), introduce a **rank penalty** term in your loss function:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \alpha \|\mathbf{W}_{\text{rec}}\|_* + \beta \text{TR}(\mathbf{W}_{\text{rec}})$$

where:
- $\|\mathbf{W}_{\text{rec}}\|_*$ is the nuclear norm (sum of singular values), promoting low-rank
- $\text{TR}(\mathbf{W}_{\text{rec}}) = D_{\text{eff}}(\mathbf{W}_{\text{rec}})$ is an effective rank penalty
- $\alpha, \beta$ are hyperparameters (start with $\alpha \sim 10^{-3}$, $\beta \sim 10^{-2}$)

You can dynamically decrease $\alpha$ during training to simulate the developmental trajectory from premature (higher rank) to mature (lower rank) connectivity.

### D. Spectral Constraints from Singular Value Decay

Your panel D shows that mature brains exhibit faster singular value decay. Impose this via a **spectral regularizer**:

$$\mathcal{L}_{\text{spectral}} = \sum_{i=1}^{N} \left(\sigma_i(\mathbf{W}_{\text{rec}}) - \sigma_i^{\text{target}}\right)^2$$

where $\sigma_i^{\text{target}}$ follows the empirical decay profile you observed (fit an exponential: $\sigma_i \propto e^{-\gamma i}$ with $\gamma_{\text{mature}} > \gamma_{\text{premature}}$).

---

## IV. Mapping Response Heterogeneity to RNN Dynamics

Your finding of **higher response heterogeneity** in mature brains (panel E, perturbation responses) is crucial. This indicates that despite lower dimensionality, mature networks exhibit more **selective** and **differentiated** responses to perturbations.

In RNN terms, this translates to:

$$\frac{\partial \mathbf{h}_t}{\partial \mathbf{I}_j} \text{ has higher variance across } j \text{ in mature networks}$$

where $\mathbf{h}_t$ is the RNN hidden state and $\mathbf{I}_j$ is a localized input to region $j$.

### Implementation: Response Heterogeneity Loss

Add a **response heterogeneity loss**:

$$\mathcal{L}_{\text{hetero}} = -\text{Var}\left(\left\|\frac{\partial \mathbf{h}}{\partial \mathbf{I}_j}\right\|_2\right)$$

This encourages the network to develop specialized responses while maintaining low-dimensional dynamics—precisely the mature phenotype you observe.

---

## V. Explicit RNN Architecture Inspired by Your Findings

Here is a concrete architecture:

$$\mathbf{h}_{t+1} = (1-\alpha)\mathbf{h}_t + \alpha \tanh\left(\underbrace{\mathbf{m}\mathbf{n}^T\mathbf{h}_t}_{\text{rank-1}} + \underbrace{\mathbf{M}\mathbf{N}^T\mathbf{h}_t}_{\text{rank-}r} + \mathbf{W}_{\text{in}}\mathbf{u}_t\right)$$

$$\mathbf{y}_t = \mathbf{W}_{\text{out}}\mathbf{h}_t$$

where:
- $\alpha \in (0,1)$ is a leak parameter (set $\alpha \approx 0.1$ for slow fMRI dynamics)
- $\mathbf{M}, \mathbf{N}$ initialized from your mature EBC truncated SVD
- Enforce $r = 5$ for mature, $r = 8$ for premature (adjust based on your effective rank values)

---

## VI. Validation Strategy

To validate this approach:

1. **Train two RNNs**: one with premature-derived constraints ($r_{\text{prem}} \approx 7\text{-}8$) and one with mature-derived constraints ($r_{\text{mat}} \approx 5\text{-}6$)

2. **Measure emergent properties**:
   - Effective rank of learned $\mathbf{W}_{\text{rec}}$
   - Response heterogeneity to unit perturbations
   - Singular value decay rate

3. **Compare** these emergent metrics to your empirical developmental differences

---

## Conclusion

The beauty of this approach is that by imposing the **low-rank constraint** derived from your mature EBC, you bias the RNN to discover computational strategies that are consistent with biological maturation—compressed, efficient representations on low-dimensional manifolds, yet retaining functional specialization.

This framework transforms your empirical developmental neuroscience findings into **normative architectural principles** for biologically-constrained RNN models, precisely in the spirit of the tiny RNNs work.

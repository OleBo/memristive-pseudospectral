# Trajectory-dependent pseudospectral stability in self-organizing memristive networks

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://github.com/OleBo/memristive-pseudospectral/workflows/CI/badge.svg)](https://github.com/OleBo/memristive-pseudospectral/actions)
[![Deploy](https://github.com/OleBo/memristive-pseudospectral/workflows/Deploy%20research%20site/badge.svg)](https://OleBo.github.io/memristive-pseudospectral/)

**Reproducible computational study by Olaf Bochmann**  
Status: research prototype / independent reproduction

This repository investigates a focused question arising from the analysis of Caravelli, Sheldon & Traversa (Science Advances, 2021):

> **Can the cooperative escape dynamics of self-organizing memristive networks be characterized by trajectory-dependent spectral or pseudospectral quantities, even when the local equilibrium Jacobian is stable?**

The original paper explicitly examined non-normality of the Jacobian near local minima and reported that it did **not** account for the observed transient instability. Its Supplementary Material reports a numerical observation: as control parameter $s$ increases from zero, the equilibrium undergoes a sharp transition from a regime where escape succeeds to one where it fails. This repository explores whether trajectory-dependent spectral objects (numerical abscissa, resolvent norm, or generalized growth rates) can characterize this transition despite the frozen equilibrium Jacobian being stable throughout.

## What is in this repository?

### Pass I — computational prototype

The initial exploratory calculation used simplified random orthogonal projectors and finite-difference Jacobians. It was useful for testing the research idea but was not suitable as a publication-quality reconstruction.

### Pass II — source-aligned reconstruction

The main implementation now uses:

- the network equation
  $$\dot{x}=\frac{1}{\beta}(I-\chi\Omega X)^{-1}\Omega S-\alpha x,$$
- an orthogonal projector constructed as
  $$\Omega=A^T(AA^T)^{-1}A,$$
  with iid $A_{ij}\sim U(0,1)$,
- the source-paper parameters $\alpha=\beta=1$, $\chi=0.9$, $N=200$, and the main-text statement $\mathop{\text{rank}}(\Omega)=N-M=150$ for $M=50$,
- the $p=100$ smooth non-absorbing boundary window used in the paper,
- Biolek $p=2$, Prodromakis $p=1$, and Joglekar $p=1$ window functions,
- the nonlinear $\sinh$ current model with $i_c=T=R_{off}=1$,
- analytic Jacobians derived from the matrix-inverse identity, and
- direct propagator, numerical-abscissa and resolvent/Kreiss diagnostics.

The Supplementary Material separately gives a basin-of-attraction experiment with $N=200$, $N_c=100$, and $s\approx0.22$. That ensemble is exposed as a separate configuration rather than silently baked into the main output.

## Current exact-network result

For the deterministic reconstruction used in `results/exact/` at $s=0.15$, every tested model has negative spectral and numerical abscissa and no Euclidean transient gain above one in the sample period $[0, 5]$:

| model | spectral abscissa | numerical abscissa | max $\|e^{Jt}\|_2$ |
|---|---:|---:|---:|
| standard | -0.7960 | -0.7960 | 1.000 |
| source boundary, $p=100$ | -0.7960 | -0.7960 | 1.000 |
| Joglekar $p=1$ | -0.4616 | -0.4616 | 1.000 |
| Prodromakis $p=1$ | -0.1154 | -0.1154 | 1.000 |
| Biolek $p=2$ | -0.7950 | -0.7950 | 1.000 |
| $\sinh$ current | -0.7918 | -0.7918 | 1.000 |

These numbers are **our independent numerical reconstruction**, not numbers copied from the paper. They should not be interpreted as an exact reproduction of every figure in the paper because the paper does not specify the full componentwise control vector $S$.

The important qualitative observation is robust in this reconstruction: at the tested stable equilibrium, the symmetric part of the Jacobian is negative definite and the frozen linear propagator does not exhibit transient gain.

## Why the trajectory question matters

A frozen-equilibrium calculation studies

$$\delta\dot{x}=J(x^\*)\delta x \qquad \delta x(t)=e^{J(x^\*)t}\delta x(0).$$

A nonlinear escape trajectory instead generates the non-autonomous tangent equation

$$\dot{\Phi}(t,t_0)=J(x(t))\Phi(t,t_0),\qquad \Phi(t_0,t_0)=I,$$

so that

$$\Phi(t,t_0)=\mathcal{T}\exp\left(\int_{t_0}^{t}J(x(\tau))\,d\tau\right).$$

The matrices at different times generally need not commute. Consequently, the relevant finite-time object is not determined by the spectrum of one equilibrium Jacobian. This motivates monitoring spectral abscissa, numerical abscissa, and resolvent norm along the trajectory.

## Exact analytic Jacobian

For
$$P=(I-\chi\Omega X)^{-1}\Omega,\qquad V=PS,$$
the derivative of the network voltage/current term is

$$\frac{\partial P}{\partial x_i}=\chi P E_i P,$$

which gives

$$J_{ji}=\frac{\chi}{\beta}P_{ji}V_i-\alpha\delta_{ji}.$$

This expression is implemented directly in `models.py`. The windowed Jacobian includes the derivative of the window itself, and the $\sinh$ model uses the chain rule through the current nonlinearity.

## Reproduce locally

Requires Python 3.10+.

```bash
git clone https://github.com/OleBo/memristive-pseudospectral.git
cd memristive-pseudospectral
python -m pip install -e .
pytest -q
python scripts/run_exact.py
```

The main output is:

```text
results/exact/local_exact_summary.csv
results/exact/config.json
results/exact/A.npy
results/exact/omega.npy
```

The research presentation is generated from the same result set and is published through [GitHub Pages](https://OleBo.github.io/memristive-pseudospectral/).

## GitHub Actions

Three workflows are included:

- `CI`: installs the package and runs the analytic-Jacobian/projector tests.
- `Reproduce exact analysis`: executes the source-aligned calculation and uploads the numerical results as a workflow artifact.
- `Deploy research site`: publishes `docs/` plus the reproducibility results through [GitHub Pages](https://OleBo.github.io/memristive-pseudospectral/).

## Research status and limitations

This is an independent computational study, not an official reproduction by the original authors. Two source-level details require explicit care:

1. The paper specifies the scalar control parameter $s=\langle\Omega S\rangle/(\alpha\beta)$, but does not provide the full componentwise vector $S$ used for every numerical realization. We therefore construct $S$ from a pseudorandom Gaussian ensemble and report the single-realization result corresponding to the specified scalar value $s$.

2. The main Materials-and-Methods text and Supplementary Material D use different rank conventions for their separate numerical experiments. The repository preserves both statements rather than silently choosing one.

The project should therefore be read as a **source-aligned independent reproduction and extension**, rather than as a claim of bitwise replication of the authors' original simulation data.

## References

- F. Caravelli, F. C. Sheldon & F. L. Traversa, *Global minimization via classical tunneling assisted by collective force field formation*, Science Advances **7**, eabh1542 (2021). DOI: 10.1126/sciadv.abh1542
- F. Caravelli et al., *Self-organizing memristive networks as physical learning systems*, Nature Reviews Physics (2026), DOI: 10.1038/s42254-026-00978-x.
- L. N. Trefethen & M. Embree, *Spectra and Pseudospectra: The Behavior of Nonnormal Matrices and Operators*, Princeton University Press (2005).

## License

Code in this repository is released under the MIT License. The original scientific work remains the property of its authors and publisher; please cite it when using the model or its equations.

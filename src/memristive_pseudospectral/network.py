from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class ProjectorNetwork:
    """Random orthogonal projector used in Caravelli et al. (2021).

    Omega = A.T (A A.T)^(-1) A, with A having iid Uniform(0,1) entries.
    The source specifies N=200 and, in different experiments, rank 50 or 100.
    """
    omega: np.ndarray
    A: np.ndarray
    seed: int
    rank: int
    N: int

    @classmethod
    def from_source_ensemble(cls, N: int = 200, rank: int = 50, seed: int = 0) -> "ProjectorNetwork":
        rng = np.random.default_rng(seed)
        A = rng.uniform(0.0, 1.0, size=(rank, N))
        # Stable construction of the orthogonal projector onto row(A).
        G = A @ A.T
        omega = A.T @ np.linalg.solve(G, A)
        omega = 0.5 * (omega + omega.T)
        return cls(omega=omega, A=A, seed=seed, rank=rank, N=N)

def make_drive(omega: np.ndarray, s: float, alpha: float = 1.0, beta: float = 1.0) -> np.ndarray:
    """Construct a deterministic drive with <Omega S>/(alpha beta) == s.

    The paper specifies s=<Omega S>/(alpha beta), but does not publish the
    componentwise vector S used in every numerical run. We therefore use the
    gauge-fixed normalized all-ones drive and record this choice explicitly.
    """
    ones = np.ones(omega.shape[0])
    projected = omega @ ones
    mean_projected = projected.mean()
    if abs(mean_projected) < 1e-14:
        raise ValueError("<Omega 1> is numerically zero; cannot normalize the drive")
    return alpha * beta * s * ones / mean_projected

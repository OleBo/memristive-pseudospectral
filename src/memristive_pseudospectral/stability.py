from __future__ import annotations
import numpy as np
from scipy.linalg import expm, eigvals, svdvals

def local_metrics(J, t_grid=None):
    if t_grid is None:
        t_grid = np.linspace(0, 20, 81)
    ev = eigvals(J)
    spectral_abscissa = float(np.max(ev.real))
    H = 0.5*(J+J.T)
    numerical_abscissa = float(np.max(np.linalg.eigvalsh(H)))
    gains = np.array([svdvals(expm(J*t))[0] for t in t_grid])
    return {
        "spectral_abscissa": spectral_abscissa,
        "numerical_abscissa": numerical_abscissa,
        "max_expm_gain": float(gains.max()),
        "t_at_max_gain": float(t_grid[gains.argmax()]),
        "eigenvalues": ev,
        "gain_curve": gains,
    }

def kreiss_sample(J, real_grid=None, imag_grid=None):
    """Numerical lower-bound/sample estimate of the Kreiss quantity.

    K = sup_{Re z > 0} Re(z) ||(zI-J)^(-1)||_2.
    This is a grid approximation, not a rigorous supremum.
    """
    n = J.shape[0]
    if real_grid is None: real_grid = np.geomspace(1e-4, 1.0, 8)
    if imag_grid is None:
        scale = max(1.0, float(np.max(np.abs(np.linalg.eigvals(J)))))
        imag_grid = np.linspace(-2*scale, 2*scale, 11)
    I = np.eye(n)
    best = (0.0, None)
    for re in real_grid:
        for im in imag_grid:
            z = re + 1j*im
            try:
                r = np.linalg.solve(z*I-J, I)
                val = re * svdvals(r)[0]
            except np.linalg.LinAlgError:
                continue
            if val > best[0]: best = (float(val), complex(z))
    return {"K_sample": best[0], "z_at_K_sample": best[1]}

def finite_time_metrics(rhs, jac, x0, dt=0.01, t_end=2.0):
    """Integrate x and tangent propagator with RK4 on a uniform grid."""
    n = len(x0); steps = int(round(t_end/dt))
    x = x0.copy(); Phi = np.eye(n)
    rows=[]
    for k in range(steps+1):
        t=k*dt; J=jac(x)
        smax=svdvals(Phi)[0]
        ev=eigvals(J)
        rows.append((t, np.max(ev.real), np.max(np.linalg.eigvalsh((J+J.T)/2)), smax))
        if k==steps: break
        def F(xx, PP): return rhs(xx), jac(xx)@PP
        k1x,k1p=F(x,Phi)
        k2x,k2p=F(x+0.5*dt*k1x, Phi+0.5*dt*k1p)
        k3x,k3p=F(x+0.5*dt*k2x, Phi+0.5*dt*k2p)
        k4x,k4p=F(x+dt*k3x, Phi+dt*k3p)
        x += dt*(k1x+2*k2x+2*k3x+k4x)/6
        Phi += dt*(k1p+2*k2p+2*k3p+k4p)/6
    return np.asarray(rows), x

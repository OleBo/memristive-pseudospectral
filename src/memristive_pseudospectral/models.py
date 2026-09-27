from __future__ import annotations
import numpy as np


def _common(x, omega, S, alpha, beta, chi):
    n = len(x)
    I = np.eye(n)
    X = np.diag(x)
    A = I - chi * omega @ X
    P = np.linalg.solve(A, omega)
    V = P @ S
    return P, V


def standard_rhs(x, omega, S, alpha=1.0, beta=1.0, chi=0.9):
    P, V = _common(x, omega, S, alpha, beta, chi)
    return V / beta - alpha * x


def standard_jacobian(x, omega, S, alpha=1.0, beta=1.0, chi=0.9):
    P, V = _common(x, omega, S, alpha, beta, chi)
    # Exact analytic derivative: dP/dx_i = chi P E_i P.
    return (chi / beta) * (P * V[None, :]) - alpha * np.eye(len(x))


def sinh_rhs(x, omega, S, alpha=1.0, T=1.0, chi=0.9, Roff=1.0, ic=1.0):
    P, Icur = _common(x, omega, S, alpha, T, chi)
    return np.sinh(Icur / (ic * Roff)) / T - alpha * x


def sinh_jacobian(x, omega, S, alpha=1.0, T=1.0, chi=0.9, Roff=1.0, ic=1.0):
    P, Icur = _common(x, omega, S, alpha, T, chi)
    dI = chi * (P * Icur[None, :])
    scale = np.cosh(Icur / (ic * Roff)) / (T * ic * Roff)
    return scale[:, None] * dI - alpha * np.eye(len(x))


def window(x, current, kind: str, p: int):
    if kind == "joglekar":
        w = 1.0 - (2*x - 1.0)**(2*p)
        dw = -4*p*(2*x - 1.0)**(2*p - 1)
    elif kind == "prodromakis":
        q = (x - 0.5)**2 + 0.75
        w = 1.0 - q**p
        dw = -2*p*(x - 0.5)*q**(p - 1)
    elif kind == "biolek":
        # The source defines W_p(x,I)=1-(x-theta(-I))^(2p).
        theta = (current < 0).astype(float)
        z = x - theta
        w = 1.0 - z**(2*p)
        dw = -2*p*z**(2*p - 1)
    else:
        raise ValueError(f"unknown window: {kind}")
    return w, dw


def windowed_rhs(x, omega, S, alpha=1.0, beta=1.0, chi=0.9, kind="joglekar", p=1):
    P, V = _common(x, omega, S, alpha, beta, chi)
    f = V / beta - alpha*x
    w, _ = window(x, V, kind, p)
    return w*f


def windowed_jacobian(x, omega, S, alpha=1.0, beta=1.0, chi=0.9, kind="joglekar", p=1):
    P, V = _common(x, omega, S, alpha, beta, chi)
    f = V / beta - alpha*x
    Jf = (chi/beta)*(P * V[None,:]) - alpha*np.eye(len(x))
    w, dw = window(x, V, kind, p)
    return w[:,None]*Jf + np.diag(f*dw)


def theta_p(a, p=100.0):
    return 0.5 + 0.5*np.tanh(p*a)

def theta_prime(a, p=100.0):
    c=np.cosh(np.clip(p*a,-350,350))
    return 0.5*p/(c*c)

def source_boundary_rhs(x, omega, S, alpha=1.0, beta=1.0, chi=0.9, p=100.0):
    P,V=_common(x,omega,S,alpha,beta,chi)
    f=V/beta-alpha*x
    w=theta_p(-f,p)*theta_p(x,p)+theta_p(f,p)*theta_p(1-x,p)
    return w*f

def source_boundary_jacobian(x, omega, S, alpha=1.0, beta=1.0, chi=0.9, p=100.0):
    P,V=_common(x,omega,S,alpha,beta,chi)
    f=V/beta-alpha*x
    Jf=(chi/beta)*(P*V[None,:])-alpha*np.eye(len(x))
    hm=theta_p(-f,p); hp=theta_p(f,p); hx=theta_p(x,p); h1=theta_p(1-x,p)
    dHm=-theta_prime(-f,p)[:,None]*Jf
    dHp= theta_prime(f,p)[:,None]*Jf
    dHx=np.diag(theta_prime(x,p))
    dH1=-np.diag(theta_prime(1-x,p))
    dW=dHm*hx[:,None]+hm[:,None]*dHx+dHp*h1[:,None]+hp[:,None]*dH1
    w=hm*hx+hp*h1
    return w[:,None]*Jf + f[:,None]*dW

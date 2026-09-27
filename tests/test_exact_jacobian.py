import numpy as np
from memristive_pseudospectral.network import ProjectorNetwork, make_drive
from memristive_pseudospectral.models import standard_rhs, standard_jacobian

def test_exact_jacobian_matches_central_difference():
    net=ProjectorNetwork.from_source_ensemble(N=18,rank=7,seed=4)
    S=make_drive(net.omega,0.15)
    x=np.linspace(0.1,0.8,18)
    J=standard_jacobian(x,net.omega,S)
    eps=1e-6; Jfd=np.zeros_like(J)
    for i in range(18):
        e=np.zeros(18); e[i]=eps
        Jfd[:,i]=(standard_rhs(x+e,net.omega,S)-standard_rhs(x-e,net.omega,S))/(2*eps)
    assert np.max(np.abs(J-Jfd)) < 1e-7

def test_projector_is_symmetric_idempotent():
    O=ProjectorNetwork.from_source_ensemble(N=30,rank=12,seed=3).omega
    assert np.linalg.norm(O-O.T) < 1e-10
    assert np.linalg.norm(O@O-O) < 1e-9

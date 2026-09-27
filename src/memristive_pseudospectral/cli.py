from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import root
from .network import ProjectorNetwork, make_drive
from .models import standard_rhs, standard_jacobian, windowed_rhs, windowed_jacobian, sinh_rhs, sinh_jacobian, source_boundary_rhs, source_boundary_jacobian
from .stability import local_metrics, kreiss_sample


def run_exact(out: Path, N=200, rank=150, seed=2026, s=0.15):
    out.mkdir(parents=True, exist_ok=True)
    net=ProjectorNetwork.from_source_ensemble(N=N, rank=rank, seed=seed)
    S=make_drive(net.omega,s)
    models={
      'source_boundary_p100':(lambda x:source_boundary_rhs(x,net.omega,S,p=100), lambda x:source_boundary_jacobian(x,net.omega,S,p=100)),
      'standard':(lambda x:standard_rhs(x,net.omega,S), lambda x:standard_jacobian(x,net.omega,S)),
      'joglekar_p1':(lambda x:windowed_rhs(x,net.omega,S,kind='joglekar',p=1), lambda x:windowed_jacobian(x,net.omega,S,kind='joglekar',p=1)),
      'prodromakis_p1':(lambda x:windowed_rhs(x,net.omega,S,kind='prodromakis',p=1), lambda x:windowed_jacobian(x,net.omega,S,kind='prodromakis',p=1)),
      'biolek_p2':(lambda x:windowed_rhs(x,net.omega,S,kind='biolek',p=2), lambda x:windowed_jacobian(x,net.omega,S,kind='biolek',p=2)),
      'sinh':(lambda x:sinh_rhs(x,net.omega,S), lambda x:sinh_jacobian(x,net.omega,S)),
    }
    records=[]
    # Uniform mean-field starting point is used only as a root-search seed.
    xseed=np.full(N, 0.2 if s<0.2 else 0.25)
    for name,(rhs,jac) in models.items():
        sol=root(rhs,xseed,jac=jac,method='hybr',options={'maxfev':2000})
        if not sol.success or np.linalg.norm(rhs(sol.x),np.inf)>1e-7:
            # fall back to least-squares-like seed perturbation through a few starts
            found=False
            rng=np.random.default_rng(seed+17)
            for _ in range(8):
                trial=np.clip(xseed+0.03*rng.standard_normal(N),0,1)
                sol2=root(rhs,trial,jac=jac,method='hybr',options={'maxfev':3000})
                if sol2.success and np.linalg.norm(rhs(sol2.x),np.inf)<1e-7:
                    sol=sol2; found=True; break
            if not found:
                print(f'WARNING: root failed for {name}: {sol.message}; residual={np.linalg.norm(rhs(sol.x),np.inf):.3e}')
        x=sol.x; J=jac(x)
        m=local_metrics(J,np.linspace(0,8,17))
        k=kreiss_sample(J)
        records.append({'model':name,'success':bool(sol.success),'residual_inf':float(np.linalg.norm(rhs(x),np.inf)), 'spectral_abscissa':m['spectral_abscissa'],'numerical_abscissa':m['numerical_abscissa'],'max_expm_gain':m['max_expm_gain'],'t_at_max_gain':m['t_at_max_gain'],'K_sample':k['K_sample'],'K_z_real':k['z_at_K_sample'].real if k['z_at_K_sample'] else np.nan,'K_z_imag':k['z_at_K_sample'].imag if k['z_at_K_sample'] else np.nan,'mean_x':float(x.mean())})
    pd.DataFrame(records).to_csv(out/'local_exact_summary.csv',index=False)
    np.save(out/'omega.npy',net.omega)
    np.save(out/'A.npy',net.A)
    with open(out/'config.json','w') as f: json.dump({'N':N,'rank':rank,'seed':seed,'s':s,'alpha':1,'beta':1,'chi':0.9,'drive':'normalized_all_ones','source_note':'The main-paper text states span(Omega)=N-M with M=50, implying rank(Omega)=150 for N=200; Supplementary D separately states rank Nc=100 for its basin experiment. The componentwise S vector is not specified, so this study fixes the gauge by normalizing the all-ones drive to the published scalar s.'},f,indent=2)
    return pd.DataFrame(records)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',default='results/exact'); p.add_argument('--N',type=int,default=200); p.add_argument('--rank',type=int,default=50); p.add_argument('--seed',type=int,default=2026); p.add_argument('--s',type=float,default=0.15); a=p.parse_args(); print(run_exact(Path(a.out),a.N,a.rank,a.seed,a.s).to_string(index=False))
if __name__=='__main__': main()

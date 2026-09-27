from pathlib import Path
import sys
sys.path.insert(0,'src')
from memristive_pseudospectral.cli import run_exact
run_exact(Path('results/exact'), N=200, rank=150, seed=2026, s=0.15)

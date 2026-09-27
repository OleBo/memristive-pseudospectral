"""Pseudospectral analysis of self-organizing memristive networks."""
from .network import ProjectorNetwork, make_drive
from .models import standard_rhs, standard_jacobian, sinh_rhs, sinh_jacobian
from .stability import local_metrics, finite_time_metrics

__all__ = ["ProjectorNetwork", "make_drive", "standard_rhs", "standard_jacobian", "sinh_rhs", "sinh_jacobian", "local_metrics", "finite_time_metrics"]

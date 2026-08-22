"""Information-theoretic manipulation and mutual information modules."""

from src.info.mutual_information import (
    compute_mi_profile,
    compute_mutual_information,
    information_loss_ratio,
)
from src.info.truncation import inject_noise, sample_fourier_label, truncate_label

__all__ = [
    "truncate_label",
    "inject_noise",
    "sample_fourier_label",
    "compute_mutual_information",
    "information_loss_ratio",
    "compute_mi_profile",
]

"""Engines for quantum simulation, information processing, recovery, and statistics."""

from src.engines.dcp_engine import DCPEngine, DCPState, verify_dcp_state
from src.engines.info_engine import InformationEngine, InformationResult
from src.engines.qft_engine import QFTEngine, QFTResult, extract_fourier_info, verify_phases
from src.engines.recovery_engine import RecoveryEngine, RecoveryResult
from src.engines.stats_engine import AggregatedMetrics, StatisticsEngine

__all__ = [
    "DCPEngine",
    "DCPState",
    "verify_dcp_state",
    "QFTEngine",
    "QFTResult",
    "extract_fourier_info",
    "verify_phases",
    "InformationEngine",
    "InformationResult",
    "RecoveryEngine",
    "RecoveryResult",
    "StatisticsEngine",
    "AggregatedMetrics",
]

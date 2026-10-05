"""Engines for quantum simulation, information processing, recovery, and statistics."""

from src.engines.dcp_engine import DCPEngine, DCPState, DCPSample, generate_dcp_sample, verify_dcp_state
from src.engines.info_engine import InformationEngine, InformationResult
from src.engines.qft_engine import FourierSample, QFTEngine, QFTResult, extract_fourier_info, fourier_sample, verify_phases
from src.engines.recovery_engine import (
    DCPSecretRecoveryResult,
    RecoveryEngine,
    RecoveryResult,
    SecretBitRecoveryResult,
    recover_secret,
    recover_secret_bit,
)
from src.engines.subset_sum_engine import GroupAnalysis, GroupPartition, SubsetSumAnalysis, SubsetSumEngine
from src.engines.stats_engine import AggregatedMetrics, StatisticsEngine
from src.engines.simon_attack_engine import (
    BMeasurementResult,
    SimonAttackEngine,
    SimonSecretBitEstimate,
    SimonSecretEstimate,
)

__all__ = [
    "DCPEngine",
    "DCPState",
    "DCPSample",
    "verify_dcp_state",
    "generate_dcp_sample",
    "QFTEngine",
    "QFTResult",
    "FourierSample",
    "extract_fourier_info",
    "verify_phases",
    "fourier_sample",
    "InformationEngine",
    "InformationResult",
    "RecoveryEngine",
    "RecoveryResult",
    "SecretBitRecoveryResult",
    "DCPSecretRecoveryResult",
    "recover_secret_bit",
    "recover_secret",
    "SubsetSumEngine",
    "SubsetSumAnalysis",
    "GroupAnalysis",
    "GroupPartition",
    "SimonAttackEngine",
    "BMeasurementResult",
    "SimonSecretBitEstimate",
    "SimonSecretEstimate",
    "StatisticsEngine",
    "AggregatedMetrics",
]

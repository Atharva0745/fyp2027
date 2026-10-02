"""Toy Bai-style modulus-halving engine for EDCP."""

from __future__ import annotations

from typing import Any
import numpy as np
from src.engines.edcp_engine import EDCPEngine, get_standard_chi
from src.engines.qft_engine import QFTEngine
from src.engines.info_engine import InformationEngine


class ModulusHalvingEngine:
    """Implements a toy version of Bai et al.'s modulus-halving approach."""

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self.rng = rng if rng is not None else np.random.default_rng()
        self.edcp_engine = EDCPEngine()
        self.qft_engine = QFTEngine()
        self.info_engine = InformationEngine(rng=self.rng)

    def run_iteration(self, N: int, s: int, chi: dict[int, complex], k: int | None = None) -> dict[str, Any]:
        """Run a single modulus-halving iteration.
        
        Args:
            N: Current modulus (must be even).
            s: Current secret.
            chi: Error distribution.
            k: Bits to retain from Fourier labels (if truncated).
            
        Returns:
            Dictionary with next modulus, next secret, and observation data.
        """
        if N % 2 != 0:
            raise ValueError(f"Modulus N={N} must be even for halving.")
            
        x = int(self.rng.integers(0, N))
        
        # 1. Construct EDCP state
        state = self.edcp_engine.create_state(N, s, x, chi)
        
        # 2. QFT
        qft_res = self.qft_engine.transform(state)
        
        # 3. Sample Fourier label
        info_res = self.info_engine.process(qft_res, k=k, rng=self.rng)
        
        # 4. Modulus halving logic (toy model)
        # Assume we correctly guess s mod 2. 
        # In this toy pipeline, we just demonstrate that N halves and s halves.
        s_lsb = s % 2
        s_next = s // 2
        N_next = N // 2
        
        return {
            "N_prev": N,
            "N_next": N_next,
            "s_prev": s,
            "s_next": s_next,
            "s_lsb": s_lsb,
            "y_observed": info_res.Y_full,
            "y_truncated": info_res.Y_truncated
        }
        
    def run_pipeline(self, N_start: int, s_start: int, chi_name: str = "2-term", iterations: int = 1) -> list[dict[str, Any]]:
        """Run a multi-iteration modulus halving pipeline.
        
        Args:
            N_start: Starting modulus (must be divisible by 2^iterations).
            s_start: Starting secret.
            chi_name: Name of standard chi distribution to use.
            iterations: Number of halving iterations to perform.
            
        Returns:
            List of iteration results.
        """
        chi = get_standard_chi(chi_name)
        
        N_current = N_start
        s_current = s_start
        results = []
        
        for i in range(iterations):
            if N_current % 2 != 0:
                break
                
            iter_res = self.run_iteration(N_current, s_current, chi)
            iter_res["iteration"] = i + 1
            results.append(iter_res)
            
            N_current = iter_res["N_next"]
            s_current = iter_res["s_next"]
            
        return results

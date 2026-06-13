"""
Unit test for FPGA placement optimizer (no Vivado DCP required).

Creates a minimal synthetic coupling matrix / site coords and runs the
optimiser.  The test passes if optimisation completes without error.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from fem_placer import FPGAPlacementOptimizer
from fem_placer.logger import SET_LEVEL

SET_LEVEL('WARNING')

NUM_TRIALS = 2
NUM_STEPS = 50
DEV = 'cpu'


def test_optimizer_synthetic():
    """Optimiser must run on synthetic data without error."""
    N, M = 20, 10  # 20 instances, 10 sites

    coupling = torch.randn(N, N)
    coupling = (coupling + coupling.T) / 2
    site_coords = torch.rand(M, 2) * 10

    optimizer = FPGAPlacementOptimizer(
        regions=['logic'],
        region_sizes={'logic': (N, M)},
        region_coupling={'logic': {'logic': coupling}},
        region_site_coords={'logic': site_coords},
        constraint_coeffs=[30.0],
        h_factors=[0.01],
        num_trials=NUM_TRIALS, num_steps=NUM_STEPS, dev=DEV,
        betamin=0.01, betamax=0.5, anneal='exp',
        optimizer="adam", learning_rate=0.1, seed=1,
        dtype=torch.float32, manual_grad=False, distance_metric='manhattan',
    )

    t0 = time.time()
    config, result = optimizer.optimize()
    elapsed = time.time() - t0

    hpwl = result.min().item()

    print(f"\nSynthetic test: {N} inst, {M} sites, {NUM_STEPS} steps")
    print(f"  Time: {elapsed:.3f}s, HPWL: {hpwl:.4f}")

    assert elapsed > 0
    assert config is not None
    assert result is not None


if __name__ == "__main__":
    test_optimizer_synthetic()
    print("\nPASSED")
    sys.exit(0)

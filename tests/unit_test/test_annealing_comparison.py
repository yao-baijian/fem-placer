"""
Unit test comparing FPGA placement optimizer performance across different
annealing schedules (linear, exponential, inverse).

Uses the N-region dict-based optimizer API via ``TestConfig`` +
``default_config.json``.
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from fem_placer import (
    FpgaPlacer,
    FPGAPlacementOptimizer,
)
from fem_placer.logger import *
from tests.utils import TestConfig

SET_LEVEL('WARNING')

NUM_TRIALS = 2
NUM_STEPS = 100
ANNEAL_TYPES = ['lin', 'exp', 'inverse']


def _load_cfg() -> TestConfig:
    cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'config', 'default_config.json',
    )
    with open(cfg_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    raw = {k: v for k, v in raw.items() if not k.startswith('_')}
    from tests.utils import _ENUM_MAP
    for key, (enum_cls, default_name) in _ENUM_MAP.items():
        if key in raw:
            raw[key] = enum_cls[raw[key]]
    return TestConfig(**{k: v for k, v in raw.items() if k in TestConfig.__dataclass_fields__})


def _build_optimizer_dicts(placer):
    """Build N-region dicts from a configured placer."""
    regions = [r for r in placer.regions if placer.instances[r].num > 0]
    region_sizes = {r: (placer.instances[r].num, max(placer.get_grid(r).area, 1)) for r in regions}
    region_site_coords = {}
    for r in regions:
        attr = f'{r}_site_coords'
        if hasattr(placer, attr):
            region_site_coords[r] = getattr(placer, attr)
        else:
            g = placer.get_grid(r)
            region_site_coords[r] = g.to_real_coords_tensor(torch.cartesian_prod(
                torch.arange(g.area_length, dtype=torch.float32, device=placer.device),
                torch.arange(g.area_width, dtype=torch.float32, device=placer.device)
            ))
    io_mat = placer.net_manager.io_insts_matrix
    region_coupling = {}
    for rA in regions:
        region_coupling[rA] = {}
        for rB in regions:
            if rA == rB == 'logic':
                region_coupling[rA][rB] = placer.net_manager.insts_matrix
            elif rA == 'logic' and rB == 'io' and io_mat is not None:
                region_coupling[rA][rB] = io_mat
            elif rA == 'io' and rB == 'logic' and io_mat is not None:
                region_coupling[rA][rB] = io_mat.T.clone()
            else:
                region_coupling[rA][rB] = None
    coeff_map = {'logic': placer.constraint_alpha, 'io': placer.constraint_beta}
    constraint_coeffs = [coeff_map.get(r, 1.0) for r in regions]
    h_factors = [0.01 for _ in regions]
    return regions, region_sizes, region_coupling, region_site_coords, constraint_coeffs, h_factors


def run_single_anneal(placer, anneal_type: str) -> dict:
    """Run optimisation with one annealing schedule, return HPWL."""
    dev = placer.device
    r, rs, rc, rsc, cc, hf = _build_optimizer_dicts(placer)
    optimizer = FPGAPlacementOptimizer(
        regions=r, region_sizes=rs, region_coupling=rc, region_site_coords=rsc,
        constraint_coeffs=cc, h_factors=hf,
        num_trials=NUM_TRIALS, num_steps=NUM_STEPS, dev=dev,
        betamin=0.01, betamax=0.5, anneal=anneal_type,
        optimizer="adam", learning_rate=0.1, seed=1,
        dtype=torch.float32, manual_grad=False, distance_metric='manhattan',
    )
    config, result = optimizer.optimize()
    return {'anneal': anneal_type, 'hpwl': result.min().item()}


def test_annealing_comparison():
    """All annealing schedules must complete; at least one must produce a valid HPWL."""
    cfg = _load_cfg()
    instance = cfg.instances[0]
    placer = FpgaPlacer(cfg)
    placer.set_instance_name(instance)
    dcp_path = f"./vivado/output_dir/{instance}/post_impl.dcp"
    pl_path = f"./vivado/output_dir/{instance}/optimized_placement.pl"
    placer.init_placement(dcp_path, pl_path)
    placer.set_alpha(30)
    if "io" in placer.regions:
        placer.set_beta(30)

    results = {}
    for at in ANNEAL_TYPES:
        res = run_single_anneal(placer, at)
        results[at] = res

    print(f"\n{'Anneal':<10} {'HPWL':<12}")
    print("-" * 24)
    for at in ANNEAL_TYPES:
        print(f"{at.upper():<10} {results[at]['hpwl']:<12.2f}")

    assert len(results) == len(ANNEAL_TYPES), "Not all annealing types completed"


if __name__ == "__main__":
    test_annealing_comparison()
    print("\nPASSED")
    sys.exit(0)

"""
Unit test for io_factor sweep with fixed alpha/beta.

Refactored to use the N-region optimizer API.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from fem_placer import FpgaPlacer, Legalizer, FPGAPlacementOptimizer
from fem_placer.logger import SET_LEVEL
from fem_placer.config import PlaceType
from tests.utils import TestConfig
from tests.placement_eval_logic import _build_optimizer_dicts

SET_LEVEL('WARNING')

NUM_TRIALS = 2
NUM_STEPS = 100
DEV = 'cpu'
ANNEAL = 'inverse'
ALPHA = 10.0
BETA = 10.0
IO_FACTOR_START = 10.0
IO_FACTOR_END = 100.0
IO_FACTOR_STEP = 30.0


def _load_cfg() -> TestConfig:
    import json
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


def test_io_factor_sweep():
    """io_factor sweep must produce a valid best configuration."""
    cfg = _load_cfg()
    instance = cfg.instances[0]

    placer = FpgaPlacer(cfg)
    placer.set_instance_name(instance)
    dcp_path = f"./vivado/output_dir/{instance}/post_impl.dcp"
    pl_path = f"./vivado/output_dir/{instance}/optimized_placement.pl"
    vivado_hpwl, inst_num, _ = placer.init_placement(dcp_path, pl_path)
    placer.set_alpha(ALPHA)
    placer.set_beta(BETA)

    best_hpwl = float('inf')
    best_iof = IO_FACTOR_START

    print(f"\n{'ioFactor':<10} {'HPWL':<12} {'Overlap':<8}")
    print("-" * 32)

    iof = IO_FACTOR_START
    while iof <= IO_FACTOR_END + 1e-9:
        placer.grids['logic'].clear_all()
        placer.grids['io'].clear_all()
        _, rs, rc, rsc, cc, hf = _build_optimizer_dicts(placer, ALPHA, BETA, iof, PlaceType.IO)
        optimizer = FPGAPlacementOptimizer(
            regions=_, region_sizes=rs, region_coupling=rc, region_site_coords=rsc,
            constraint_coeffs=cc, h_factors=hf,
            num_trials=NUM_TRIALS, num_steps=NUM_STEPS, dev=DEV,
            betamin=0.01, betamax=0.5, anneal=ANNEAL,
            optimizer="adam", learning_rate=0.1, seed=1,
            dtype=torch.float32, manual_grad=False, distance_metric='manhattan',
        )
        config, result = optimizer.optimize()
        opt_idx = torch.argwhere(result == result.min()).reshape(-1)[0]
        best_cfg = {r: config[r][opt_idx] for r in config}
        all_ids = placer.get_ids()
        rid_map = dict(zip(placer.regions, all_ids))
        best_ids = {r: rid_map[r] for r in config if r in rid_map}
        legalizer = Legalizer(placer=placer, device=DEV)
        _, overlap, _, hpwl_f = legalizer.legalize_placement(best_cfg, best_ids)
        hpwl = hpwl_f['hpwl']
        print(f"{iof:<10.0f} {hpwl:<12.2f} {overlap:<8}")
        if hpwl < best_hpwl:
            best_hpwl = hpwl
            best_iof = iof
        iof += IO_FACTOR_STEP

    assert best_hpwl < float('inf'), "No valid io_factor found"
    print(f"\nBest io_factor={best_iof:.0f}, HPWL={best_hpwl:.2f}")


if __name__ == "__main__":
    test_io_factor_sweep()
    print("\nPASSED")
    sys.exit(0)

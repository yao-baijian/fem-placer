"""
Unit test for alpha grid-search training (logic-only, CENTERED placement).

Refactored to use the N-region optimizer API.  The alpha parameter
maps to the constraint coefficient for the ``logic`` region.

NOTE: The ML training/prediction flow (``ml/train.py``, ``ml/predict.py``)
currently uses scalar alpha/beta targets.  For full N-region support,
the ML pipeline should be extended to predict one coefficient per region.
"""

import sys
import os
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from fem_placer import FpgaPlacer
from fem_placer.logger import SET_LEVEL
from fem_placer.config import PlaceType
from tests.placement_eval_logic import run_logic_placement, _build_optimizer_dicts
from tests.utils import TestConfig
from ml.dataset import extract_features_from_placer, append_row, clear_dataset

SET_LEVEL('WARNING')

NUM_TRIALS = 2
NUM_STEPS = 100
DEV = 'cpu'
ANNEAL = 'inverse'


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


def test_alpha_sweep():
    """Coarse alpha sweep must produce a valid best configuration."""
    cfg = _load_cfg()
    instance = cfg.instances[0]
    place_type = PlaceType.CENTERED

    placer = FpgaPlacer(cfg)
    placer.set_instance_name(instance)
    dcp_path = f"./vivado/output_dir/{instance}/post_impl.dcp"
    pl_path = f"./vivado/output_dir/{instance}/optimized_placement.pl"
    vivado_hpwl, inst_num, net_num = placer.init_placement(dcp_path, pl_path)

    coarse_start, coarse_end, coarse_step = 0, 50, 10
    overlap_allowed_max = 0.08

    best_hpwl = float('inf')
    best_alpha = 0.0

    print(f"\n{'Alpha':<8} {'HPWL':<12} {'Overlap':<8} {'In Range':<10}")
    print("-" * 42)

    for used_alpha in range(coarse_start, coarse_end, coarse_step):
        res = run_logic_placement(
            placer, alpha=float(used_alpha), beta=0.0,
            num_trials=NUM_TRIALS, num_steps=NUM_STEPS,
            dev=DEV, anneal=ANNEAL, place_type=place_type,
        )
        overlap = res['overlap']
        hpwl = res['fem_hpwl_final']['hpwl_no_io']
        overlap_pct = overlap / max(inst_num['logic_inst_num'], 1)
        in_range = overlap_pct <= overlap_allowed_max
        print(f"{used_alpha:<8} {hpwl:<12.2f} {overlap:<8} {str(in_range):<10}")

        if in_range and hpwl < best_hpwl:
            best_hpwl = hpwl
            best_alpha = used_alpha

    assert best_alpha > 0 or best_hpwl < float('inf'), "No valid alpha found"
    print(f"\nBest alpha={best_alpha:.1f}, HPWL={best_hpwl:.2f}")


if __name__ == "__main__":
    test_alpha_sweep()
    print("\nPASSED")
    sys.exit(0)

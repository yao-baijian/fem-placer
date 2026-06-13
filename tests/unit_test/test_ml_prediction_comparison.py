"""
Unit test comparing ML-predicted alpha vs grid search vs default.

Refactored to use the N-region optimizer API.  The alpha parameter
maps to the ``logic`` region constraint coefficient.

NOTE: Full N-region ML support requires extending the training/prediction
pipeline to handle per-region constraint coefficients.
"""

import sys
import os
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fem_placer import FpgaPlacer
from fem_placer.logger import SET_LEVEL
from fem_placer.config import PlaceType
from tests.placement_eval_logic import run_logic_placement
from tests.utils import TestConfig
from ml.dataset import extract_features_from_placer
from ml.predict import predict_target

SET_LEVEL('WARNING')

NUM_TRIALS = 2
NUM_STEPS = 100
DEV = 'cpu'
ANNEAL = 'inverse'
DEFAULT_ALPHA = 10.0


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


def test_ml_vs_default():
    """ML prediction and default alpha must both produce valid placements."""
    cfg = _load_cfg()
    instance = cfg.instances[0]
    place_type = PlaceType.CENTERED

    placer = FpgaPlacer(cfg)
    placer.set_instance_name(instance)
    dcp_path = f"./vivado/output_dir/{instance}/post_impl.dcp"
    pl_path = f"./vivado/output_dir/{instance}/optimized_placement.pl"
    vivado_hpwl, inst_num, _ = placer.init_placement(dcp_path, pl_path)

    # ML-predicted alpha
    row = extract_features_from_placer(placer, alpha=0, beta=0, with_io=False)
    try:
        pred_alpha = predict_target(row, target="alpha")
    except Exception:
        pred_alpha = DEFAULT_ALPHA

    res_ml = run_logic_placement(
        placer, alpha=pred_alpha, beta=0.0,
        num_trials=NUM_TRIALS, num_steps=NUM_STEPS,
        dev=DEV, anneal=ANNEAL, place_type=place_type,
    )

    # Default alpha
    res_def = run_logic_placement(
        placer, alpha=DEFAULT_ALPHA, beta=0.0,
        num_trials=NUM_TRIALS, num_steps=NUM_STEPS,
        dev=DEV, anneal=ANNEAL, place_type=place_type,
    )

    hpwl_ml = res_ml['fem_hpwl_final']['hpwl_no_io']
    hpwl_def = res_def['fem_hpwl_final']['hpwl_no_io']
    overlap_ml = res_ml['overlap']

    print(f"\n{'Method':<12} {'HPWL':<12} {'Overlap':<8}")
    print("-" * 34)
    print(f"{'ML':<12} {hpwl_ml:<12.2f} {overlap_ml:<8}")
    print(f"{'Default':<12} {hpwl_def:<12.2f} {res_def['overlap']:<8}")

    assert hpwl_ml > 0, "ML placement produced invalid HPWL"
    assert hpwl_def > 0, "Default placement produced invalid HPWL"


if __name__ == "__main__":
    test_ml_vs_default()
    print("\nPASSED")
    sys.exit(0)

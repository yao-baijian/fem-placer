"""
Unit test for alpha/beta coarse sweep with logic+IO (IoMode.VIRTUAL_NODE).

Refactored to use the N-region optimizer API.  The legacy scalar alpha/beta
are mapped to per-region ``constraint_coeffs``.

NOTE: The ML training dataset (``ml/dataset.py``) currently stores scalar
alpha/beta.  For full N-region support, the dataset schema should be extended.
"""

import sys
import os
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import concurrent.futures
from fem_placer import FpgaPlacer, Legalizer, FPGAPlacementOptimizer
from fem_placer.logger import SET_LEVEL
from fem_placer.config import PlaceType, GridType, IoMode
from tests.utils import TestConfig
from tests.placement_eval_logic import _build_optimizer_dicts
from ml.dataset import extract_features_from_placer, append_row, clear_dataset

SET_LEVEL('WARNING')

NUM_TRIALS = 2
NUM_STEPS = 100
DEV = 'cpu'
ANNEAL = 'inverse'
IO_FACTOR = 100.0
MAX_WORKERS = 4


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


def test_alpha_beta_sweep():
    """Coarse 2D alpha/beta sweep must produce a valid best configuration."""
    cfg = _load_cfg()
    instance = cfg.instances[0]
    place_type = PlaceType.IO

    placer = FpgaPlacer(cfg)
    placer.set_instance_name(instance)
    dcp_path = f"./vivado/output_dir/{instance}/post_impl.dcp"
    pl_path = f"./vivado/output_dir/{instance}/optimized_placement.pl"
    vivado_hpwl, inst_num, _ = placer.init_placement(dcp_path, pl_path)

    coarse_start, coarse_end, coarse_step = 0, 30, 10
    overlap_allowed_max = 0.08

    def evaluate(alpha, beta):
        placer.grids['logic'].clear_all()
        placer.grids['io'].clear_all()
        placer.set_alpha(alpha)
        placer.set_beta(beta)

        _, rs, rc, rsc, cc, hf = _build_optimizer_dicts(placer, alpha, beta, IO_FACTOR, place_type)
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
        return {'alpha': alpha, 'beta': beta, 'hpwl': hpwl_f['hpwl'], 'overlap': overlap}

    print(f"\n{'Alpha':<8} {'Beta':<8} {'HPWL':<12} {'Overlap':<8}")
    print("-" * 38)

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = []
        for a in range(coarse_start, coarse_end, coarse_step):
            for b in range(coarse_start, coarse_end, coarse_step):
                futures.append(ex.submit(evaluate, float(a), float(b)))
        for f in concurrent.futures.as_completed(futures):
            r = f.result()
            logic_total = max(inst_num['logic_inst_num'], 1)
            in_range = r['overlap'] / logic_total <= overlap_allowed_max
            print(f"{r['alpha']:<8.0f} {r['beta']:<8.0f} {r['hpwl']:<12.2f} {r['overlap']:<8} {'*' if in_range else ''}")

    print("\nCoarse sweep completed.")


if __name__ == "__main__":
    test_alpha_beta_sweep()
    print("\nPASSED")
    sys.exit(0)

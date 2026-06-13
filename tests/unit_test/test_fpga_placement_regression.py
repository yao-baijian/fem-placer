"""
Regression test for FPGA placement optimizer.

Runs the full placement pipeline on the s15850 benchmark with
logic+IO regions and verifies that the optimised HPWL matches
an expected value (deterministic given seed=1).

Usage::

    python -m pytest tests/unit_test/test_fpga_placement_regression.py -v

    # or run directly:
    python tests/unit_test/test_fpga_placement_regression.py
"""

import sys
import os
import json

# Point to project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import warnings
warnings.filterwarnings("ignore", message="Trying to unpickle estimator.*")

from fem_placer import FpgaPlacer, FPGAPlacementOptimizer
from tests.utils import TestConfig

# ── Expected HPWL after optimisation (seed=1, 500 steps, s15850) ──────────
# These values are sensitive to algorithm changes; update when the optimiser
# logic is deliberately modified.
EXPECTED_HPWL_AFTER_OPT = 6303.0   # from legalizer global-opt stage
TOLERANCE = 1.0                     # accept ±1 unit HPWL drift


def _make_regression_config() -> TestConfig:
    """Load the default regression config (s15850, logic+io)."""
    cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'config', 'default_config.json',
    )
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(
            f"Default config not found at {cfg_path}. "
            f"Create it from fem_placer/config.json"
        )
    with open(cfg_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    raw = {k: v for k, v in raw.items() if not k.startswith('_')}

    from tests.utils import _ENUM_MAP
    for key, (enum_cls, default_name) in _ENUM_MAP.items():
        if key in raw:
            raw[key] = enum_cls[raw[key]]

    return TestConfig(**{k: v for k, v in raw.items() if k in TestConfig.__dataclass_fields__})


def run_placement_pipeline(cfg: TestConfig) -> float:
    """
    Run the FEM placement pipeline and return the final HPWL
    (after legalizer global optimisation, with IO nets).
    """
    instance = cfg.instances[0]

    placer = FpgaPlacer(cfg)
    placer.set_instance_name(instance)

    dcp_path = f"./vivado/output_dir/{instance}/post_impl.dcp"
    pl_path = f"./vivado/output_dir/{instance}/optimized_placement.pl"
    vivado_hpwl, inst_num, net_num = placer.init_placement(dcp_path, pl_path)

    alpha_val = 30
    placer.set_alpha(alpha_val)
    if "io" in placer.regions:
        placer.set_beta(30)

    # Build N-region dicts
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
                torch.arange(g.area_length, dtype=torch.float32, device=cfg.dev),
                torch.arange(g.area_width, dtype=torch.float32, device=cfg.dev)
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
    h_factors = [cfg.io_factor * cfg.h_factor if r == 'io' else cfg.h_factor for r in regions]

    optimizer = FPGAPlacementOptimizer(
        regions=regions,
        region_sizes=region_sizes,
        region_coupling=region_coupling,
        region_site_coords=region_site_coords,
        constraint_coeffs=constraint_coeffs,
        h_factors=h_factors,
        num_trials=cfg.num_trials,
        num_steps=cfg.num_steps,
        dev=cfg.dev,
        betamin=cfg.betamin,
        betamax=cfg.betamax,
        anneal=cfg.anneal,
        optimizer="adam",
        learning_rate=cfg.learning_rate,
        seed=cfg.seed,
        dtype=torch.float32,
        manual_grad=cfg.manual_grad,
        distance_metric='manhattan',
    )

    # Optimise
    config, result = optimizer.optimize()
    optimal_inds = torch.argwhere(result == result.min()).reshape(-1)

    # Build region coords/ids dicts
    best_config = {r: config[r][optimal_inds[0]] for r in config}
    all_ids = placer.get_ids()
    region_id_map = dict(zip(placer.regions, all_ids))
    best_ids = {r: region_id_map[r] for r in config if r in region_id_map}

    # Legalize
    from fem_placer import Legalizer
    legalizer = Legalizer(placer=placer, device=cfg.dev)
    legalized, overlap, hpwl_before, hpwl_after = legalizer.legalize_placement(
        best_config, best_ids
    )

    # Return the 'hpwl' value (includes IO nets)
    return hpwl_after.get('hpwl', 0.0)


def test_placement_hpwl_regression():
    """The optimised HPWL for s15850 must match the expected value."""
    cfg = _make_regression_config()
    hpwl = run_placement_pipeline(cfg)
    assert abs(hpwl - EXPECTED_HPWL_AFTER_OPT) <= TOLERANCE, (
        f"HPWL {hpwl:.2f} differs from expected {EXPECTED_HPWL_AFTER_OPT:.2f} "
        f"(tolerance ±{TOLERANCE})"
    )


if __name__ == "__main__":
    cfg = _make_regression_config()
    hpwl = run_placement_pipeline(cfg)
    print(f"HPWL after optimisation: {hpwl:.2f}  (expected {EXPECTED_HPWL_AFTER_OPT:.2f})")
    ok = abs(hpwl - EXPECTED_HPWL_AFTER_OPT) <= TOLERANCE
    print("PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)

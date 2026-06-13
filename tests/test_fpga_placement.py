
"""
Test script for FPGA placement optimizer using QUBO formulation.

Usage::

    cfg = TestConfig.load()
    placer = FpgaPlacer(cfg)
    placer.set_instance_name('s15850')
    ...
"""

import sys
sys.path.insert(0, '.')

import time
import torch
import warnings
warnings.filterwarnings("ignore", message="Trying to unpickle estimator.*")

from fem_placer import (
    FpgaPlacer,
    PlacementDrawer,
    Router,
    Legalizer,
    FPGAPlacementOptimizer,
)
from fem_placer.logger import *
from tests.utils import (
    TestConfig,
    get_vivado_place_times,
    run_timing_analysis,
    RESULT_HEADER,
    format_result_row,
)
from ml.dataset import extract_features_from_placer
from ml.predict import predict_alpha
import json
import os

cfg = TestConfig.load()
SET_LEVEL("INFO")

# Setup per-instance log file
def _setup_log(instance: str):
    log_dir = f'result/{instance}'
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'run.log')
    Logger.get_instance().set_log_file(log_path)
    INFO(f"Config: {json.dumps({k: str(v) if not isinstance(v, (str, int, float, bool, list)) else v for k, v in cfg.__dict__.items() if not k.startswith('_')}, indent=2)}")

vivado_place_times = get_vivado_place_times()
print(RESULT_HEADER)

for instance in cfg.instances:
    _setup_log(instance)
    placer = FpgaPlacer(cfg)
    placer.set_instance_name(instance)

    dcp_path = f"./vivado/output_dir/{instance}/post_impl.dcp"
    pl_path = f"./vivado/output_dir/{instance}/optimized_placement.pl"
    vivado_hpwl, inst_num, net_num = placer.init_placement(dcp_path, pl_path)

    net_ratio = f"{net_num['logic_net_num']}/{net_num['total_net_num']}"
    drawer = PlacementDrawer(placer=placer)

    # row = extract_features_from_placer(placer, alpha=0, beta=0, with_io=False)
    # alpha_val = predict_alpha(row) * 0.001
    alpha_val = 30
    placer.set_alpha(alpha_val)
    if "io" in placer.regions:
        placer.set_beta(30)

    # Build N-region dicts for the optimizer
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

    t0 = time.time()
    config, result = optimizer.optimize()
    optimize_time = time.time() - t0

    optimal_inds = torch.argwhere(result == result.min()).reshape(-1)
    legalizer = Legalizer(placer=placer, device=cfg.dev)
    router = Router(placer=placer)
    all_ids = placer.get_ids()
    region_id_map = dict(zip(placer.regions, all_ids))

    # Pick best trial and build region coords/ids dicts
    best_config = {r: config[r][optimal_inds[0]] for r in config}
    best_ids = {r: region_id_map[r] for r in config if r in region_id_map}

    # Legalize — returns dict {r: legalized_coords}
    legalized, overlap, hpwl_i, hpwl_f = legalizer.legalize_placement(best_config, best_ids)

    # Route — concatenate all region coords
    all_coords = torch.cat([legalized[r] for r in legalized], dim=0)
    routes = router.route_connections(placer.net_manager.insts_matrix, all_coords)

    vivado_time_str = str(vivado_place_times.get(instance, "N/A"))

    # Timing — extract logic/io for the legacy timer interface
    timing_result = run_timing_analysis(
        placer=placer,
        placement_legalized=(legalized.get('logic'), legalized.get('io')),
        clock_period_ns=cfg.clock_period_ns,
        use_rapidwright=True,
        instance_name=instance,
    )

    print(format_result_row(
        instance=instance, inst_num=inst_num, net_ratio=net_ratio,
        overlap=overlap,
        used_alpha=alpha_val, used_beta=0.0,
        fem_hpwl_initial=hpwl_i, fem_hpwl_final=hpwl_f,
        vivado_hpwl=vivado_hpwl,
        optimize_time=optimize_time, vivado_time_str=vivado_time_str,
        wns_ns=timing_result.wns * 1e9, fmax_mhz=timing_result.fmax,
        include_io=('io' in config),
    ))
    print()
    print(timing_result.format_report())
    print()

    if cfg.draw_loss_function:
        drawer.plot_fpga_placement_loss(f"result/{instance}/hpwl_loss.png")
    if cfg.draw_evolution:
        drawer.draw_multi_step_placement(f"result/{instance}/placement_evolution.png")
    if cfg.draw_final_placement:
        io_c = legalized[1] if include_io else None
        drawer.draw_place_and_route(legalized[0], routes, io_c, include_io, 1000)

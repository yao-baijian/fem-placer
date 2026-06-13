import time
from typing import Dict, Any

import torch

from fem_placer import Legalizer, FPGAPlacementOptimizer
from fem_placer.config import PlaceType


def _build_optimizer_dicts(fpga_placer, alpha, beta, io_factor, place_type):
    """Build N-region dicts for the optimizer from legacy scalar alpha/beta/io_factor."""
    regions = [r for r in fpga_placer.regions if fpga_placer.instances[r].num > 0]
    region_sizes = {r: (fpga_placer.instances[r].num, max(fpga_placer.get_grid(r).area, 1)) for r in regions}
    region_site_coords = {}
    for r in regions:
        attr = f'{r}_site_coords'
        if hasattr(fpga_placer, attr):
            region_site_coords[r] = getattr(fpga_placer, attr)
        else:
            g = fpga_placer.get_grid(r)
            region_site_coords[r] = g.to_real_coords_tensor(torch.cartesian_prod(
                torch.arange(g.area_length, dtype=torch.float32, device=fpga_placer.device),
                torch.arange(g.area_width, dtype=torch.float32, device=fpga_placer.device)
            ))
    io_mat = fpga_placer.net_manager.io_insts_matrix
    region_coupling = {}
    for rA in regions:
        region_coupling[rA] = {}
        for rB in regions:
            if rA == rB == 'logic':
                region_coupling[rA][rB] = fpga_placer.net_manager.insts_matrix
            elif rA == 'logic' and rB == 'io' and io_mat is not None:
                region_coupling[rA][rB] = io_mat
            elif rA == 'io' and rB == 'logic' and io_mat is not None:
                region_coupling[rA][rB] = io_mat.T.clone()
            else:
                region_coupling[rA][rB] = None

    # constraint_coeffs and h_factors as lists positionally mapped to regions
    coeff_map = {'logic': alpha, 'io': beta}
    constraint_coeffs = [coeff_map.get(r, 1.0) for r in regions]
    h_factors = [io_factor * 0.01 if r == 'io' else 0.01 for r in regions]

    return regions, region_sizes, region_coupling, region_site_coords, constraint_coeffs, h_factors


def run_logic_placement(
    fpga_placer,
    alpha: float,
    beta: float = 0.0,
    io_factor: float = 1.0,
    num_trials: int = 5,
    num_steps: int = 200,
    dev: str = "cpu",
    manual_grad: bool = False,
    anneal: str = "inverse",
    place_type: PlaceType = PlaceType.CENTERED,
) -> Dict[str, Any]:
    """Run a single logic placement optimization + legalization.

    Uses the new N-region optimizer API internally. Legacy scalar
    alpha/beta params are mapped to per-region ``constraint_coeffs`` lists.

    Returns a dict with placement, overlap, HPWL dictionaries, and runtime.
    """
    # Clear grid state before each run
    for r in fpga_placer.regions:
        fpga_placer.grids[r].clear_all()

    fpga_placer.set_alpha(alpha)
    if place_type == PlaceType.IO:
        fpga_placer.set_beta(beta)

    regions, region_sizes, region_coupling, region_site_coords, constraint_coeffs, h_factors = \
        _build_optimizer_dicts(fpga_placer, alpha, beta, io_factor, place_type)

    optimizer = FPGAPlacementOptimizer(
        regions=regions,
        region_sizes=region_sizes,
        region_coupling=region_coupling,
        region_site_coords=region_site_coords,
        constraint_coeffs=constraint_coeffs,
        h_factors=h_factors,
        num_trials=num_trials,
        num_steps=num_steps,
        dev=dev,
        betamin=0.01,
        betamax=0.5,
        anneal=anneal,
        optimizer="adam",
        learning_rate=0.1,
        seed=1,
        dtype=torch.float32,
        manual_grad=manual_grad,
        distance_metric='manhattan',
    )

    t0 = time.time()
    config, result = optimizer.optimize()
    optimal_inds = torch.argwhere(result == result.min()).reshape(-1)
    legalizer = Legalizer(placer=fpga_placer, device=dev)
    all_ids = fpga_placer.get_ids()
    region_id_map = dict(zip(fpga_placer.regions, all_ids))

    best_config = {r: config[r][optimal_inds[0]] for r in config}
    best_ids = {r: region_id_map[r] for r in config if r in region_id_map}

    placement_legalized, overlap, fem_hpwl_initial, fem_hpwl_final = legalizer.legalize_placement(
        best_config, best_ids
    )

    t1 = time.time()

    return {
        "placement_legalized": placement_legalized,
        "overlap": overlap,
        "fem_hpwl_initial": fem_hpwl_initial,
        "fem_hpwl_final": fem_hpwl_final,
        "time": t1 - t0,
    }

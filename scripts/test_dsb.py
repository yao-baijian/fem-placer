"""
Test script for FPGA placement using discrete Simulated Bifurcation (dSB).

This script solves the placement problem by constructing a full QUBO matrix
with one-hot and at-most-one constraints, then solving it with the
simulated-bifurcation library (heated discrete mode).

Uses the master branch FpgaPlacer API (net_manager for coupling matrix).
"""

import sys
import os
import json
sys.path.insert(0, '.')

import torch
from fem_placer import (
    FpgaPlacer,
    Legalizer,
    solve_placement_sb,
    FPGAPlacementOptimizer,
)
from fem_placer.logger import *
from scripts.qubo_utils import reconstruct_logic_site_coords

SET_LEVEL('WARNING')  # Set higher to suppress unnecessary logs from libraries

# Configuration
instances = ['c2670', 'c5315', 'c6288', 'c7552',
             's1488', 's5378', 's9234', 's15850']  # Change to your desired instance

agents = 32
max_steps = 1000
default_lam = 0.1        # one-hot constraint weight, fallback default
default_mu = 1200         # at-most-one constraint weight, fallback default
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

# Set to False to run without Vivado, using saved init_params.json
USE_VIVADO = True

if USE_VIVADO:
    config_path = './scripts/config/bsb_summary.json'
else:
    config_path = './scripts/config/bsb_summary_no_vivado.json'

bsb_config = {}
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        bsb_config = json.load(f)

if USE_VIVADO:
    print(f"{'Benchmarks':<12} {'Instance':<10} {'Inst':<6} {'Overlap':<8} "
          f"{'HPWL Init':<18} {'HPWL Final':<16} {'QUBO Energy':<12}")
else:
    print(f"{'Benchmarks':<12} {'Instance':<10} {'Inst':<6} {'Overlap':<8} "
          f"{'QUBO Energy':<12} {'QUBO Mem(MB)':<12} {'Time (s)':<10}")

for instance in instances:
    lam = bsb_config.get(instance, {}).get('lam', default_lam)
    mu = bsb_config.get(instance, {}).get('mu', default_mu)

    if USE_VIVADO:
        dcp_file = f'./vivado/output_dir/{instance}/post_impl.dcp'
        output_file = f'bsb_placement_{instance}.dcp'
        INFO(f"Processing instance: {dcp_file}")
        fpga_placer = FpgaPlacer(utilization_factor=0.4)
        vivado_hpwl, inst_num, net_num = fpga_placer.init_placement(dcp_file=dcp_file, dcp_output=output_file)

        J = fpga_placer.net_manager.insts_matrix
        num_inst = fpga_placer.instances['logic'].num
        # Get grid information
        logic_grid = fpga_placer.get_grid('logic')
        # Create site coordinates matrix
        logic_site_coords = torch.cartesian_prod(
            torch.arange(logic_grid.width, dtype=torch.float32),
            torch.arange(logic_grid.height, dtype=torch.float32)
        )

        # QUBO size warning
        n_sites = logic_site_coords.shape[0]
        qubo_dim = num_inst * n_sites + n_sites
        qubo_memory_gb = qubo_dim ** 2 * 4 / 1024 ** 3
        INFO(f"qubo matrix size: {qubo_dim} x {qubo_dim}, memory: ~{qubo_memory_gb:.2f} GB")
        INFO(f"dsb params: agents={agents}, max_steps={max_steps}, lam={lam}, mu={mu}")

        # Solve with dSB
        INFO("solving placement with dsb...")
        site_indices, grid_coords, energy, meta = solve_placement_sb(
            J, logic_site_coords,
            lam=lam, mu=mu,
            agents=agents, max_steps=max_steps,
            best_only=True,
        )
        # Check feasibility
        n_unique = len(torch.unique(site_indices))
        INFO(f"Unique sites used: {n_unique} / {num_inst} instances")
        if n_unique < num_inst:
            INFO(f"Only {n_unique} distinct sites — {num_inst - n_unique} instances overlap")
        INFO(f"QUBO energy: {energy:.4f}")

        # Legalizing
        INFO("Legalizing placement...")
        coords = logic_grid.to_real_coords_tensor(grid_coords)
        logic_ids, io_ids = fpga_placer.get_ids()
        legalizer = Legalizer(placer=fpga_placer, device=dev)
        placement_legalized, overlap, hpwl_before, hpwl_after = legalizer.legalize_placement(
            coords, logic_ids
        )

        inst_num_val = inst_num['logic_inst_num']
        hpwl_before_val = hpwl_before
        hpwl_after_val = hpwl_after

        print(f"{'Benchmarks':<12} {instance:<10} {inst_num_val:<6} {overlap:<8} "
              f"{hpwl_before_val:<18.2f} {hpwl_after_val:<16.2f} {energy:<12.2f}")
    else:
        # No-Vivado mode: load optimizer params and solve QUBO directly
        try:
            optimizer = FPGAPlacementOptimizer.from_saved_params(
                f'result/{instance}/init_params.json',
                num_trials=1,
                num_steps=1,
                dev=dev
            )
        except Exception as e:
            print(f"Skipping {instance} (init_params.json not found or error loading): {e}")
            continue

        J = optimizer.coupling_matrix.cpu()
        n_sites = optimizer.num_site
        num_inst = optimizer.num_inst

        logic_site_coords = reconstruct_logic_site_coords(n_sites)

        qubo_dim = num_inst * n_sites + n_sites
        qubo_memory_mb = qubo_dim ** 2 * 4 / 1024 ** 2

        start_time = time.time()
        site_indices, grid_coords, energy, meta = solve_placement_sb(
            J, logic_site_coords,
            lam=lam, mu=mu,
            agents=agents, max_steps=max_steps,
            best_only=True,
        )
        elapsed_time = time.time() - start_time

        # Check feasibility
        n_unique = len(torch.unique(site_indices))
        overlap = num_inst - n_unique

        print(f"{'Benchmarks':<12} {instance:<10} {num_inst:<6} {overlap:<8} "
              f"{energy:<12.2f} {qubo_memory_mb:<12.2f} {elapsed_time:<10.2f}")
    

    # print(f"{'Benchmarks':<12} {'Instance':<10} {'Inst':<6} {'Site/Total':<14} {'Overlap':<8} "
    #     f"{'HPWL Init':<18} {'HPWL Final':<16} {'QUBO Energy':<12}")

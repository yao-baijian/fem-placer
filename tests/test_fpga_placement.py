"""
Test script for FPGA placement optimizer using QUBO formulation.

This script tests the FPGAPlacementOptimizer on a real FPGA design.
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
    Legalizer,
    Router,
    FPGAPlacementOptimizer,
    TimingAnalyzer,
    analyze_placement_timing,
    parse_vivado_timing,
)
from fem_placer.logger import *
from fem_placer.config import *
from ml.dataset import *
from ml.predict import predict_alpha
import glob
import re
import os

def get_vivado_place_times(logs_dir='./vivado/output_dir'):
    vivado_times = {}
    
    if not os.path.exists(logs_dir):
        return vivado_times
        
    for instance_dir in os.listdir(logs_dir):
        place_time_file = os.path.join(logs_dir, instance_dir, 'place_time.txt')
        if os.path.isfile(place_time_file):
            try:
                with open(place_time_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        vivado_times[instance_dir] = int(float(content))
            except Exception as e:
                print(f"Error reading {place_time_file}: {e}")
                
    return vivado_times

vivado_place_times = get_vivado_place_times()

# Timing analysis configuration
TIMING_CLOCK_PERIOD_NS = 5.0  # 200 MHz target

def get_vivado_timing_metrics(instance_dir='./vivado/output_dir'):
    """Read Vivado timing metrics if available."""
    metrics = {}
    metrics_file = os.path.join(instance_dir, 'timing_metrics.txt')
    if os.path.isfile(metrics_file):
        try:
            with open(metrics_file, 'r') as f:
                for line in f:
                    if ':' in line:
                        key, val = line.split(':', 1)
                        metrics[key.strip()] = val.strip()
        except Exception:
            pass
    return metrics

SET_LEVEL('INFO')

# instances = ['c2670', 'c5315', 'c6288', 'c7552',
#              's1488', 's5378', 's9234', 's15850', 'bgm', 'sha1', 'RLE_BlobMerging', 'FPGA-example1']

# instances = ['bgm', 'blob_merge', 'boundtop', 'ch_intrinsics', 'diffeq', 'diffeq2', 'LU8PEEng', 
#             'LU32PEEng', 'mcml', 'mkDelayWorker32B', 'mkPktMerge', 'mkSMAdapter4B', 'or1200', 
#             'raygentop', 'sha', 'stereovision0', 'stereovision1', 'stereovision2', 'stereovision3', 'RLE_BlobMerging']

# instances = ['c2670_boundary', 'c5315_boundary', 'c6288_boundary', 'c7552_boundary',
#              's1488_boundary', 's5378_boundary', 's9234_boundary', 's15850_boundary', 'FPGA-example1_boundary']

# instances = ['c2670_boundary']

instances = ['FPGA-example2']

# 'bgm_boundary', 'RLE_BlobMerging_boundary', 'sha1_boundary'

# instances = ['bgm', 'sha1', 'RLE_BlobMerging']
            
draw_evolution = False
draw_loss_function = False
draw_final_placement = False
num_trials = 5
num_steps = 500
dev = 'cuda'
manual_grad = False
anneal='exp'
io_factor = 1

print(f"{'Benchmarks':<12} {'Instance':<10} {'Inst':<6} {'IO Inst':<6} {'Net/Total':<14} {'Overlap':<8} "
    f"{'Alpha':<8} {'Beta':<8} {'HPWL Init':<18} {'HPWL Final':<16} {'HPWL Vivado':<12} "
    f"{'Time(s)':<10} {'VivadoT(s)':<10} "
    f"{'FEM-WNS(ns)':<14} {'FEM-Fmax':<12} {'Vvd-WNS':<12} {'Vvd-Fmax':<12}")

for instance in instances:
    place_type = PlaceType.CENTERED
    debug = False
    fpga_placer = FpgaPlacer(place_orientation = place_type, 
                            grid_type = GridType.SQUARE,
                            place_mode = IoMode.NORMAL,
                            utilization_factor = 0.4,
                            debug = debug,
                            device = dev)

    fpga_placer.set_instance_name(instance)
    
    vivado_hpwl, inst_num, net_num = fpga_placer.init_placement(f'./vivado/output_dir/{instance}/post_impl.dcp', f'./vivado/output_dir/{instance}/optimized_placement.pl')
    net_ratio = f"{net_num['logic_net_num']}/{net_num['total_net_num']}"
    global_drawer = PlacementDrawer(placer=fpga_placer)
    row = extract_features_from_placer(fpga_placer,
                                       alpha=0, 
                                       beta=0, 
                                       with_io=False)
    
    alpha = predict_alpha(row)
    INFO(f'instance {instance}, predicted alpha {alpha}')
    used_alpha = alpha * 0.001
    fpga_placer.set_alpha(used_alpha)

    if place_type == PlaceType.IO:
        used_beta =alpha * 0.001
        fpga_placer.set_beta(used_beta)
    else:
        used_beta = 0.0

    # fpga_placer.set_alpha(30)
    
    optimizer = FPGAPlacementOptimizer(
        num_inst=fpga_placer.instances['logic'].num,
        num_fixed_inst=fpga_placer.instances['io'].num,
        num_site=fpga_placer.get_grid('logic').area,
        num_fixed_site=fpga_placer.get_grid('io').area,
        coupling_matrix=fpga_placer.net_manager.insts_matrix,
        site_coords_matrix=fpga_placer.logic_site_coords,
        io_site_connect_matrix=fpga_placer.net_manager.io_insts_matrix,
        io_site_coords=fpga_placer.io_site_coords,
        constraint_alpha=fpga_placer.constraint_alpha,
        constraint_beta=fpga_placer.constraint_beta,  # For IO placements, beta is set separately
        num_trials=num_trials,
        num_steps=num_steps,
        dev=dev,
        betamin=0.01,
        betamax=0.5,
        anneal=anneal,
        optimizer='adam',
        learning_rate=0.1,
        h_factor=0.01,
        io_factor=io_factor,
        seed=1,
        dtype=torch.float32,
        with_io=(place_type == PlaceType.IO),
        manual_grad=manual_grad
    )
    
    start_time = time.time()
    config, result = optimizer.optimize()
    end_time = time.time()
    optimize_time = end_time - start_time
    
    optimal_inds = torch.argwhere(result==result.min()).reshape(-1)
    legalizer = Legalizer(placer=fpga_placer,
                        device=dev)
    router = Router(placer=fpga_placer)
    logic_ids, io_ids = fpga_placer.get_ids()

    if place_type == PlaceType.IO:
        # real_logic_coords = fpga_placer.get_grid('logic').to_real_coords_tensor(config[0][optimal_inds[0]])
        # real_io_coords = fpga_placer.get_grid('io').to_real_coords_tensor(config[1][optimal_inds[0]])

        real_logic_coords = config[0][optimal_inds[0]]
        real_io_coords = config[1][optimal_inds[0]]
        placement_legalized, overlap, fem_hpwl_initial, fem_hpwl_final = legalizer.legalize_placement(real_logic_coords, logic_ids, real_io_coords, io_ids, include_io = True)
        all_coords = torch.cat([placement_legalized[0], placement_legalized[1]], dim=0)
        routes = router.route_connections(fpga_placer.net_manager.insts_matrix, all_coords)
        vivado_time_str = str(vivado_place_times.get(instance, 'N/A'))
        
        # --- Timing Analysis (Framework) ---
        timing_result = analyze_placement_timing(
            fpga_placer,
            placement_legalized[0],
            placement_legalized[1],
            include_io=True,
            clock_period_ns=TIMING_CLOCK_PERIOD_NS,
        )
        
        # --- Timing Analysis (Vivado) ---
        vivado_timing = get_vivado_timing_metrics(f'./vivado/output_dir/{instance}')
        vvd_wns = vivado_timing.get('WNS', 'N/A')
        vvd_fmax = vivado_timing.get('Fmax (MHz)', 'N/A')
        
        print(f"{'Benchmarks':<12} {instance:<10} {inst_num['logic_inst_num']:<6} {inst_num['io_inst_num']:<6} {net_ratio:<14} {overlap:<8} "
            f"{used_alpha:<8.2f} {used_beta:<8.2f} {fem_hpwl_initial['hpwl']:<18.2f} {fem_hpwl_final['hpwl']:<16.2f} {vivado_hpwl['hpwl']:<12.2f} {optimize_time:<10.2f} {vivado_time_str:<10} "
            f"{timing_result.wns*1e9:<14.3f} {timing_result.fmax:<12.1f} {vvd_wns:<12} {vvd_fmax:<12}")
    else:
        real_logic_coords = config[optimal_inds[0]]
        placement_legalized, overlap, fem_hpwl_initial, fem_hpwl_final = legalizer.legalize_placement(real_logic_coords, logic_ids)
        routes = router.route_connections(fpga_placer.net_manager.insts_matrix, (placement_legalized[0]))
        vivado_time_str = str(vivado_place_times.get(instance, 'N/A'))
        
        # --- Timing Analysis (Framework) ---
        timing_result = analyze_placement_timing(
            fpga_placer,
            placement_legalized[0],
            clock_period_ns=TIMING_CLOCK_PERIOD_NS,
        )
        
        # --- Timing Analysis (Vivado) ---
        vivado_timing = get_vivado_timing_metrics(f'./vivado/output_dir/{instance}')
        vvd_wns = vivado_timing.get('WNS', 'N/A')
        vvd_fmax = vivado_timing.get('Fmax (MHz)', 'N/A')
        
        print(f"{'Benchmarks':<12} {instance:<10} {inst_num['logic_inst_num']:<6} {inst_num['io_inst_num']:<6} {net_ratio:<14} {overlap:<8} "
            f"{used_alpha:<8.2f} {used_beta:<8.2f} {fem_hpwl_initial['hpwl_no_io']:<18.2f} {fem_hpwl_final['hpwl_no_io']:<16.2f} {vivado_hpwl['hpwl_no_io']:<12.2f} {optimize_time:<10.2f} {vivado_time_str:<10} "
            f"{timing_result.wns*1e9:<14.3f} {timing_result.fmax:<12.1f} {vvd_wns:<12} {vvd_fmax:<12}")
    
    # Print detailed timing report
    print()
    print(timing_result.format_report())
    print()

    if draw_loss_function:
        global_drawer.plot_fpga_placement_loss(f'result/{instance}/hpwl_loss.png')

    if draw_evolution:
        global_drawer.draw_multi_step_placement(f'result/{instance}/placement_evolution.png')

    if draw_final_placement:
        include_io = (place_type == PlaceType.IO)
        io_coords = placement_legalized[1] if include_io else None
        global_drawer.draw_place_and_route(placement_legalized[0], routes, io_coords, include_io, 1000, title_suffix="Final Placement with Routing")


# =============================================================================
# Example: How to use saved parameters for collaborator without Vivado
# =============================================================================
#
# 1. Save parameters after init_placement (on machine with Vivado):
#    fpga_placer.save_init_params(instance_name='c7552')
#    # Output: result/c7552/init_params.json
#
# 2. Load parameters and create optimizer (on machine without Vivado):
#    from fem_placer import FPGAPlacementOptimizer
#
#    optimizer = FPGAPlacementOptimizer.from_saved_params(
#        'result/c7552/init_params.json',
#        num_trials=10,
#        num_steps=1000,
#        dev='cpu'
#    )
#
#    config, result = optimizer.optimize()
# =============================================================================


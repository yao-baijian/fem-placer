"""
FPGA Placement Package

This package provides FPGA placement functionality using the FEM framework.
Uses QUBO formulation for placement optimization.
"""

from .placer import FpgaPlacer
from .drawer import PlacementDrawer
from .legalizer import Legalizer
from .router import Router
from .optimizer import FPGAPlacementOptimizer
from .timer import Timer
from .timing_analyzer import (
    TimingAnalyzer,
    TimingSummary,
    analyze_placement_timing,
    parse_vivado_timing,
    generate_vivado_timing_tcl,
)

# QUBO approach (site_coords_matrix based)
from .objectives import (
    get_site_distance_matrix,
    get_expected_placements_from_index,
    get_hard_placements_from_index,
    get_placements_from_index_st,
    get_hpwl_loss_qubo,
    get_hpwl_loss_qubo_with_io,
    get_constraints_loss,
    get_constraints_loss_with_io,
    expected_fpga_placement,
    expected_fpga_placement_with_io,
    infer_placements,
    infer_placements_with_io,
    get_loss_history,
    get_placement_history,
    clear_history,
    manual_grad_hpwl_loss,
    manual_grad_constraint_loss,
    manual_grad_placement,
    export_placement_qubo,
    decode_qubo_solution,
    solve_placement_sb,
    solve_placement_cyclic,
)

__all__ = [
    # Core classes
    'FpgaPlacer',
    'PlacementDrawer',
    'Legalizer',
    'Router',
    'FPGAPlacementOptimizer',
    'Timer',
    'TimingAnalyzer',
    'TimingSummary',
    'analyze_placement_timing',
    'parse_vivado_timing',
    'generate_vivado_timing_tcl',

    # QUBO functions
    'get_site_distance_matrix',
    'get_expected_placements_from_index',
    'get_hard_placements_from_index',
    'get_placements_from_index_st',
    'get_hpwl_loss_qubo',
    'get_hpwl_loss_qubo_with_io',
    'get_constraints_loss',
    'get_constraints_loss_with_io',
    'expected_fpga_placement',
    'expected_fpga_placement_with_io',
    'infer_placements',
    'infer_placements_with_io',
    'get_loss_history',
    'get_placement_history',
    'clear_history',
    'manual_grad_hpwl_loss',
    'manual_grad_constraint_loss',
    'manual_grad_placement',
    'export_placement_qubo',
    'decode_qubo_solution',
    'solve_placement_sb',
    'solve_placement_cyclic',

    # Hypergraph balanced min-cut
    'balance_constrain',
    'balance_constrain_softplus',
    'balance_constrain_relu',
    'infer_hyperbmincut',
    'expected_hyperbmincut',
    'expected_hyperbmincut_expected_nodes_temped',
    'expected_hyperbmincut_max_expected_nodes',
    'expected_hyperbmincut_all_comb',
    'expected_hyperbmincut_expected_crossing_simplified',
    'manual_grad_hyperbmincut',
]

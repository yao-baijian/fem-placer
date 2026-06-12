"""
Shared utilities for FEM placer tests.

Consolidates helpers from:
- ``test_fpga_placement.py`` (Vivado I/O, print formatting)
- ``placement_eval_logic.py`` (``run_logic_placement``)
- ``extract_placer_info.py`` (connectivity extraction)
"""

import os
import sys
import json
import time
import shutil
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

from fem_placer import (
    FpgaPlacer,
    Legalizer,
    Router,
    FPGAPlacementOptimizer,
    RapidWrightTimer,
)
from fem_placer.logger import INFO, WARNING, ERROR
from fem_placer.config import PlaceType, GridType, IoMode


# ---------------------------------------------------------------------------
# TestConfig — single-source config with attribute access
# ---------------------------------------------------------------------------

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'fem_placer', 'config.json')
LOCAL_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config', 'config.json')

_ENUM_MAP = {
    'grid_type': (GridType, 'SQUARE'),
    'place_mode': (IoMode, 'NORMAL'),
}


@dataclass
class TestConfig:
    """Test configuration loaded from JSON.  Access fields as attributes."""
    instances: List[str] = field(default_factory=lambda: ['s15850'])
    regions: List[str] = field(default_factory=lambda: ['logic', 'io'])
    verbose: bool = False
    grid_type: GridType = GridType.SQUARE
    place_mode: IoMode = IoMode.NORMAL
    utilization_factor: float = 0.4
    num_trials: int = 1
    num_steps: int = 500
    dev: str = 'cuda'
    manual_grad: bool = False
    anneal: str = 'exp'
    io_factor: int = 1
    alpha: int = 0
    beta: int = 0
    alpha_factor: int = 1
    beta_factor: int = 1
    learning_rate: float = 0.1
    h_factor: float = 0.01
    betamin: float = 0.01
    betamax: float = 0.5
    seed: int = 1
    clock_period_ns: float = 5.0
    draw_evolution: bool = False
    draw_loss_function: bool = False
    draw_final_placement: bool = False
    record_mode: str = 'inverse_sqr'
    map_mode: str = 'no'
    net_offset_coeff: float = 1.0
    hpwl_workers: Optional[int] = None
    hpwl_parallel_threshold: int = 4

    @classmethod
    def load(cls) -> 'TestConfig':
        """Load from ``tests/config/config.json``, copying default if missing."""
        if not os.path.exists(LOCAL_CONFIG_PATH):
            os.makedirs(os.path.dirname(LOCAL_CONFIG_PATH), exist_ok=True)
            if os.path.exists(DEFAULT_CONFIG_PATH):
                shutil.copy2(DEFAULT_CONFIG_PATH, LOCAL_CONFIG_PATH)
                INFO(f"Copied default config to {LOCAL_CONFIG_PATH}")
            else:
                ERROR(f"Default config not found at {DEFAULT_CONFIG_PATH}")
                return cls()

        with open(LOCAL_CONFIG_PATH, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        # Strip comment keys
        raw = {k: v for k, v in raw.items() if not k.startswith('_')}

        # Convert enum fields
        for key, (enum_cls, default_name) in _ENUM_MAP.items():
            if key in raw:
                raw[key] = enum_cls[raw[key]]

        return cls(**{k: v for k, v in raw.items() if hasattr(cls, k)})


# ---------------------------------------------------------------------------
# Vivado helpers
# ---------------------------------------------------------------------------

def get_vivado_place_times(logs_dir: str = './vivado/output_dir') -> Dict[str, str]:
    """Read Vivado placement run-times from ``place_time.txt`` files."""
    times = {}
    if not os.path.exists(logs_dir):
        return times
    for inst_dir in os.listdir(logs_dir):
        pt_file = os.path.join(logs_dir, inst_dir, 'place_time.txt')
        if os.path.isfile(pt_file):
            try:
                with open(pt_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        times[inst_dir] = str(int(float(content)))
            except Exception as e:
                print(f"Error reading {pt_file}: {e}")
    return times


def get_vivado_timing_metrics(instance_dir: str = './vivado/output_dir') -> Dict[str, str]:
    """Read Vivado timing metrics from ``timing_metrics.txt``."""
    metrics = {}
    metrics_file = os.path.join(instance_dir, 'timing_metrics.txt')
    if os.path.isfile(metrics_file):
        try:
            with open(metrics_file, 'r') as f:
                for line in f:
                    if ':' in line:
                        k, v = line.split(':', 1)
                        metrics[k.strip()] = v.strip()
        except Exception:
            pass
    return metrics


# ---------------------------------------------------------------------------
# Placement runner (from placement_eval_logic.py)
# ---------------------------------------------------------------------------

def run_logic_placement(
    fpga_placer: FpgaPlacer,
    alpha: float,
    beta: float = 0.0,
    io_factor: float = 1.0,
    num_trials: int = 5,
    num_steps: int = 200,
    dev: str = "cpu",
    manual_grad: bool = False,
    anneal: str = "inverse",
    place_type: PlaceType = PlaceType.CENTERED,
    learning_rate: float = 0.1,
    h_factor: float = 0.01,
    betamin: float = 0.01,
    betamax: float = 0.5,
    seed: int = 1,
) -> Dict[str, Any]:
    """Run a single logic placement optimisation + legalization.

    Returns a dict with keys:
    ``placement_legalized``, ``overlap``, ``fem_hpwl_initial``,
    ``fem_hpwl_final``, ``time``.
    """
    # Clear grid state before each run
    fpga_placer.grids["logic"].clear_all()
    if place_type == PlaceType.IO:
        fpga_placer.grids["io"].clear_all()

    fpga_placer.set_alpha(alpha)
    if place_type == PlaceType.IO:
        fpga_placer.set_beta(beta)

    optimizer = FPGAPlacementOptimizer(
        num_inst=fpga_placer.instances["logic"].num,
        num_fixed_inst=fpga_placer.instances["io"].num,
        num_site=fpga_placer.get_grid("logic").area,
        num_fixed_site=fpga_placer.get_grid("io").area,
        coupling_matrix=fpga_placer.net_manager.insts_matrix,
        site_coords_matrix=fpga_placer.logic_site_coords,
        io_site_connect_matrix=fpga_placer.net_manager.io_insts_matrix,
        io_site_coords=fpga_placer.io_site_coords,
        constraint_alpha=fpga_placer.constraint_alpha,
        constraint_beta=fpga_placer.constraint_alpha,
        num_trials=num_trials,
        num_steps=num_steps,
        dev=dev,
        betamin=betamin,
        betamax=betamax,
        anneal=anneal,
        optimizer="adam",
        learning_rate=learning_rate,
        h_factor=h_factor,
        io_factor=io_factor,
        seed=seed,
        dtype=torch.float32,
        with_io=(place_type == PlaceType.IO),
        manual_grad=manual_grad,
    )

    t0 = time.time()
    config, result = optimizer.optimize()
    optimal_inds = torch.argwhere(result == result.min()).reshape(-1)
    legalizer = Legalizer(placer=fpga_placer, device=dev)
    logic_ids, io_ids = fpga_placer.get_ids()

    if place_type == PlaceType.IO:
        real_logic_coords = config[0][optimal_inds[0]]
        real_io_coords = config[1][optimal_inds[0]]
        placement_legalized, overlap, fem_hpwl_initial, fem_hpwl_final = \
            legalizer.legalize_placement(
                real_logic_coords, logic_ids,
                real_io_coords, io_ids, include_io=True)
    else:
        real_logic_coords = config[optimal_inds[0]]
        placement_legalized, overlap, fem_hpwl_initial, fem_hpwl_final = \
            legalizer.legalize_placement(real_logic_coords, logic_ids)

    t1 = time.time()

    return {
        "placement_legalized": placement_legalized,
        "overlap": overlap,
        "fem_hpwl_initial": fem_hpwl_initial,
        "fem_hpwl_final": fem_hpwl_final,
        "time": t1 - t0,
    }


# ---------------------------------------------------------------------------
# Timing runner
# ---------------------------------------------------------------------------

def run_timing_analysis(
    placer: FpgaPlacer,
    placement_legalized: Tuple[torch.Tensor, Optional[torch.Tensor]],
    clock_period_ns: float,
    use_rapidwright: bool = True,
    instance_name: str = "fem_placement",
) -> Any:
    """Run timing analysis on a FEM placement result.

    Args:
        use_rapidwright: If True, use ``RapidWrightTimer`` (RW place + route).
                         If False, use ``VivadoTimingRunner`` (Vivado Tcl flow).
    """
    logic_coords = placement_legalized[0]
    io_coords = placement_legalized[1] if len(placement_legalized) > 1 else None

    if use_rapidwright:
        from fem_placer import RapidWrightTimer
        timer = RapidWrightTimer()
        return timer.run(
            design=placer.design,
            placer=placer,
            logic_coords=logic_coords,
            io_coords=io_coords,
            include_io=(io_coords is not None),
            clock_period_ns=clock_period_ns,
            instance_name=instance_name,
            route_design=True,
        )
    else:
        from fem_placer import VivadoTimingRunner
        timer = VivadoTimingRunner(vivado_path='vivado')
        return timer.run(
            design=placer.design,
            placer=placer,
            logic_coords=logic_coords,
            io_coords=io_coords,
            include_io=(io_coords is not None),
            clock_period_ns=clock_period_ns,
            instance_name=instance_name,
            run_vivado=True,
        )


# ---------------------------------------------------------------------------
# Connectivity extraction (from extract_placer_info.py)
# ---------------------------------------------------------------------------

def extract_connectivity(fpga_placer: FpgaPlacer) -> Dict[str, Any]:
    """Compute min/max/avg connectivity from the placer's net manager."""
    net_sizes = [len(sites) for sites in fpga_placer.net_manager.net_to_sites.values()]
    if net_sizes:
        min_conn = min(net_sizes)
        max_conn = max(net_sizes)
        avg_conn = sum(net_sizes) / len(net_sizes)
    else:
        min_conn = max_conn = avg_conn = 0
    return {'min': min_conn, 'max': max_conn, 'avg': avg_conn}


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

RESULT_HEADER = (
    f"{'Benchmarks':<12} {'Instance':<10} {'Inst':<6} {'IO Inst':<6} "
    f"{'Net/Total':<14} {'Overlap':<8} {'Alpha':<8} {'Beta':<8} "
    f"{'HPWL Init':<18} {'HPWL Final':<16} {'HPWL Vivado':<12} "
    f"{'Time(s)':<10} {'VivadoT(s)':<10} "
    f"{'WNS(ns)':<14} {'Fmax(MHz)':<12}"
)


def format_result_row(
    instance: str,
    inst_num: Dict[str, int],
    net_ratio: str,
    overlap: float,
    used_alpha: float,
    used_beta: float,
    fem_hpwl_initial: Dict[str, float],
    fem_hpwl_final: Dict[str, float],
    vivado_hpwl: Dict[str, float],
    optimize_time: float,
    vivado_time_str: str,
    wns_ns: float,
    fmax_mhz: float,
    include_io: bool = False,
) -> str:
    """Format a single result row matching the header columns."""
    hpwl_init_key = 'hpwl' if include_io else 'hpwl_no_io'
    hpwl_final_key = 'hpwl' if include_io else 'hpwl_no_io'
    hpwl_viv_key = 'hpwl' if include_io else 'hpwl_no_io'

    return (
        f"{'Benchmarks':<12} {instance:<10} {inst_num['logic_inst_num']:<6} "
        f"{inst_num['io_inst_num']:<6} {net_ratio:<14} {overlap:<8} "
        f"{used_alpha:<8.2f} {used_beta:<8.2f} "
        f"{fem_hpwl_initial.get(hpwl_init_key, 0):<18.2f} "
        f"{fem_hpwl_final.get(hpwl_final_key, 0):<16.2f} "
        f"{vivado_hpwl.get(hpwl_viv_key, 0):<12.2f} "
        f"{optimize_time:<10.2f} {vivado_time_str:<10} "
        f"{wns_ns:<14.3f} {fmax_mhz:<12.1f}"
    )


# ---------------------------------------------------------------------------
# PlacementTestRunner — clean, reusable test pipeline
# ---------------------------------------------------------------------------

class PlacementTestRunner:
    """High-level runner that handles the full FEM placement pipeline for one instance.

    Usage::

        cfg = TestConfig.load()
        runner = PlacementTestRunner(cfg)
        print(RESULT_HEADER)
        for inst in cfg.instances:
            result = runner.run(inst)
            print(result.format_row())
    """

    def __init__(self, cfg: TestConfig):
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, instance: str) -> 'PlacementResult':
        """Run the full pipeline for *instance* and return a ``PlacementResult``."""
        import time as _time

        placer = FpgaPlacer(
            place_orientation=self.cfg.place_type,
            grid_type=self.cfg.grid_type,
            place_mode=self.cfg.place_mode,
            utilization_factor=self.cfg.utilization_factor,
            debug=self.cfg.debug,
            device=self.cfg.dev,
        )
        placer.set_instance_name(instance)

        dcp = f'./vivado/output_dir/{instance}/post_impl.dcp'
        pl_path = f'./vivado/output_dir/{instance}/optimized_placement.pl'
        vivado_hpwl, inst_num, net_num = placer.init_placement(dcp, pl_path)

        # Feature extraction & alpha prediction
        from ml.dataset import extract_features_from_placer
        from ml.predict import predict_alpha
        row = extract_features_from_placer(placer, alpha=0, beta=0, with_io=False)
        alpha_val = predict_alpha(row) * 0.001
        placer.set_alpha(alpha_val)

        # Optimizer
        optimizer = FPGAPlacementOptimizer(
            num_inst=placer.instances['logic'].num,
            num_fixed_inst=placer.instances['io'].num,
            num_site=placer.get_grid('logic').area,
            num_fixed_site=placer.get_grid('io').area,
            coupling_matrix=placer.net_manager.insts_matrix,
            site_coords_matrix=placer.logic_site_coords,
            io_site_connect_matrix=placer.net_manager.io_insts_matrix,
            io_site_coords=placer.io_site_coords,
            constraint_alpha=placer.constraint_alpha,
            constraint_beta=placer.constraint_beta,
            num_trials=self.cfg.num_trials,
            num_steps=self.cfg.num_steps,
            dev=self.cfg.dev,
            betamin=self.cfg.betamin,
            betamax=self.cfg.betamax,
            anneal=self.cfg.anneal,
            optimizer='adam',
            learning_rate=self.cfg.learning_rate,
            h_factor=self.cfg.h_factor,
            io_factor=self.cfg.io_factor,
            seed=self.cfg.seed,
            dtype=torch.float32,
            with_io=(self.cfg.place_type == PlaceType.IO),
            manual_grad=self.cfg.manual_grad,
        )

        t0 = _time.time()
        config, result = optimizer.optimize()
        optimize_time = _time.time() - t0

        # Legalize
        optimal_inds = torch.argwhere(result == result.min()).reshape(-1)
        legalizer = Legalizer(placer=placer, device=self.cfg.dev)
        router = Router(placer=placer)
        logic_ids, io_ids = placer.get_ids()

        include_io = (self.cfg.place_type == PlaceType.IO)
        if include_io:
            real_logic_coords = config[0][optimal_inds[0]]
            real_io_coords = config[1][optimal_inds[0]]
            legalized, overlap, hpwl_i, hpwl_f = legalizer.legalize_placement(
                real_logic_coords, logic_ids, real_io_coords, io_ids, include_io=True)
            all_coords = torch.cat([legalized[0], legalized[1]], dim=0)
            routes = router.route_connections(placer.net_manager.insts_matrix, all_coords)
        else:
            real_logic_coords = config[optimal_inds[0]]
            legalized, overlap, hpwl_i, hpwl_f = legalizer.legalize_placement(
                real_logic_coords, logic_ids)
            routes = router.route_connections(placer.net_manager.insts_matrix, legalized[0])

        # Timing
        timing_result = run_timing_analysis(
            placer=placer,
            placement_legalized=legalized,
            clock_period_ns=self.cfg.clock_period_ns,
            use_rapidwright=True,
            instance_name=instance,
        )

        # Drawings
        if self.cfg.draw_loss_function:
            from fem_placer import PlacementDrawer
            drawer = PlacementDrawer(placer=placer)
            drawer.plot_fpga_placement_loss(f'result/{instance}/hpwl_loss.png')

        if self.cfg.draw_final_placement:
            from fem_placer import PlacementDrawer
            drawer = PlacementDrawer(placer=placer)
            io_c = legalized[1] if include_io else None
            drawer.draw_place_and_route(legalized[0], routes, io_c, include_io, 1000)

        return PlacementResult(
            instance=instance,
            inst_num=inst_num,
            net_num=net_num,
            net_ratio=f"{net_num['logic_net_num']}/{net_num['total_net_num']}",
            overlap=overlap,
            alpha=alpha_val,
            beta=0.0,
            hpwl_initial=hpwl_i,
            hpwl_final=hpwl_f,
            hpwl_vivado=vivado_hpwl,
            optimize_time=optimize_time,
            timing=timing_result,
        )


@dataclass
class PlacementResult:
    """Structured result from a single placement run."""
    instance: str
    inst_num: Dict[str, int]
    net_num: Dict[str, int]
    net_ratio: str
    overlap: float
    alpha: float
    beta: float
    hpwl_initial: Dict[str, float]
    hpwl_final: Dict[str, float]
    hpwl_vivado: Dict[str, float]
    optimize_time: float
    timing: Any

    def format_row(self) -> str:
        v = self
        wns_ns = v.timing.wns * 1e9
        fmax_mhz = v.timing.fmax
        hpwl_init_key = 'hpwl'
        hpwl_final_key = 'hpwl'
        hpwl_viv_key = 'hpwl'
        return (
            f"{'Benchmarks':<12} {v.instance:<10} {v.inst_num['logic_inst_num']:<6} "
            f"{v.inst_num.get('io_inst_num', 0):<6} {v.net_ratio:<14} {v.overlap:<8} "
            f"{v.alpha:<8.2f} {v.beta:<8.2f} "
            f"{v.hpwl_initial.get(hpwl_init_key, 0):<18.2f} "
            f"{v.hpwl_final.get(hpwl_final_key, 0):<16.2f} "
            f"{v.hpwl_vivado.get(hpwl_viv_key, 0):<12.2f} "
            f"{v.optimize_time:<10.2f} {'N/A':<10} "
            f"{wns_ns:<14.3f} {fmax_mhz:<12.1f}"
        )

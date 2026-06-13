"""
Timing Analyzer for FPGA Placement.

Provides two complementary timing analysis capabilities:

1. **Framework-level timing estimation**: Estimates WNS, TNS, and Fmax from the
   placement coordinates and netlist topology without requiring Vivado. Uses
   wirelength-based delay models and logic depth estimation.

2. **Vivado timing report parser**: Parses Vivado timing_summary.rpt files to
   extract WNS, TNS, and other timing metrics for comparison.
"""

import re
import os
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from .logger import INFO, WARNING, ERROR


# =============================================================================
# Default technology parameters (7-series / Ultrascale like)
# These can be overridden via the TimingAnalyzer constructor.
# Values are representative for a typical 28nm/20nm FPGA process.
# =============================================================================
DEFAULT_CELL_DELAYS = {
    'LUT':      {'min': 50e-12,  'max': 120e-12},   # 50-120 ps
    'FF':       {'clk2q': 60e-12, 'setup': 40e-12,   'hold': 10e-12},
    'CARRY':    {'min': 30e-12,  'max': 80e-12},
    'MUX':      {'min': 40e-12,  'max': 90e-12},
    'DSP':      {'min': 200e-12, 'max': 500e-12},
    'BRAM':     {'min': 300e-12, 'max': 600e-12},
    'IO':       {'min': 100e-12, 'max': 200e-12},
}

# Wire delay per site pitch (ps per SLICE)
# For Ultrascale: ~0.2-0.5 ps per site for local routing
DEFAULT_WIRE_DELAY_PER_SITE = 0.3e-12  # 0.3 ps per site pitch

# Default clock period if none specified (ns)
DEFAULT_CLOCK_PERIOD = 5.0  # 200 MHz

# =============================================================================
# Cell type classification helpers
# =============================================================================
# FF cell type prefixes (sequential elements)
FF_TYPE_PREFIXES = ('FD', 'FDE', 'FDCE', 'FDPE', 'FDRE', 'FDSE', 'FDRSE',
                     'FD_1', 'FDC', 'FDP', 'FDR', 'FDS')

# Combinational cell type prefixes
LUT_TYPE_PREFIXES = ('LUT1', 'LUT2', 'LUT3', 'LUT4', 'LUT5', 'LUT6')
CARRY_TYPE_PREFIXES = ('CARRY4', 'CARRY8')
MUX_TYPE_PREFIXES = ('MUXF7', 'MUXF8', 'MUXF9', 'MUXF')
GATE_TYPE_PREFIXES = ('AND', 'OR', 'NAND', 'NOR', 'XOR', 'XNOR', 'BUF', 'INV',
                       'MUXCY', 'XORCY')

# Combinational cell output pin names (for tracing)
FF_OUTPUT_PINS = ('Q', 'Q0', 'Q1', 'Q2', 'Q3')
FF_INPUT_PINS = ('D', 'D0', 'D1', 'D2', 'D3')
LUT_OUTPUT_PINS = ('O', 'O5', 'O6')



class Timer:
    """
    Timer class for timing-aware placement optimisation.

    Handles timing analysis, path delay calculation, timing-weighted
    objectives, and congestion estimation.
    """

    def __init__(self):
        self.timing_library = {}
        self.cell_delays = {}
        self.net_delays = {}
        self.timing_paths = []
        self.site_to_index = {}
        self.optimizable_sites = []
        self.available_target_sites = []

    def setup_timing_analysis(self, design, timing_library):
        self.timing_library = timing_library
        self.cell_delays = self.extract_cell_delays(design)
        self.net_delays = self.extract_net_delays(design)
        self.timing_paths = self.extract_timing_paths(design)

    def extract_cell_delays(self, design):
        cell_delays = {}
        for cell in design.getCells():
            cell_type = cell.getType()
            if cell_type in self.timing_library:
                cell_delays[cell.getName()] = self.timing_library[cell_type]
            else:
                cell_delays[cell.getName()] = {
                    'min_delay': 0.1, 'max_delay': 0.2,
                    'setup_time': 0.05, 'hold_time': 0.02
                }
        return cell_delays

    def extract_net_delays(self, design):
        net_delays = {}
        for net in design.getNets():
            net_length = self.estimate_net_length(net)
            net_delays[net.getName()] = {
                'unit_delay': 0.01,
                'estimated_delay': net_length * 0.01
            }
        return net_delays

    def estimate_net_length(self, net):
        return len(net.getPins()) * 10

    def extract_timing_paths(self, design):
        timing_paths = []
        for cell in design.getCells():
            if 'FD' in cell.getType():
                timing_paths.append({
                    'start_cell': cell.getName(),
                    'end_cell': self.find_connected_flipflop(cell),
                    'required_time': 10.0,
                    'criticality': 1.0
                })
        return timing_paths

    def find_connected_flipflop(self, cell):
        for net in cell.getNets():
            for pin in net.getPins():
                other_cell = pin.getCell()
                if other_cell and other_cell != cell and 'FD' in other_cell.getType():
                    return other_cell.getName()
        return None



@dataclass
class TimingSummary:
    """Container for timing analysis results."""
    wns: float               # Worst Negative Slack (ns) — negative means violation
    tns: float               # Total Negative Slack (ns)
    fmax: float              # Estimated maximum clock frequency (MHz)
    estimated_period: float  # Estimated minimum clock period (ns)
    critical_path_delay: float  # Estimated critical path delay (ns)
    logic_depth: int         # Estimated logic depth (levels of logic)
    num_paths: int           # Number of timing paths analyzed
    num_violations: int      # Number of paths with violations
    wire_delays: Dict[str, float] = field(default_factory=dict)  # Per-net wire delay
    top_violated_nets: List[Tuple[str, float]] = field(default_factory=list)

    def format_report(self) -> str:
        """Format timing summary as a readable report string."""
        lines = []
        lines.append("=" * 65)
        lines.append("  Timing Analysis Report")
        lines.append("=" * 65)
        lines.append(f"  Critical Path Delay : {self.critical_path_delay*1e9:.3f} ns")
        lines.append(f"  Estimated Period    : {self.estimated_period*1e9:.3f} ns")
        lines.append(f"  Estimated Fmax      : {self.fmax:.1f} MHz")
        lines.append(f"  WNS (Worst Neg Slack): {self.wns*1e9:.3f} ns")
        lines.append(f"  TNS (Total Neg Slack): {self.tns*1e9:.3f} ns")
        lines.append(f"  Logic Depth         : {self.logic_depth} levels")
        lines.append(f"  Timing Paths        : {self.num_paths}")
        lines.append(f"  Violating Paths     : {self.num_violations}")
        if self.top_violated_nets:
            lines.append(f"  Top Violated Nets   :")
            for net_name, slack in self.top_violated_nets[:10]:
                lines.append(f"    {net_name:<40s} slack={slack*1e9:.3f} ns")
        lines.append("=" * 65)
        return "\n".join(lines)


class TimingAnalyzer:
    """
    Analyzes timing for FPGA placements.

    Provides two modes:
    - `analyze_framework()`: Estimate timing from placement coordinates + netlist.
    - `parse_vivado_report()`: Parse Vivado timing_summary.rpt for comparison.
    """

    def __init__(
        self,
        cell_delays: Optional[Dict[str, Dict[str, float]]] = None,
        wire_delay_per_site: float = DEFAULT_WIRE_DELAY_PER_SITE,
        clock_period: float = DEFAULT_CLOCK_PERIOD,
        ff_setup_time: float = 40e-12,
        ff_clk2q_time: float = 60e-12,
    ):
        """
        Args:
            cell_delays: Dict mapping cell types to delay dicts.
            wire_delay_per_site: Wire delay per SLICE pitch (seconds).
            clock_period: Target clock period in seconds.
            ff_setup_time: Flip-flop setup time (seconds).
            ff_clk2q_time: Flip-flop clock-to-Q delay (seconds).
        """
        self.cell_delays = cell_delays or DEFAULT_CELL_DELAYS
        self.wire_delay_per_site = wire_delay_per_site
        self.clock_period = clock_period
        self.ff_setup_time = ff_setup_time
        self.ff_clk2q_time = ff_clk2q_time

    # =========================================================================
    # Framework-Level Timing Estimation
    # =========================================================================

    def analyze_framework(
        self,
        net_manager: Any,
        logic_coords: torch.Tensor,
        io_coords: Optional[torch.Tensor] = None,
        include_io: bool = False,
        clock_period: Optional[float] = None,
    ) -> TimingSummary:
        """
        Estimate timing metrics from placement coordinates and netlist.

        Uses a simple model:
        - Net wire delay = HPWL * wire_delay_per_unit
        - Path delay = sum of cell delays + sum of wire delays along path
        - Critical path approximated by logic_depth most congested nets

        Args:
            net_manager: The NetManager instance (provides net_to_sites, etc.)
            logic_coords: Placed logic instance coordinates [N, 2]
            io_coords: Placed IO instance coordinates [M, 2] (optional)
            include_io: Whether to include IO instances in analysis.
            clock_period: Override target clock period (seconds).

        Returns:
            TimingSummary with estimated timing metrics.
        """
        if clock_period is not None:
            self.clock_period = clock_period

        device = logic_coords.device

        # Build coordinate lookup: site_name -> (x, y)
        inst_coords = self._build_coord_lookup(
            net_manager, logic_coords, io_coords, include_io
        )

        # Compute per-net wire delays using HPWL (Manhattan distance)
        net_wire_delays: Dict[str, float] = {}
        net_hpwls: Dict[str, float] = {}
        net_site_counts: Dict[str, int] = {}

        for net_name, sites in net_manager.net_to_sites.items():
            coords_list = []
            for s in sites:
                if s in inst_coords:
                    coords_list.append(inst_coords[s])

            if len(coords_list) < 2:
                continue

            # Compute HPWL (Manhattan bounding box)
            xs = [c[0] for c in coords_list]
            ys = [c[1] for c in coords_list]
            hpwl = (max(xs) - min(xs)) + (max(ys) - min(ys))
            net_hpwls[net_name] = hpwl

            # Estimate wire delay: HPWL * wire_delay_per_site
            wire_delay = hpwl * self.wire_delay_per_site
            net_wire_delays[net_name] = wire_delay
            net_site_counts[net_name] = len(coords_list)

        # Estimate logic depth from net_manager (already computed) or compute
        logic_depth_val = getattr(net_manager, 'logic_depth', 1.0)
        # logic_depth_val is a factor around 1.0; convert to approximate levels
        logic_levels = max(1, int(round(logic_depth_val * 8)))

        # Estimate cell delay per logic level
        avg_cell_delay = (
            self.cell_delays.get('LUT', {}).get('max', 100e-12) +
            self.cell_delays.get('CARRY', {}).get('max', 60e-12)
        ) / 2.0

        # Sort nets by wire delay (longest first) — these are potential critical paths
        sorted_nets = sorted(net_wire_delays.items(), key=lambda x: x[1], reverse=True)

        # --- Critical Path Estimation ---
        # Model: A critical path goes through `logic_levels` stages.
        # Each stage has: 1 cell delay + wire delay of the net driving it.
        # We take the top N nets (where N = logic_levels) as the critical path.
        num_critical_nets = min(logic_levels, len(sorted_nets))
        critical_nets = sorted_nets[:num_critical_nets]

        # Sum delays along estimated critical path
        # Cell delay per stage
        total_cell_delay = logic_levels * avg_cell_delay

        # Wire delay along critical path (sum of top N longest nets)
        total_wire_delay_critical = sum(d for _, d in critical_nets)

        # Clock-to-Q and setup overhead
        total_ff_overhead = self.ff_clk2q_time + self.ff_setup_time

        critical_path_delay = total_cell_delay + total_wire_delay_critical + total_ff_overhead

        # Compute WNS
        wns = self.clock_period - critical_path_delay

        # Compute TNS (sum of negative slacks across all paths)
        # Estimate per-path delay for all significant nets
        tns = 0.0
        num_violations = 0
        all_path_delays = []

        for net_name, wire_delay in sorted_nets:
            # Each net represents a potential path stage
            path_delay = avg_cell_delay + wire_delay + self.ff_overhead_per_stage()
            all_path_delays.append(path_delay)

            slack = self.clock_period - path_delay
            if slack < 0:
                tns += slack
                num_violations += 1

        # Also add some multi-stage paths
        if len(sorted_nets) >= logic_levels:
            for i in range(len(sorted_nets) - logic_levels + 1):
                group = sorted_nets[i:i+logic_levels]
                path_delay = (
                    logic_levels * avg_cell_delay
                    + sum(d for _, d in group)
                    + self.ff_clk2q_time + self.ff_setup_time
                )
                slack = self.clock_period - path_delay
                if slack < 0:
                    tns += slack

        # Compute Fmax
        estimated_period = critical_path_delay
        fmax = 1.0 / estimated_period if estimated_period > 0 else 0.0

        # Top violated nets
        top_violated = []
        for net_name, wire_delay in sorted_nets:
            path_delay = avg_cell_delay + wire_delay + self.ff_overhead_per_stage()
            slack = self.clock_period - path_delay
            if slack < 0:
                top_violated.append((net_name, slack))

        return TimingSummary(
            wns=wns,
            tns=tns,
            fmax=fmax / 1e6,  # Convert to MHz
            estimated_period=estimated_period,
            critical_path_delay=critical_path_delay,
            logic_depth=logic_levels,
            num_paths=len(sorted_nets) + max(0, len(sorted_nets) - logic_levels + 1),
            num_violations=num_violations,
            wire_delays=net_wire_delays,
            top_violated_nets=top_violated[:20],
        )

    # =========================================================================
    # Path-Based Timing Analysis (actual FF-to-FF paths)
    # =========================================================================

    def analyze_path_based(
        self,
        design: Any,
        placer: Any,
        logic_coords: torch.Tensor,
        io_coords: Optional[torch.Tensor] = None,
        include_io: bool = False,
        clock_period: Optional[float] = None,
        max_paths: int = 0,
    ) -> TimingSummary:
        """
        Perform path-based timing analysis by tracing actual FF-to-FF paths
        through the netlist and computing delays from placement coordinates.

        This is more accurate than the heuristic ``analyze_framework`` method
        because it traces real combinational paths and accounts for all cells
        (LUTs, CARRY, MUX) and wire delays along each path.

        Args:
            design: RapidWright Design object (provides getCells(), getNets()).
            placer: FpgaPlacer instance (provides instance name<->id mappings).
            logic_coords: Placed logic instance coordinates [N, 2].
            io_coords: Placed IO instance coordinates [M, 2] (optional).
            include_io: Whether to include IO in analysis.
            clock_period: Override target clock period (seconds).
            max_paths: Maximum number of paths (0 = unlimited).

        Returns:
            TimingSummary with actual path-based timing metrics.
        """
        import time as _time
        _t0 = _time.time()

        if clock_period is not None:
            self.clock_period = clock_period

        # ------------------------------------------------------------------
        # 1. Build site_name -> (x, y) coordinate lookup
        # ------------------------------------------------------------------
        site_to_coord: Dict[str, Tuple[float, float]] = {}
        logic_insts = placer.instances.get('logic')
        if logic_insts is not None:
            for name, inst_id in logic_insts.name_to_id.items():
                if inst_id < len(logic_coords):
                    c = logic_coords[inst_id]
                    site_to_coord[name] = (float(c[0]), float(c[1]))
        if include_io and io_coords is not None:
            io_insts = placer.instances.get('io')
            if io_insts is not None:
                for name, inst_id in io_insts.name_to_id.items():
                    if inst_id < len(io_coords):
                        c = io_coords[inst_id]
                        site_to_coord[name] = (float(c[0]), float(c[1]))
        INFO(f"  [path_timing] coord lookup: {len(site_to_coord)} sites, {_time.time()-_t0:.1f}s")

        # ------------------------------------------------------------------
        # 2. Build logical-level cell connectivity via EDIF netlist
        # ------------------------------------------------------------------
        # Each physical Cell has a logical EDIFCellInst.  EDIF nets connect
        # EDIF cell ports — no MUXes, no SitePIPs.  We classify cells by
        # their type string (FDRE -> FF, LUT6 -> combo, etc.) and trace
        # FF -> LUT -> FF paths at the logical level, then map back to
        # physical placement for wire delay.

        # Step 2a: Build cell name -> (type, site_name) lookup.
        # KEY INSIGHT: Vivado renames cells between EDIF and physical netlists.
        # The EDIF instance name is "DFF_0" but the physical cell name is
        # "DFF_0/Q_reg".  We key by BOTH names so lookups from logical nets
        # (which use EDIF names) and from physical nets both work.
        cell_info: Dict[str, tuple] = {}
        cell_type_counts: Dict[str, int] = {}

        for cell in design.getCells():
            ctype = str(cell.getType())
            cell_type_counts[ctype] = cell_type_counts.get(ctype, 0) + 1
            si = cell.getSiteInst()
            sname = str(si.getName()) if si is not None else ''
            phys_name = str(cell.getName())
            # Always store by physical cell name
            cell_info[phys_name] = (ctype, sname)
            # Also store by EDIF instance name if available
            ec = cell.getEDIFCellInst()
            if ec is not None:
                edif_name = str(ec.getName())
                if edif_name != phys_name:
                    cell_info[edif_name] = (ctype, sname)

        def _ctype(ct: str) -> tuple:
            ct_up = ct.upper()
            is_ff = ct_up.startswith('FD') or 'FF' in ct_up or ct_up in ('RAM32X1S',)
            is_lut = ct_up.startswith('LUT') and len(ct_up) >= 4 and ct_up[3:].isdigit()
            is_carry = 'CARRY' in ct_up
            is_mux = 'MUXF' in ct_up or ct_up in ('MUXCY', 'XORCY')
            return (is_ff, is_lut or is_carry or is_mux)

        n_ff_cells = sum(1 for _, (ct, _) in cell_info.items() if _ctype(ct)[0])
        n_combo_cells = sum(1 for _, (ct, _) in cell_info.items() if _ctype(ct)[1])
        top_types = sorted(cell_type_counts.items(), key=lambda x: -x[1])[:10]
        INFO(f"  [path_timing] {len(cell_info)} cell name keys "
             f"({n_ff_cells} FF, {n_combo_cells} combo), "
             f"top types: {top_types}")

        # Step 2b: Build cell connectivity from PHYSICAL BEL pin -> net
        # For each cell, iterate its BEL pins.  Output pins tell us which
        # nets the cell drives.  Data input pins tell us which nets the
        # cell receives.  This bypasses EDIF entirely.
        #
        # We use: si.getNetFromSiteWire(belPin.getSiteWireName())
        # to find the net connected to a BEL pin.

        cell_out_nets: Dict[str, List[str]] = {}
        net_in_cells: Dict[str, List[tuple]] = {}
        n_checked = 0
        n_has_out = 0

        for cell in design.getCells():
            phys_name = str(cell.getName())
            ctype = str(cell.getType())
            is_ff, is_combo = _ctype(ctype)
            if not (is_ff or is_combo):
                continue
            n_checked += 1
            si = cell.getSiteInst()
            if si is None: continue
            bel = cell.getBEL()
            if bel is None: continue

            for bel_pin in bel.getPins():
                try:
                    sw = str(bel_pin.getSiteWireName())
                except:
                    continue
                if not sw:
                    continue
                try:
                    net = si.getNetFromSiteWire(sw)
                except:
                    continue
                if net is None: continue
                net_name = str(net.getName())
                if net.isClockNet() or net.isVCCNet() or net.isGNDNet():
                    continue
                try:
                    is_out = bool(bel_pin.isOutput())
                except:
                    continue

                if is_out:
                    cell_out_nets.setdefault(phys_name, []).append(net_name)
                    n_has_out += 1
                else:
                    # Skip control pins (CLK, CE, SR, etc.)
                    try:
                        if bool(bel_pin.isClock()) or bool(bel_pin.isEnable()) \
                           or bool(bel_pin.isReset()) or bool(bel_pin.isSet()):
                            continue
                    except:
                        pass
                    net_in_cells.setdefault(net_name, []).append(
                        (phys_name, is_ff, is_combo))

        ff_src_count = sum(1 for c in cell_out_nets
                           if _ctype(cell_info.get(c, ('', ''))[0])[0])
        unique_ff_sinks = set()
        for _, sinks in net_in_cells.items():
            for (snk, is_ff, _) in sinks:
                if is_ff: unique_ff_sinks.add(snk)
        INFO(f"  [path_timing] physical cells: {n_checked} classified, "
             f"{len(cell_out_nets)} drive nets ({ff_src_count} FF src), "
             f"{len(net_in_cells)} driven nets, "
             f"{len(unique_ff_sinks)} unique FF sinks")

        # ------------------------------------------------------------------
        # 3. Build DAG: fanin & fanout maps from physical connectivity
        # ------------------------------------------------------------------
        # fanin[snk_cell]  = [driver_cell, ...]  — all cells feeding snk_cell
        # fanout[src_cell] = [snk_cell, ...]     — all cells src_cell drives
        fanin: Dict[str, List[str]] = {}
        fanout: Dict[str, List[str]] = {}
        for src_cell, net_list in cell_out_nets.items():
            for net_name in net_list:
                for (snk_cell, _, _) in net_in_cells.get(net_name, []):
                    fanin.setdefault(snk_cell, []).append(src_cell)
                    fanout.setdefault(src_cell, []).append(snk_cell)

        # Identify FF sources and FF sinks
        all_ff_cells = {c for c, (ct, _) in cell_info.items() if _ctype(ct)[0]}
        ff_sources = {c for c in all_ff_cells if c in fanout}
        ff_sinks   = {c for c in all_ff_cells if c in fanin}
        combo_cells = {c for c, (ct, _) in cell_info.items()
                       if not _ctype(ct)[0] and (c in fanout or c in fanin)}

        # ------------------------------------------------------------------
        # 4. Topological sort (Kahn's algorithm) on combinational cells
        # ------------------------------------------------------------------
        # in_degree counts only incoming edges *from other combos*
        in_degree: Dict[str, int] = {}
        for c in combo_cells:
            in_degree[c] = sum(1 for d in fanin.get(c, []) if d in combo_cells)

        from collections import deque
        queue = deque(c for c in combo_cells if in_degree.get(c, 0) == 0)
        sorted_combos: List[str] = []

        while queue:
            node = queue.popleft()
            sorted_combos.append(node)
            for snk in fanout.get(node, []):
                if snk in combo_cells:
                    in_degree[snk] -= 1
                    if in_degree[snk] == 0:
                        queue.append(snk)

        # Cycle detection: if some combos were never sorted, break the loop
        if len(sorted_combos) != len(combo_cells):
            cyclic = set(combo_cells) - set(sorted_combos)
            WARNING(f"  [path_timing] cycle detected in {len(cyclic)} combos, "
                    f"appending unsorted at end")
            sorted_combos.extend(cyclic)

        # ------------------------------------------------------------------
        # 5. Cell-type-to-delay helper (same logic as before)
        # ------------------------------------------------------------------
        def _cell_type_to_delay(ctype: str) -> float:
            ct = ctype.upper()
            if ct.startswith('LUT') and ct[3:].isdigit():
                return self.cell_delays.get('LUT', {}).get('max', 100e-12)
            if 'CARRY' in ct:
                return self.cell_delays.get('CARRY', {}).get('max', 80e-12)
            if 'MUX' in ct:
                return self.cell_delays.get('MUX', {}).get('max', 90e-12)
            if 'DSP' in ct:
                return self.cell_delays.get('DSP', {}).get('max', 500e-12)
            if 'RAMB' in ct or 'BRAM' in ct:
                return self.cell_delays.get('BRAM', {}).get('max', 600e-12)
            return 60e-12

        def _wire_delay_between(a: str, b: str) -> float:
            _, site_a = cell_info.get(a, ('', ''))
            _, site_b = cell_info.get(b, ('', ''))
            if site_a in site_to_coord and site_b in site_to_coord:
                x1, y1 = site_to_coord[site_a]
                x2, y2 = site_to_coord[site_b]
                return (abs(x1 - x2) + abs(y1 - y2)) * self.wire_delay_per_site
            return 0.0

        # ------------------------------------------------------------------
        # 6. Propagate Arrival Times (AT) through the DAG
        # ------------------------------------------------------------------
        AT: Dict[str, float] = {}

        # Initialize source FF outputs to clock-to-Q delay
        for ff in ff_sources:
            AT[ff] = self.ff_clk2q_time

        # Initialize remaining cells to 0 (combo cells without AT yet)
        for c in combo_cells:
            AT.setdefault(c, 0.0)

        # Track logic depth (max combos on any path) via topological propagation
        depth: Dict[str, int] = {}
        for ff in ff_sources:
            depth[ff] = 0

        # Process combos in topological order
        for combo in sorted_combos:
            max_arrival = 0.0
            max_depth = 0
            for driver in fanin.get(combo, []):
                driver_at = AT.get(driver, 0.0)
                arrival = driver_at + _wire_delay_between(driver, combo)
                max_arrival = max(max_arrival, arrival)
                max_depth = max(max_depth, depth.get(driver, 0))
            ctype, _ = cell_info.get(combo, ('', ''))
            AT[combo] = max_arrival + _cell_type_to_delay(ctype)
            depth[combo] = max_depth + 1

        # ------------------------------------------------------------------
        # 7. Compute arrival times at FF sink inputs, then slack
        # ------------------------------------------------------------------
        required_time = self.clock_period - self.ff_setup_time

        critical_path_delay = 0.0
        wns = float('inf')
        tns = 0.0
        num_violations = 0
        endpoint_slacks: List[Tuple[str, float]] = []
        logic_levels = 0

        # Track worst path info for the critical path description
        worst_ff_sink = ''
        worst_at = 0.0

        for ff in ff_sinks:
            max_arrival_at_d = 0.0
            longest_driver = ''
            for driver in fanin.get(ff, []):
                driver_at = AT.get(driver, 0.0)
                arrival = driver_at + _wire_delay_between(driver, ff)
                if arrival > max_arrival_at_d:
                    max_arrival_at_d = arrival
                    longest_driver = driver
            if max_arrival_at_d == 0.0:
                continue  # no data path to this FF

            slack = required_time - max_arrival_at_d
            endpoint_slacks.append((ff, slack))
            if slack < wns:
                wns = slack
                worst_ff_sink = ff
                worst_at = max_arrival_at_d

            if slack < 0:
                tns += slack
                num_violations += 1

            if max_arrival_at_d > critical_path_delay:
                critical_path_delay = max_arrival_at_d

            # Logic depth = max combo depth feeding this FF sink
            for driver in fanin.get(ff, []):
                logic_levels = max(logic_levels, depth.get(driver, 0))

        # Also check FF->FF direct connections (no combos in between)
        for ff_src in ff_sources:
            for snk in fanout.get(ff_src, []):
                if snk in ff_sinks:
                    wire_d = _wire_delay_between(ff_src, snk)
                    at_d = AT.get(ff_src, self.ff_clk2q_time) + wire_d
                    slack = required_time - at_d
                    endpoint_slacks.append((snk, slack))
                    if slack < wns:
                        wns = slack
                        worst_ff_sink = snk
                        worst_at = at_d
                    if slack < 0:
                        tns += slack
                        num_violations += 1
                    if at_d > critical_path_delay:
                        critical_path_delay = at_d
                    logic_levels = max(logic_levels, 1)

        # ------------------------------------------------------------------
        # 8. Compute final metrics
        # ------------------------------------------------------------------
        if not ff_sinks and not ff_sources:
            INFO("  [path_timing] No FF sinks/sources found — falling back to heuristic.")
            return self.analyze_framework(
                placer.net_manager, logic_coords, io_coords, include_io, clock_period
            )

        estimated_period = critical_path_delay
        fmax = 1.0 / estimated_period if estimated_period > 0 else 0.0

        # Sort slacks ascending (most negative first) for violated list
        endpoint_slacks.sort(key=lambda x: x[1])
        num_endpoints = len(endpoint_slacks)

        INFO(f"  [path_timing] DAG: {len(combo_cells)} combos, "
             f"{len(ff_sources)} FF src, {len(ff_sinks)} FF snk, "
             f"{len(sorted_combos)} topo-sorted, "
             f"{_time.time()-_t0:.1f}s")
        INFO(f"  [path_timing] critical path: {critical_path_delay*1e9:.3f} ns, "
             f"WNS={wns*1e9:.3f} ns, Fmax={fmax/1e6:.1f} MHz, "
             f"endpoints={num_endpoints}, "
             f"took {_time.time()-_t0:.1f}s")

        return TimingSummary(
            wns=wns,
            tns=tns,
            fmax=fmax / 1e6,  # Convert to MHz
            estimated_period=estimated_period,
            critical_path_delay=critical_path_delay,
            logic_depth=logic_levels,
            num_paths=num_endpoints,
            num_violations=num_violations,
            wire_delays={},
            top_violated_nets=endpoint_slacks[:20],
        )

    def ff_overhead_per_stage(self) -> float:
        """Clock-to-Q + setup time for one flip-flop stage."""
        return self.ff_clk2q_time + self.ff_setup_time

    def _build_coord_lookup(
        self,
        net_manager: Any,
        logic_coords: torch.Tensor,
        io_coords: Optional[torch.Tensor],
        include_io: bool,
    ) -> Dict[str, Tuple[float, float]]:
        """Build site_name -> (x, y) mapping from placement coordinates.

        Uses the net_manager's site-to-name and name-to-site mappings
        which are built during the analyze_nets phase.
        """
        inst_coords: Dict[str, Tuple[float, float]] = {}

        # Map logic instances by looking up the get_site_inst_id_by_name_func
        # which is a bound method on the placer, providing name<->id mappings.
        get_name_by_id = getattr(net_manager, 'get_site_inst_id_by_name_func', None)
        get_id_by_name = getattr(net_manager, 'get_site_inst_id_by_name_func', None)

        # Approach: iterate over all instance names known to the net_manager
        # via site_to_site_connectivity and net_to_sites dictionaries.
        known_names: set = set()
        for net_sites in net_manager.net_to_sites.values():
            known_names.update(net_sites)

        # Build a reverse mapping: we know site names from net_to_sites,
        # but we need their indices. We can infer from the matrix dimensions.
        # Simpler approach: use the placer's internal name_to_id mappings
        # by accessing through the bound function reference.
        try:
            # The get_site_inst_id_by_name_func is typically a bound method
            # of the FpgaPlacer instance. Access the parent to get instance maps.
            placer = get_id_by_name.__self__ if hasattr(get_id_by_name, '__self__') else None
            if placer is not None:
                logic_insts = placer.instances.get('logic')
                if logic_insts is not None:
                    for name, inst_id in logic_insts.name_to_id.items():
                        if inst_id < len(logic_coords):
                            c = logic_coords[inst_id]
                            inst_coords[name] = (float(c[0]), float(c[1]))

                if include_io and io_coords is not None:
                    io_insts = placer.instances.get('io')
                    if io_insts is not None:
                        for name, inst_id in io_insts.name_to_id.items():
                            if inst_id < len(io_coords):
                                c = io_coords[inst_id]
                                inst_coords[name] = (float(c[0]), float(c[1]))
            else:
                # Fallback: use net_to_sites with index-based mapping
                self._build_coord_lookup_fallback(
                    inst_coords, net_manager, logic_coords, io_coords, include_io
                )
        except (AttributeError, IndexError, TypeError) as e:
            # Fallback
            self._build_coord_lookup_fallback(
                inst_coords, net_manager, logic_coords, io_coords, include_io
            )

        return inst_coords

    def _build_coord_lookup_fallback(
        self,
        inst_coords: Dict[str, Tuple[float, float]],
        net_manager: Any,
        logic_coords: torch.Tensor,
        io_coords: Optional[torch.Tensor],
        include_io: bool,
    ):
        """Fallback coordinate lookup using index-based assignment."""
        known_names: list = []
        for net_sites in net_manager.net_to_sites.values():
            for s in net_sites:
                if s not in inst_coords:
                    known_names.append(s)

        # Assign coordinates by index (order-preserving)
        for i, name in enumerate(known_names):
            if i < len(logic_coords):
                c = logic_coords[i]
                inst_coords[name] = (float(c[0]), float(c[1]))

        if include_io and io_coords is not None:
            io_names = []
            for net_sites in net_manager.net_to_sites.values():
                for s in net_sites:
                    if s not in inst_coords and s not in io_names:
                        io_names.append(s)
            for i, name in enumerate(io_names):
                if i < len(io_coords):
                    c = io_coords[i]
                    inst_coords[name] = (float(c[0]), float(c[1]))

    # =========================================================================
    # Vivado Timing Report Parser
    # =========================================================================

    def parse_vivado_report(self, report_path: str) -> Optional[TimingSummary]:
        """
        Parse a Vivado timing_summary.rpt file to extract timing metrics.

        Returns None if the report contains no timing constraints or cannot be parsed.

        Args:
            report_path: Path to the Vivado timing_summary.rpt file.

        Returns:
            TimingSummary with metrics from Vivado, or None if not available.
        """
        if not os.path.exists(report_path):
            return None

        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception:
            return None

        # Check if there are timing constraints
        if "There are no user specified timing constraints" in content:
            return None

        # Parse Design Timing Summary section
        # Pattern: WNS(ns)  TNS(ns)  TNS Failing ...
        timing_section = self._extract_section(content, "Design Timing Summary", "Timing Details")
        if not timing_section:
            return None

        wns = self._parse_float_field(timing_section, "WNS(ns)")
        tns = self._parse_float_field(timing_section, "TNS(ns)")
        whs = self._parse_float_field(timing_section, "WHS(ns)")
        wpws = self._parse_float_field(timing_section, "WPWS(ns)")
        tns_failing = self._parse_int_field(timing_section, "TNS Failing Endpoints")
        tns_total = self._parse_int_field(timing_section, "TNS Total Endpoints")

        # Parse clock period from the Intra Clock Table if available
        clock_period = None
        clock_section = self._extract_section(content, "Intra Clock Table", "Inter Clock Table")
        if clock_section:
            # Look for clock names and their WNS
            lines = clock_section.strip().split('\n')
            for line in lines:
                parts = line.split()
                if len(parts) >= 2 and parts[0] not in ('Clock', '-----', ''):
                    # First column is clock name, second is WNS
                    clock_wns = self._safe_float(parts[1])
                    if clock_wns is not None and clock_wns != 'NA':
                        # Clock period can be inferred if we know the target
                        pass

        # Parse timing paths if available for Fmax estimation
        path_section = self._extract_section(content, "Timing Details", "")
        if path_section and wns is not None and wns != 'NA':
            # Fmax from WNS: Fmax = 1 / (clock_period - WNS) if clock_period known
            # Without clock period, we can't compute Fmax from the report alone
            pass

        if wns is None or wns == 'NA':
            return None

        # Convert WNS, TNS from ns to seconds (Vivado reports in ns)
        wns_sec = float(wns) * 1e-9 if wns != 'NA' else 0.0
        tns_sec = float(tns) * 1e-9 if tns is not None and tns != 'NA' else 0.0

        num_failing = int(tns_failing) if tns_failing is not None and tns_failing != 'NA' else 0
        num_total = int(tns_total) if tns_total is not None and tns_total != 'NA' else 0

        return TimingSummary(
            wns=wns_sec,
            tns=tns_sec,
            fmax=0.0,  # Cannot compute Fmax without clock definition
            estimated_period=0.0,
            critical_path_delay=0.0,
            logic_depth=0,
            num_paths=num_total,
            num_violations=num_failing,
            wire_delays={},
            top_violated_nets=[],
        )

    def parse_vivado_timing_paths(self, report_path: str) -> List[Dict[str, Any]]:
        """
        Parse detailed timing path report for path-specific delays.

        Args:
            report_path: Path to a Vivado timing report (e.g., timing_paths.rpt).

        Returns:
            List of path dicts with slack, delay, source, destination.
        """
        if not os.path.exists(report_path):
            return []

        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception:
            return []

        paths = []
        # Parse individual timing paths (Vivado detailed timing report format)
        # Each path starts with "---" separator and contains "Slack:" and "Delay:"
        path_blocks = re.split(r'-{3,}', content)

        for block in path_blocks:
            if 'Slack' not in block and 'slack' not in block.lower():
                continue

            slack = self._parse_field_value(block, r'Slack\s*:\s*([-\d.]+)')
            delay = self._parse_field_value(block, r'(?:Total|Logic)\s+Delay\s*:\s*([-\d.]+)')
            source = self._parse_field_value(block, r'Source\s*:\s*(\S+)')
            dest = self._parse_field_value(block, r'Destination\s*:\s*(\S+)')

            if slack is not None:
                paths.append({
                    'slack': float(slack),
                    'delay': float(delay) if delay else 0.0,
                    'source': source or '',
                    'destination': dest or '',
                    'clock_period': float(delay) + float(slack) if delay and slack else 0.0,
                })

        return paths

    # =========================================================================
    # Utility methods
    # =========================================================================

    def _extract_section(self, content: str, start_header: str, end_header: str) -> Optional[str]:
        """Extract a section between two headers."""
        start_idx = content.find(start_header)
        if start_idx == -1:
            return None

        if end_header:
            end_idx = content.find(end_header, start_idx + len(start_header))
            if end_idx == -1:
                return content[start_idx:]
            return content[start_idx:end_idx]
        return content[start_idx:]

    def _parse_float_field(self, section: str, field_name: str) -> Optional[str]:
        """Extract a float value field from a table section."""
        # Try to find the field in a table row
        pattern = rf'{re.escape(field_name)}\s+([-\d.]+|NA)'
        match = re.search(pattern, section)
        if match:
            val = match.group(1)
            return val if val != 'NA' else 'NA'
        return None

    def _parse_int_field(self, section: str, field_name: str) -> Optional[str]:
        """Extract an integer value field from a table section."""
        pattern = rf'{re.escape(field_name)}\s+(\d+|NA)'
        match = re.search(pattern, section)
        if match:
            val = match.group(1)
            return val if val != 'NA' else 'NA'
        return None

    def _parse_field_value(self, text: str, pattern: str) -> Optional[str]:
        """Extract a field value using regex."""
        match = re.search(pattern, text)
        return match.group(1) if match else None

    def _safe_float(self, val: str) -> Optional[float]:
        """Safely parse a float, returning None on failure."""
        try:
            return float(val)
        except (ValueError, TypeError):
            return None


# =============================================================================
# Convenience functions
# =============================================================================

def analyze_placement_timing(
    placer: Any,
    logic_coords: torch.Tensor,
    io_coords: Optional[torch.Tensor] = None,
    include_io: bool = False,
    clock_period_ns: float = 5.0,
    mode: str = 'heuristic',
    design: Optional[Any] = None,
    max_paths: int = 0,
) -> TimingSummary:
    """
    Convenience function to analyze timing for a placement result.

    Args:
        placer: An FpgaPlacer instance (provides net_manager).
        logic_coords: Placed logic coordinates [N, 2].
        io_coords: Placed IO coordinates [M, 2] (optional).
        include_io: Whether to include IO in analysis.
        clock_period_ns: Target clock period in nanoseconds.
        mode: ``'heuristic'`` (HPWL-based estimate) or ``'path'`` (actual
            FF-to-FF path tracing; requires ``design``).
        design: RapidWright Design object (required for ``mode='path'``).
        max_paths: Max paths to trace (0 = unlimited).

    Returns:
        TimingSummary with estimated timing metrics.
    """
    analyzer = TimingAnalyzer(clock_period=clock_period_ns * 1e-9)
    if mode == 'path':
        if design is None:
            raise ValueError("'design' argument required for mode='path'")
        return analyzer.analyze_path_based(
            design=design,
            placer=placer,
            logic_coords=logic_coords,
            io_coords=io_coords,
            include_io=include_io,
            clock_period=clock_period_ns * 1e-9,
            max_paths=max_paths,
        )
    return analyzer.analyze_framework(
        net_manager=placer.net_manager,
        logic_coords=logic_coords,
        io_coords=io_coords,
        include_io=include_io,
        clock_period=clock_period_ns * 1e-9,
    )


def parse_vivado_timing(report_path: str) -> Optional[TimingSummary]:
    """
    Convenience function to parse a Vivado timing summary report.

    Args:
        report_path: Path to Vivado timing_summary.rpt

    Returns:
        TimingSummary or None if parsing failed.
    """
    analyzer = TimingAnalyzer()
    return analyzer.parse_vivado_report(report_path)


def analyze_path_based_timing(
    design: Any,
    placer: Any,
    logic_coords: torch.Tensor,
    io_coords: Optional[torch.Tensor] = None,
    include_io: bool = False,
    clock_period_ns: float = 5.0,
    max_paths: int = 0,
) -> TimingSummary:
    """
    Convenience function for path-based (FF-to-FF) timing analysis.

    Args:
        design: RapidWright Design object (provides getCells(), getNets()).
        placer: An FpgaPlacer instance.
        logic_coords: Placed logic coordinates [N, 2].
        io_coords: Placed IO coordinates [M, 2] (optional).
        include_io: Whether to include IO in analysis.
        clock_period_ns: Target clock period in nanoseconds.
        max_paths: Maximum paths to trace (0 = unlimited).

    Returns:
        TimingSummary with actual path-based timing metrics.
    """
    analyzer = TimingAnalyzer(clock_period=clock_period_ns * 1e-9)
    return analyzer.analyze_path_based(
        design=design,
        placer=placer,
        logic_coords=logic_coords,
        io_coords=io_coords,
        include_io=include_io,
        clock_period=clock_period_ns * 1e-9,
        max_paths=max_paths,
    )


# =============================================================================
# Vivado Feedback Mode — map FEM placement back to Vivado for exact timing
# =============================================================================


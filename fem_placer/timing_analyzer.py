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
) -> TimingSummary:
    """
    Convenience function to analyze timing for a placement result.

    Args:
        placer: An FpgaPlacer instance (provides net_manager).
        logic_coords: Placed logic coordinates [N, 2].
        io_coords: Placed IO coordinates [M, 2] (optional).
        include_io: Whether to include IO in analysis.
        clock_period_ns: Target clock period in nanoseconds.

    Returns:
        TimingSummary with estimated timing metrics.
    """
    analyzer = TimingAnalyzer(clock_period=clock_period_ns * 1e-9)
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


def generate_vivado_timing_tcl(
    output_path: str,
    clock_period_ns: float = 5.0,
    clock_name: str = "clk",
    clock_port: str = "clk",
    part_name: str = "xcvu065-ffvc1517-1-i",
) -> str:
    """
    Generate a TCL script to run timing analysis on a post-impl DCP.

    This script can be sourced in Vivado to add clock constraints and
    generate timing reports for a placed-and-routed design.

    Args:
        output_path: Path to write the generated TCL file.
        clock_period_ns: Clock period in nanoseconds.
        clock_name: Name for the created clock.
        clock_port: Clock port name (or net name).
        part_name: Target FPGA part.

    Returns:
        Path to the generated TCL file.
    """
    tcl_content = f"""# Auto-generated timing analysis script
# Usage: vivado -mode batch -source {os.path.basename(output_path)} -tclargs <dcp_file> <output_dir>

if {{ $argc < 2 }} {{
    puts "Usage: vivado -mode batch -source [file tail [info script]] -tclargs <dcp_file> <output_dir>"
    exit 1
}}

set dcp_file [lindex $argv 0]
set output_dir [lindex $argv 1]

file mkdir $output_dir

puts "Opening DCP: $dcp_file"
open_checkpoint $dcp_file

# Create clock constraint
puts "Creating clock: {clock_name} with period {clock_period_ns}ns on port {clock_port}"
create_clock -period {clock_period_ns} -name {clock_name} [get_ports {clock_port}]

# Run timing-driven optimization (optional, comment out if not needed)
# opt_design -directive Explore

# Report timing
puts "Generating timing reports..."
report_timing_summary -file [file join $output_dir timing_summary.rpt] -delay_type min_max
report_timing -max_paths 100 -file [file join $output_dir timing_paths.rpt] -delay_type min_max
report_timing -max_paths 10 -nworst 10 -file [file join $output_dir critical_paths.rpt]
report_utilization -file [file join $output_dir utilization.rpt]

# Extract WNS and TNS for easy parsing
set paths [get_timing_paths -max_paths 1 -nworst 1 -setup]
if {{ [llength $paths] > 0 }} {{
    set wns [get_property SLACK [lindex $paths 0]]
    puts "WNS: $wns ns"
}}

set tns 0.0
foreach path [get_timing_paths -max_paths 1000 -nworst 1000 -setup] {{
    set slack [get_property SLACK $path]
    if {{ $slack < 0 }} {{
        set tns [expr {{$tns + $slack}}]
    }}
}}
puts "TNS: $tns ns"

# Save timing metrics as text for easy parsing
set fp [open [file join $output_dir timing_metrics.txt] w]
puts $fp "WNS: $wns ns"
puts $fp "TNS: $tns ns"
if {{ $wns != "" }} {{
    set period {clock_period_ns}
    set fmax [expr {{1000.0 / ($period - $wns)}}]
    puts $fp "Fmax: $fmax MHz"
}}
close $fp

puts "Timing analysis complete. Results in $output_dir"
quit
"""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(tcl_content)

    return output_path

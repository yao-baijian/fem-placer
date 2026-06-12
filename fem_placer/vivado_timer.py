"""
Vivado Feedback Timer — maps FEM placement back to Vivado for exact timing.
"""

import os
import torch
from typing import Dict, List, Optional, Any
from .logger import INFO, WARNING, ERROR
from .timer import TimingSummary, TimingAnalyzer


# =============================================================================
# Vivado Feedback Mode — map FEM placement back to Vivado for exact timing
# =============================================================================

class VivadoTimingRunner:
    """
    Maps FEM placement results back to the original Vivado design,
    runs routing & timing analysis in Vivado, and reports results.

    Usage::

        runner = VivadoTimingRunner(vivado_path='vivado')
        result = runner.run(
            design=design,
            placer=placer,
            logic_coords=legalized_logic,
            io_coords=legalized_io,
            clock_period_ns=5.0,
            output_dir='./vivado/fem_timing/instance_name',
        )
        print(result.format_report())
    """

    def __init__(
        self,
        vivado_path: str = 'vivado',
        work_dir: str = './vivado/fem_timing',
    ):
        self.vivado_path = vivado_path
        self.work_dir = work_dir

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        design: Any,
        placer: Any,
        logic_coords: torch.Tensor,
        io_coords: Optional[torch.Tensor] = None,
        include_io: bool = False,
        clock_period_ns: float = 5.0,
        output_dir: Optional[str] = None,
        instance_name: str = 'fem_placement',
        run_vivado: bool = True,
        keep_files: bool = True,
    ) -> TimingSummary:
        """
        Full pipeline: write placement → generate Tcl → run Vivado → parse.

        Args:
            design: RapidWright Design (the original DCP already loaded).
            placer: FpgaPlacer instance.
            logic_coords: FEM legalized logic coordinates [N, 2].
            io_coords: FEM legalized IO coordinates [M, 2].
            include_io: Whether IO coordinates are provided.
            clock_period_ns: Target clock period in ns.
            output_dir: Output directory for all generated files.
            instance_name: Name for this placement run.
            run_vivado: If True, launches Vivado in batch mode.
            keep_files: If True, keeps intermediate files (Tcl, DCP).

        Returns:
            TimingSummary with Vivado's exact timing results.
        """
        import time as _time

        if output_dir is None:
            output_dir = os.path.join(self.work_dir, instance_name)

        os.makedirs(output_dir, exist_ok=True)

        # --- Step 1: Map FEM coords → device sites → write constraint Tcl ---
        tcl_path = os.path.abspath(os.path.join(output_dir, 'place_and_route.tcl')).replace('\\', '/')
        placed_dcp = os.path.abspath(os.path.join(output_dir, 'fem_placed.dcp')).replace('\\', '/')
        metrics_txt = os.path.abspath(os.path.join(output_dir, 'timing_metrics.txt')).replace('\\', '/')
        impl_dcp = os.path.abspath(os.path.join(output_dir, 'post_route.dcp')).replace('\\', '/')
        timing_rpt = os.path.abspath(os.path.join(output_dir, 'timing_summary.rpt')).replace('\\', '/')

        t0 = _time.time()
        self._write_placement_tcl(
            design=design,
            placer=placer,
            logic_coords=logic_coords,
            io_coords=io_coords,
            include_io=include_io,
            tcl_path=tcl_path,
            placed_dcp=placed_dcp,
            impl_dcp=impl_dcp,
            timing_rpt=timing_rpt,
            metrics_txt=metrics_txt,
            clock_period_ns=clock_period_ns,
        )
        INFO(f"[VivadoTimingRunner] Placement Tcl written to {tcl_path} "
             f"({_time.time()-t0:.1f}s)")

        # --- Step 2: Write a simple DCP copy for Vivado to work with ---
        if not os.path.exists(placed_dcp):
            t0 = _time.time()
            design.writeCheckpoint(placed_dcp)
            INFO(f"[VivadoTimingRunner] Placed DCP written to {placed_dcp} "
                 f"({_time.time()-t0:.1f}s)")

        # --- Step 3: Run Vivado in batch mode ---
        if run_vivado:
            t0 = _time.time()
            retcode = self._run_vivado(tcl_path, output_dir)
            INFO(f"[VivadoTimingRunner] Vivado finished (ret={retcode}) "
                 f"in {_time.time()-t0:.1f}s")

        # --- Step 4: Parse results ---
        result = self._parse_results(metrics_txt, timing_rpt, clock_period_ns)

        # --- Step 5: Clean up ---
        if not keep_files:
            for p in [tcl_path, placed_dcp]:
                if os.path.exists(p):
                    os.remove(p)

        return result

    # ------------------------------------------------------------------
    # Step 1: Map FEM coordinates to device sites and write Tcl
    # ------------------------------------------------------------------

    def _write_placement_tcl(
        self,
        design: Any,
        placer: Any,
        logic_coords: torch.Tensor,
        io_coords: Optional[torch.Tensor],
        include_io: bool,
        tcl_path: str,
        placed_dcp: str,
        impl_dcp: str,
        timing_rpt: str,
        metrics_txt: str,
        clock_period_ns: float,
    ):
        """
        Map each logic instance from its FEM grid coordinate to the
        corresponding physical SLICE site on the device, then emit a
        Vivado Tcl script that places, routes, and runs timing.
        """

        # ----------------------------------------------------------
        # Build coordinate-to-site mapping — use indexed lookup
        # instead of coordinate matching (FEM grid coords don't
        # correspond to absolute device coordinates).
        # ----------------------------------------------------------
        site_insts = placer.instances.get('sites')
        if site_insts is None or len(site_insts) == 0:
            ERROR("[VivadoTimingRunner] No target sites available — "
                  "did you call get_available_target_sites()?")
            return

        # Sort available sites in row-major order to match FEM grid
        sorted_sites = sorted(site_insts.insts,
                              key=lambda s: (s.getInstanceY(), s.getInstanceX()))

        logic_insts = placer.instances.get('logic')
        if logic_insts is None:
            return

        # Cell types that cannot be placed on generic SLICEL/SLICEM
        # (e.g. SRL16E needs SLICEM, others may need specialised sites)
        NON_SLICE_SRL_TYPES = {'SRL16E', 'SRLC32E'}
        NON_SLICE_RAM_TYPES = {'RAM32X1S', 'RAM64X1S', 'RAM128X1S', 'RAM256X1S',
                                'RAM32M', 'RAM64M', 'RAMB18E1', 'RAMB36E1',
                                'DSP48E1', 'DSP48E2'}
        NON_SLICE_TYPES = NON_SLICE_SRL_TYPES | NON_SLICE_RAM_TYPES

        # ----------------------------------------------------------
        # Map FEM-placed sites to physical SLICE sites 1:1.
        #
        # sorted_sites only contains SLICEL/SLICEM (no DSP/BRAM gaps),
        # so a flat index mapping naturally skips non-SLICE columns.
        #
        # We start from the middle of the available region to give
        # Vivado more routing flexibility versus starting at a corner.
        #
        # CRITICAL: Every primitive cell inside the original SiteInst
        # (FFs, LUTs, CARRY, MUXF7, etc.) is moved together as a block
        # to the same target site. This preserves intra-SLICE BEL
        # relationships and prevents control-set (CE Difference) errors.
        # ----------------------------------------------------------
        num_src = len(logic_insts)
        num_dst = len(sorted_sites)
        offset = max(0, (num_dst - num_src) // 2)

        cell_placements: List[str] = []
        n_mapped = 0
        total_cell_commands = 0
        n_skipped_cells = 0

        for i, (site_inst, fem_coord) in enumerate(
            zip(logic_insts.insts, logic_coords)
        ):
            dst_idx = offset + i
            if dst_idx >= num_dst:
                WARNING(f"[VivadoTimingRunner] No more target sites for instance "
                        f"{site_inst.getName()} — ran out at dst_idx={dst_idx}/{num_dst}")
                continue
            target = sorted_sites[dst_idx]
            site_name = site_inst.getName()
            target_site_name = target.getName()

            # Use the RapidWright API to get ALL primitive cells on this
            # original SiteInst, then move them as a family to the target.
            try:
                cells_on_site = list(site_inst.getCells())
            except Exception:
                cells_on_site = []

            if not cells_on_site:
                WARNING(f"[VivadoTimingRunner] No cells found on site {site_name}")
                continue

            for cell in cells_on_site:
                cname = str(cell.getName())
                ctype = str(cell.getType())

                if ctype in NON_SLICE_TYPES:
                    WARNING(f"[VivadoTimingRunner] Skipping {cname} (type={ctype}) "
                            f"— requires specialised site")
                    n_skipped_cells += 1
                    continue
                cell_placements.append(
                    f"place_cell {{ {cname} {target_site_name} }}"
                )
                total_cell_commands += 1
            n_mapped += 1

        INFO(f"[VivadoTimingRunner] Mapped {n_mapped}/{len(logic_insts)} sites "
             f"→ {total_cell_commands} place_cell commands "
             f"({n_skipped_cells} cells skipped — non-SLICE types)")

        # ----------------------------------------------------------
        # Generate the Tcl script
        # ----------------------------------------------------------
        # Try to find a clock port name via the top module's ports
        clock_port = 'clk'
        try:
            top_mod = design.getTopModule()
            if top_mod is not None:
                for port in top_mod.getPorts():
                    pname = str(port.getName()).lower()
                    if any(kw in pname for kw in ['clk', 'clock', 'ck']):
                        clock_port = str(port.getName())
                        break
        except Exception:
            pass

        cell_place_str = '\n'.join(cell_placements)
        tcl = f"""# Auto-generated by VivadoTimingRunner
# Places FEM-optimized placement, routes, and reports timing

puts "========================================"
puts "FEM Placement Feedback — Place & Route"
puts "========================================"

# --- Open the DCP (placed by RapidWright) ---
puts "Opening checkpoint: {placed_dcp}"
open_checkpoint {placed_dcp}

# --- Unplace all primitive cells so FEM placement is clean ---
puts "Unplacing all primitive cells..."
unplace_cell [get_cells -hierarchical -filter {{IS_PRIMITIVE}}]

# --- Place cells at FEM-optimized sites ---
puts "Placing {total_cell_commands} cells at FEM-optimized locations..."
{cell_place_str}

# --- Fix placed cells, then legalize BELs with place_design ---
# place_cell preserves BEL assignments from the original site, but those
# may conflict at the target site (e.g. MUXF7 internal routing).
# We fix the site first so place_design only touches BEL-level placement.
set fixed_cells [get_cells -hierarchical -filter {{IS_BEL == 1 && PRIMITIVE_LEVEL == LEAF}}]
set_property IS_LOC_FIXED 1 $fixed_cells
puts "Running place_design to legalize BEL-level placement..."
place_design -directive Quick
set_property IS_LOC_FIXED 0 $fixed_cells

# --- Route ---
puts "Routing..."
route_design

# --- Save checkpoint ---
write_checkpoint -force {impl_dcp}

# --- Create clock ---
puts "Creating clock with period {clock_period_ns} ns on port {clock_port}"
create_clock -period {clock_period_ns} -name sys_clk [get_ports {clock_port}]

# --- Timing reports ---
puts "Reporting timing..."
report_timing_summary -file {timing_rpt} -delay_type min_max
report_timing -max_paths 10 -nworst 10 -file [file join [file dirname {timing_rpt}] critical_paths.rpt] -setup
report_route_status -file [file join [file dirname {timing_rpt}] route_status.rpt]

# --- Extract WNS / TNS for easy parsing ---
set paths [get_timing_paths -max_paths 1 -nworst 1 -setup]
if {{ [llength $paths] > 0 }} {{
    set wns [get_property SLACK [lindex $paths 0]]
    puts "WNS: $wns ns"
}} else {{
    set wns "N/A"
}}

set tns 0.0
foreach path [get_timing_paths -max_paths 1000 -nworst 1000 -setup] {{
    set slack [get_property SLACK $path]
    if {{ $slack < 0 }} {{
        set tns [expr {{$tns + $slack}}]
    }}
}}
puts "TNS: $tns ns"

set fp [open {metrics_txt} w]
puts $fp "WNS: $wns"
puts $fp "TNS: $tns"
if {{ $wns != "N/A" }} {{
    set period {clock_period_ns}
    set fmax [expr {{1000.0 / ($period - $wns)}}]
    puts $fp "Fmax (MHz): $fmax"
}}
close $fp

puts "Done. Results in [file dirname {timing_rpt}]"
quit
"""
        with open(tcl_path, 'w') as f:
            f.write(tcl)

    # ------------------------------------------------------------------
    # Step 3: Run Vivado
    # ------------------------------------------------------------------

    def _run_vivado(self, tcl_path: str, output_dir: str) -> int:
        """Launch Vivado in batch mode to execute the Tcl script."""
        import subprocess
        cmd = [
            self.vivado_path,
            '-mode', 'batch',
            '-source', tcl_path,
        ]
        INFO(f"[VivadoTimingRunner] Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            shell=True,
            timeout=3600,
        )
        # Save log
        log_path = os.path.abspath(os.path.join(output_dir, 'vivado_feedback.log')).replace('\\', '/')
        with open(log_path, 'w') as f:
            f.write(result.stdout)
            f.write('\n--- STDERR ---\n')
            f.write(result.stderr)

        if result.returncode != 0:
            WARNING(f"[VivadoTimingRunner] Vivado returned {result.returncode}. "
                    f"See {log_path} for details.")
        return result.returncode

    # ------------------------------------------------------------------
    # Step 4: Parse results
    # ------------------------------------------------------------------

    def _parse_results(
        self,
        metrics_txt: str,
        timing_rpt: str,
        clock_period_ns: float,
    ) -> TimingSummary:
        """Parse Vivado output into a TimingSummary."""

        # Try metrics.txt first (structured output)
        wns = 'N/A'
        tns = 'N/A'
        fmax = 'N/A'
        if os.path.exists(metrics_txt):
            try:
                with open(metrics_txt, 'r') as f:
                    for line in f:
                        if ':' in line:
                            k, v = line.split(':', 1)
                            k = k.strip()
                            v = v.strip()
                            if k == 'WNS':
                                wns = v
                            elif k == 'TNS':
                                tns = v
                            elif 'Fmax' in k or 'fmax' in k:
                                fmax = v
            except Exception:
                pass

        # Fallback: parse the timing report
        if wns == 'N/A' and os.path.exists(timing_rpt):
            analyzer = TimingAnalyzer()
            ts = analyzer.parse_vivado_report(timing_rpt)
            if ts is not None:
                wns = f"{ts.wns*1e9:.3f}"
                tns = f"{ts.tns*1e9:.3f}"

        def _safe_float(s: str, default: float = 0.0) -> float:
            try:
                return float(s)
            except (ValueError, TypeError):
                return default

        wns_f = _safe_float(wns)
        tns_f = _safe_float(tns)
        fmax_f = _safe_float(fmax)

        # Compute Fmax from WNS if not directly available
        if fmax_f <= 0 and wns_f != 0:
            est_period = max(clock_period_ns - wns_f, 0.01)
            fmax_f = 1000.0 / est_period

        return TimingSummary(
            wns=wns_f * 1e-9,
            tns=tns_f * 1e-9,
            fmax=fmax_f,
            estimated_period=(clock_period_ns - wns_f) * 1e-9,
            critical_path_delay=(clock_period_ns - wns_f) * 1e-9,
            logic_depth=0,
            num_paths=0,
            num_violations=1 if wns_f < 0 else 0,
            wire_delays={},
            top_violated_nets=[],
        )


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

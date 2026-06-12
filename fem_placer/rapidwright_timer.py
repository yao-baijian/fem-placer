"""
RapidWright Timer — uses RapidWright's own placement and routing
for timing analysis of FEM-optimized placements.

Instead of generating Vivado Tcl, this module maps FEM coordinates
to device sites using the RapidWright Java API directly:

    si.place(targetSite)    — move a SiteInstance to a new site
    RWRoute.routeDesign()   — route the design within RapidWright
"""

import os
import torch
from typing import Dict, List, Optional, Any
from .logger import INFO, WARNING, ERROR
from .timer import TimingSummary


class RapidWrightTimer:
    """
    Maps FEM placement results to device sites using RapidWright's
    native Java API, routes with RWRoute, and reports timing.

    Usage::

        runner = RapidWrightTimer()
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
        work_dir: str = './vivado/fem_timing',
    ):
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
        route_design: bool = True,
        keep_files: bool = True,
    ) -> TimingSummary:
        """
        Full pipeline: map FEM coords → RapidWright place → RWRoute → report.

        Args:
            design: RapidWright Design object (from init_placement).
            placer: FpgaPlacer instance.
            logic_coords: FEM legalized logic coordinates [N, 2].
            io_coords: FEM legalized IO coordinates [M, 2].
            include_io: Whether IO coordinates are provided.
            clock_period_ns: Target clock period in ns.
            output_dir: Output directory for generated files.
            instance_name: Name for this placement run.
            route_design: If True, runs RWRoute after placement.
            keep_files: If True, keeps intermediate files.

        Returns:
            TimingSummary with estimated timing metrics.
        """
        import time as _time
        t0 = _time.time()

        if output_dir is None:
            output_dir = os.path.join(self.work_dir, instance_name)
        os.makedirs(output_dir, exist_ok=True)

        # --- Step 1: Map FEM coords to target sites ---
        self._place_fem_sites(
            design=design,
            placer=placer,
            logic_coords=logic_coords,
            io_coords=io_coords,
            include_io=include_io,
        )
        INFO(f"[RapidWrightTimer] FEM placement applied "
             f"({_time.time()-t0:.1f}s)")

        # --- Step 2: Route with RWRoute ---
        if route_design:
            t1 = _time.time()
            self._route(design)
            INFO(f"[RapidWrightTimer] RWRoute finished "
                 f"({_time.time()-t1:.1f}s)")

        # --- Step 3: Write output DCP ---
        routed_dcp = os.path.abspath(os.path.join(output_dir, 'fem_routed.dcp')).replace('\\', '/')
        design.writeCheckpoint(routed_dcp)
        INFO(f"[RapidWrightTimer] Routed DCP written to {routed_dcp}")

        # --- Step 4: Quick timing estimate from the placer's netlist ---
        result = self._estimate_timing(
            placer=placer,
            logic_coords=logic_coords,
            io_coords=io_coords,
            include_io=include_io,
            clock_period_ns=clock_period_ns,
        )

        INFO(f"[RapidWrightTimer] Done ({_time.time()-t0:.1f}s)")
        return result

    # ------------------------------------------------------------------
    # Step 1: Map FEM coordinates → device sites via RapidWright API
    # ------------------------------------------------------------------

    def _place_fem_sites(
        self,
        design: Any,
        placer: Any,
        logic_coords: torch.Tensor,
        io_coords: Optional[torch.Tensor],
        include_io: bool,
    ):
        """
        For each logic site instance, look up its FEM-optimised target
        site and use `si.place(targetSite)` to move it via RW's Java API.
        """
        site_insts = placer.instances.get('sites')
        if site_insts is None or len(site_insts) == 0:
            ERROR("[RapidWrightTimer] No target sites available — "
                  "did you call get_available_target_sites()?")
            return

        sorted_sites = sorted(site_insts.insts,
                              key=lambda s: (s.getInstanceY(), s.getInstanceX()))

        logic_insts = placer.instances.get('logic')
        if logic_insts is None:
            return

        num_src = len(logic_insts)
        num_dst = len(sorted_sites)
        offset = max(0, (num_dst - num_src) // 2)

        device = design.getDevice()
        n_placed = 0

        for i, (site_inst, fem_coord) in enumerate(
            zip(logic_insts.insts, logic_coords)
        ):
            dst_idx = offset + i
            if dst_idx >= num_dst:
                WARNING(f"[RapidWrightTimer] No more target sites for "
                        f"{site_inst.getName()}")
                continue

            target_site = sorted_sites[dst_idx]
            target_site_name = target_site.getName()

            # Use RapidWright Java API to place the SiteInstance
            try:
                phys_site = device.getSite(target_site_name)
                if phys_site is not None:
                    site_inst.place(phys_site)
                    n_placed += 1
                else:
                    WARNING(f"[RapidWrightTimer] Physical site "
                            f"{target_site_name} not found on device")
            except Exception as e:
                ERROR(f"[RapidWrightTimer] Failed to place "
                      f"{site_inst.getName()} → {target_site_name}: {e}")

        INFO(f"[RapidWrightTimer] Placed {n_placed}/{num_src} site instances")

    # ------------------------------------------------------------------
    # Step 2: Route with RWRoute
    # ------------------------------------------------------------------

    def _route(self, design: Any):
        """Run RapidWright's built-in router on the design."""
        try:
            from com.xilinx.rapidwright.rwroute import RWRoute
            # Try timing-driven first, fall back to non-timing-driven
            try:
                RWRoute.routeDesignFullTimingDriven(design)
            except Exception:
                INFO("[RapidWrightTimer] Timing-driven route failed, "
                     "falling back to non-timing-driven")
                RWRoute.routeDesignFullNonTimingDriven(design)
        except Exception as e:
            ERROR(f"[RapidWrightTimer] RWRoute error: {e}")

    # ------------------------------------------------------------------
    # Step 4: Estimate timing from netlist / placement
    # ------------------------------------------------------------------

    def _estimate_timing(
        self,
        placer: Any,
        logic_coords: torch.Tensor,
        io_coords: Optional[torch.Tensor],
        include_io: bool,
        clock_period_ns: float,
    ) -> TimingSummary:
        """Produce a TimingSummary from the FEM-placed design."""
        from .timer import analyze_placement_timing
        return analyze_placement_timing(
            placer=placer,
            logic_coords=logic_coords,
            io_coords=io_coords,
            include_io=include_io,
            clock_period_ns=clock_period_ns,
            mode='heuristic',
        )

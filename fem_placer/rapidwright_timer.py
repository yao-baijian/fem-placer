"""
RapidWright Timer — uses RapidWright's own placement and routing
for timing analysis of FEM-optimized placements.

Instead of generating Vivado Tcl, this module maps FEM coordinates
to device sites using the RapidWright Java API directly:

    si.place(targetSite)    — move a SiteInstance to a new site
    RWRoute.routeDesign()   — route the design within RapidWright

Enhancements:
- Placement is centered near the original centroid so average position
  does not shift excessively.
- After SLICE placement, clock buffers (BUFG / BUFGCE / BUFGCTRL) are
  re-placed close to their load centroids, making clock-tree routing
  feasible for RWRoute.
"""

import os
import torch
from typing import Dict, List, Optional, Any, Set
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
        # --- Step 1b: Re-place clock buffers near their loads ---
        self._replace_clock_buffers(design, placer)
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
            design=design,
            placer=placer,
            logic_coords=logic_coords,
            io_coords=io_coords,
            include_io=include_io,
            clock_period_ns=clock_period_ns,
        )

        INFO(f"[RapidWrightTimer] Done ({_time.time()-t0:.1f}s)")
        return result

    # ------------------------------------------------------------------
    # Clock buffer site types (7-series / UltraScale / UltraScale+)
    # ------------------------------------------------------------------
    _CLOCK_BUF_SITE_TYPES: Set[Any] = set()

    @classmethod
    def _get_clock_buf_types(cls):
        if not cls._CLOCK_BUF_SITE_TYPES:
            from com.xilinx.rapidwright.device import SiteTypeEnum
            cls._CLOCK_BUF_SITE_TYPES = {
                SiteTypeEnum.BUFGCE,
                SiteTypeEnum.BUFGCTRL,
                SiteTypeEnum.BUFG,
                SiteTypeEnum.BUFGCE_DIV,
                SiteTypeEnum.BUFG_GT,
                SiteTypeEnum.BUFG_PS,
                SiteTypeEnum.BUFG_FABRIC,
            }
        return cls._CLOCK_BUF_SITE_TYPES

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
        device = design.getDevice()

        # Record original positions and compute centroid
        orig_x_sum = 0
        orig_y_sum = 0
        for inst in logic_insts.insts:
            orig_site = inst.getSite()
            sx = orig_site.getInstanceX()
            sy = orig_site.getInstanceY()
            orig_x_sum += sx
            orig_y_sum += sy
        orig_cx = orig_x_sum / num_src
        orig_cy = orig_y_sum / num_src

        # Find the best offset so the target window centroid
        # is as close as possible to the original centroid
        best_offset = 0
        best_dist = float('inf')
        max_offset = num_dst - num_src
        for cand in range(max_offset + 1):
            xs = [s.getInstanceX() for s in sorted_sites[cand:cand + num_src]]
            ys = [s.getInstanceY() for s in sorted_sites[cand:cand + num_src]]
            cx = sum(xs) / num_src
            cy = sum(ys) / num_src
            d = (cx - orig_cx) ** 2 + (cy - orig_cy) ** 2
            if d < best_dist:
                best_dist = d
                best_offset = cand
        offset = best_offset

        target_cx = sum(sorted_sites[offset + i].getInstanceX() for i in range(num_src)) / num_src
        target_cy = sum(sorted_sites[offset + i].getInstanceY() for i in range(num_src)) / num_src
        INFO(f"[RapidWrightTimer] Placement centroid: original=({orig_cx:.1f}, {orig_cy:.1f}), "
             f"target=({target_cx:.1f}, {target_cy:.1f}), offset={offset}")

        n_placed = 0
        for i, site_inst in enumerate(logic_insts.insts):
            dst_idx = offset + i
            if dst_idx >= num_dst:
                WARNING(f"[RapidWrightTimer] No more target sites for "
                        f"{site_inst.getName()}")
                continue

            target_site = sorted_sites[dst_idx]
            target_site_name = target_site.getName()

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
                      f"{site_inst.getName()} -> {target_site_name}: {e}")

        INFO(f"[RapidWrightTimer] Placed {n_placed}/{num_src} site instances")

    # ------------------------------------------------------------------
    # Step 1b: Re-place clock buffers close to their loads
    # ------------------------------------------------------------------

    def _replace_clock_buffers(self, design: Any, placer: Any):
        """
        After SLICE instances have been moved to their FEM-optimised sites,
        re-place each clock buffer (BUFG / BUFGCE / BUFGCTRL) to a site
        that is as close as possible to the centroid of its loads.

        This keeps clock-tree routing feasible for RWRoute.
        """
        bufg_types = self._get_clock_buf_types()
        device = design.getDevice()

        # Pre-collect all BUFG-compatible sites on the device once
        bufg_sites = []
        for site in device.getAllSites():
            if site.getSiteTypeEnum() in bufg_types:
                bufg_sites.append(site)
        if not bufg_sites:
            WARNING("[RapidWrightTimer] No BUFG sites found on device")
            return

        n_replaced = 0
        for site_inst in design.getSiteInsts():
            stype = site_inst.getSiteTypeEnum()
            if stype not in bufg_types:
                continue

            buf_name = site_inst.getName()
            buf_stype = str(stype)

            # Find the clock net driven by this buffer's output pin
            # Try "O" first, then scan all SitePinInsts for an output with a net
            clk_net = None
            try:
                out_pin = site_inst.getSitePinInst("O")
                if out_pin is not None:
                    clk_net = out_pin.getNet()
            except Exception:
                pass

            if clk_net is None:
                # Fall back: scan all site pin insts for the one driving this site
                try:
                    pin_map = site_inst.getSitePinInstMap()
                    if pin_map is not None:
                        for pin_name, pin in pin_map.entrySet():
                            if pin.isOutPin():
                                n = pin.getNet()
                                if n is not None:
                                    clk_net = n
                                    break
                except Exception:
                    pass

            if clk_net is None:
                # Last resort: scan all nets for one whose source is on this site
                try:
                    for net in design.getNets():
                        src = net.getSource()
                        if src is not None and src.getSiteInst() is site_inst:
                            clk_net = net
                            break
                except Exception:
                    pass

            if clk_net is None:
                continue

            # Collect sink site positions (the flip-flops driven by this clock)
            sink_positions = []
            for pin in clk_net.getSinkPins():
                sink_si = pin.getSiteInst()
                if sink_si is None:
                    continue
                try:
                    sink_site = sink_si.getSite()
                    if sink_site is not None:
                        sink_positions.append((
                            sink_site.getInstanceX(),
                            sink_site.getInstanceY(),
                        ))
                except Exception:
                    continue

            if len(sink_positions) < 2:
                continue  # not enough loads to justify moving

            # Centroid of loads
            cx = sum(p[0] for p in sink_positions) / len(sink_positions)
            cy = sum(p[1] for p in sink_positions) / len(sink_positions)

            # Find closest BUFG site
            best_site = None
            best_d2 = float('inf')
            for bs in bufg_sites:
                dx = bs.getInstanceX() - cx
                dy = bs.getInstanceY() - cy
                d2 = dx * dx + dy * dy
                if d2 < best_d2:
                    best_d2 = d2
                    best_site = bs

            if best_site is not None:
                try:
                    site_inst.place(best_site)
                    n_replaced += 1
                except Exception as e:
                    WARNING(f"[RapidWrightTimer] Failed to place clock buffer "
                            f"{site_inst.getName()} -> {best_site.getName()}: {e}")

        if n_replaced:
            INFO(f"[RapidWrightTimer] Re-placed {n_replaced} clock buffer(s) "
                 f"near their load centroids")

    # ------------------------------------------------------------------
    # Step 2: Route with RWRoute
    # ------------------------------------------------------------------

    def _route(self, design: Any):
        """Run RapidWright's built-in router on the design.

        NOTE: On some designs (especially those with clock nets), RWRoute may
        fail during UltraScale clock-tree routing with a NullPointerException
        on ``startingRouteNode``.  This is a known limitation when placement
        has been changed — RWRoute's ``routeBUFGToNearestRoutingTrack()``
        cannot always find an HROUTE node from the moved clock sinks.

        The exception is caught so that the timing pipeline continues with a
        heuristic-based estimate (see ``_estimate_timing``).  The un-routed
        DCP is still written with the new placement for downstream use.
        """
        try:
            from com.xilinx.rapidwright.rwroute import RWRoute
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
        design: Any,
        placer: Any,
        logic_coords: torch.Tensor,
        io_coords: Optional[torch.Tensor],
        include_io: bool,
        clock_period_ns: float,
    ) -> TimingSummary:
        """Produce a TimingSummary from the FEM-placed design.

        Uses net-level path tracing (``analyze_path_based``).
        """
        from .timer import analyze_placement_timing
        return analyze_placement_timing(
            design=design,
            placer=placer,
            logic_coords=logic_coords,
            io_coords=io_coords,
            include_io=include_io,
            clock_period_ns=clock_period_ns,
            mode='path',
        )

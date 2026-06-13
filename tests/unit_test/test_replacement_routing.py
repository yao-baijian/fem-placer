"""
Test: random / greedy site replacement + RWRoute routing on a DCP.

Loads a Vivado-placed DCP (s15850), randomly or greedily moves SLICE and
clock-buffer site instances to nearby compatible sites, then tries
RWRoute to see whether clock-tree routing succeeds or fails.
"""

import sys, os, time, random

# ── project root ──────────────────────────────────────────────────
HERE = os.path.dirname(__file__)
PROJ = os.path.normpath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, PROJ)
os.chdir(PROJ)

from fem_placer.config import SLICE_SITE_ENUM
from fem_placer.logger import INFO, WARNING, ERROR, SET_LEVEL
SET_LEVEL("INFO")

# RapidWright Java imports
from com.xilinx.rapidwright.design import Design
from com.xilinx.rapidwright.device import SiteTypeEnum
from com.xilinx.rapidwright.rwroute import RWRoute

# ── config ────────────────────────────────────────────────────────
DCP = "./vivado/output_dir/s15850/post_impl.dcp"
RNG_SEED = 42
MAX_MOVES = 100              # max sites to move (0 = all)
STRATEGY = "greedy"          # "random" | "greedy"

# ── clock-buffer site types ───────────────────────────────────────
CLOCK_BUF_TYPES = {
    SiteTypeEnum.BUFGCE,
    SiteTypeEnum.BUFGCTRL,
    SiteTypeEnum.BUFG,
    SiteTypeEnum.BUFGCE_DIV,
    SiteTypeEnum.BUFG_GT,
    SiteTypeEnum.BUFG_PS,
    SiteTypeEnum.BUFG_FABRIC,
}

# ── helpers ───────────────────────────────────────────────────────

def site_key(site):
    """Sort key: (Y, X) for column-major traversal."""
    return (site.getInstanceY(), site.getInstanceX())


def compatible_type(stype):
    """Return True if *stype* is SLICEL or SLICEM (i.e. a CLB site)."""
    return stype in SLICE_SITE_ENUM


def load_dcp(path):
    INFO(f"Loading DCP: {path}")
    design = Design.readCheckpoint(path)
    INFO(f"  Device: {design.getDevice().getName()}")
    INFO(f"  SiteInsts: {len(list(design.getSiteInsts()))}")
    INFO(f"  Nets: {len(list(design.getNets()))}")
    return design


def classify_sites(design):
    """Return (clb_site_insts, clk_buf_site_insts, all_other_site_insts)
    where each is a list of SiteInst objects.
    """
    clb = []
    clk = []
    other = []
    for si in design.getSiteInsts():
        t = si.getSiteTypeEnum()
        if t in SLICE_SITE_ENUM:
            clb.append(si)
        elif t in CLOCK_BUF_TYPES:
            clk.append(si)
        else:
            other.append(si)
    INFO(f"  CLB site insts: {len(clb)}")
    INFO(f"  Clock-buf site insts: {len(clk)}")
    INFO(f"  Other site insts: {len(other)}")
    return clb, clk, other


def collect_compatible_targets(design, stype_set):
    """Return list of device *Site* objects whose type is in *stype_set*."""
    targets = []
    dev = design.getDevice()
    for site in dev.getAllSites():
        if site.getSiteTypeEnum() in stype_set:
            targets.append(site)
    targets.sort(key=site_key)
    return targets


def random_replacement(site_insts, target_sites, max_moves=0):
    """Randomly move *site_insts* to random sites in *target_sites*.
    If *max_moves* > 0, only that many are moved (randomly chosen).
    Returns number of successfully moved instances.
    """
    to_move = list(site_insts)
    random.shuffle(to_move)
    if max_moves > 0 and max_moves < len(to_move):
        to_move = to_move[:max_moves]

    n_ok = 0
    for si in to_move:
        dst = random.choice(target_sites)
        try:
            si.place(dst)
            n_ok += 1
        except Exception as e:
            WARNING(f"  place() failed for {si.getName()} -> {dst.getName()}: {e}")
    return n_ok


def greedy_replacement(site_insts, target_sites, max_moves=0):
    """Move each site instance to the closest target site (by manhattan
    distance of the site instance's *original* position).  If *max_moves*
    > 0, only that many are moved (closest-first).
    """
    pairs = []
    for si in site_insts:
        try:
            orig = si.getSite()
            ox, oy = orig.getInstanceX(), orig.getInstanceY()
        except Exception:
            continue
        # find nearest target
        best_d2 = float('inf')
        best_s = None
        for ts in target_sites:
            dx = ts.getInstanceX() - ox
            dy = ts.getInstanceY() - oy
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best_s = ts
        pairs.append((best_d2, si, best_s))
    pairs.sort(key=lambda x: x[0])   # closest first

    if max_moves > 0 and max_moves < len(pairs):
        pairs = pairs[:max_moves]

    n_ok = 0
    for _, si, dst in pairs:
        if dst is None:
            continue
        try:
            si.place(dst)
            n_ok += 1
        except Exception as e:
            WARNING(f"  place() failed for {si.getName()} -> {dst.getName()}: {e}")
    return n_ok


def try_rwroute(design, label="routing"):
    """Call RWRoute (timing-driven first, fallback to non-timing-driven).
    Returns True on success, False on failure.
    """
    try:
        RWRoute.routeDesignFullTimingDriven(design)
        INFO(f"  [{label}] Timing-driven RWRoute succeeded")
        return True
    except Exception as e1:
        INFO(f"  [{label}] Timing-driven RWRoute failed: {e1}")
        try:
            RWRoute.routeDesignFullNonTimingDriven(design)
            INFO(f"  [{label}] Non-timing-driven RWRoute succeeded")
            return True
        except Exception as e2:
            ERROR(f"  [{label}] Non-timing-driven RWRoute also failed: {e2}")
            return False


def print_placement_spread(site_insts, tag=""):
    """Log min/max/avg X and Y of a set of placed site instances."""
    xs, ys = [], []
    for si in site_insts:
        try:
            s = si.getSite()
            xs.append(s.getInstanceX())
            ys.append(s.getInstanceY())
        except Exception:
            continue
    if xs:
        INFO(f"  {tag} X: min={min(xs)} max={max(xs)} avg={sum(xs)/len(xs):.1f}")
        INFO(f"  {tag} Y: min={min(ys)} max={max(ys)} avg={sum(ys)/len(ys):.1f}")


# ══════════════════════════════════════════════════════════════════
#  main
# ══════════════════════════════════════════════════════════════════

def main():
    random.seed(RNG_SEED)

    # 1. Load DCP
    design = load_dcp(DCP)

    # 2. Classify
    clb_insts, clk_insts, _ = classify_sites(design)

    # ── Collect target sites ──────────────────────────────────────
    slice_targets = collect_compatible_targets(design, set(SLICE_SITE_ENUM))
    clk_targets   = collect_compatible_targets(design, CLOCK_BUF_TYPES)
    INFO(f"Available slice targets on device: {len(slice_targets)}")
    INFO(f"Available clock-buf targets on device: {len(clk_targets)}")

    # 3. Print original spread
    print_placement_spread(clb_insts,  "Original CLB")
    print_placement_spread(clk_insts,  "Original clock-buf")

    # 4. Try routing *before* any changes (baseline)
    INFO("─" * 60)
    INFO("Step A — RWRoute on original placement (baseline)")
    baseline_ok = try_rwroute(design, "baseline")

    # Reload design for a clean state after the baseline attempt
    design = load_dcp(DCP)
    clb_insts, clk_insts, _ = classify_sites(design)

    # 5. Replace CLB sites
    INFO("─" * 60)
    if STRATEGY == "random":
        INFO(f"Step B — Randomly replacing up to {MAX_MOVES} CLB sites …")
        n = random_replacement(clb_insts, slice_targets, MAX_MOVES)
    else:
        INFO(f"Step B — Greedily replacing up to {MAX_MOVES} CLB sites …")
        n = greedy_replacement(clb_insts, slice_targets, MAX_MOVES)
    INFO(f"  Moved {n} CLB site instances")
    print_placement_spread(clb_insts, "After CLB move")

    # 6. Replace clock-buf sites (if any)
    if clk_insts:
        INFO("─" * 60)
        if STRATEGY == "random":
            INFO(f"Step C — Randomly replacing up to {MAX_MOVES} clock-buf sites …")
            m = random_replacement(clk_insts, clk_targets, MAX_MOVES)
        else:
            INFO(f"Step C — Greedily replacing up to {MAX_MOVES} clock-buf sites …")
            m = greedy_replacement(clk_insts, clk_targets, MAX_MOVES)
        INFO(f"  Moved {m} clock-buf site instances")
        print_placement_spread(clk_insts, "After clock-buf move")

    # 7. Try routing again
    INFO("─" * 60)
    INFO("Step D — RWRoute after replacement")
    after_ok = try_rwroute(design, "after-replacement")

    # 8. Summary
    INFO("═" * 60)
    INFO(f"Baseline routing:     {'OK' if baseline_ok else 'FAILED'}")
    INFO(f"After-replacement:    {'OK' if after_ok else 'FAILED'}")
    INFO(f"Strategy:             {STRATEGY}")
    INFO(f"Max moves (per type): {MAX_MOVES}")
    INFO("═" * 60)


if __name__ == "__main__":
    main()

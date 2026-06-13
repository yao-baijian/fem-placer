"""
Unit test for placer info extraction.

Initialises the placer on the s15850 benchmark and extracts
connectivity statistics.  The test passes if it prints results
without raising an exception.
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import warnings
warnings.filterwarnings("ignore", message="Trying to unpickle estimator.*")

from fem_placer import FpgaPlacer
from fem_placer.logger import SET_LEVEL
from fem_placer.config import PlaceType, GridType, IoMode
from tests.utils import TestConfig


def _load_default_config() -> TestConfig:
    """Load the default regression config (s15850, logic+io)."""
    cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'config', 'default_config.json',
    )
    with open(cfg_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    raw = {k: v for k, v in raw.items() if not k.startswith('_')}

    from tests.utils import _ENUM_MAP
    for key, (enum_cls, default_name) in _ENUM_MAP.items():
        if key in raw:
            raw[key] = enum_cls[raw[key]]

    return TestConfig(**{k: v for k, v in raw.items() if k in TestConfig.__dataclass_fields__})


def run_extract_placer_info() -> dict:
    """Run placer initialisation and return connectivity stats for s15850."""
    SET_LEVEL('WARNING')

    cfg = _load_default_config()
    instance = cfg.instances[0]

    placer = FpgaPlacer(cfg)
    placer.set_instance_name(instance)

    dcp_path = f"./vivado/output_dir/{instance}/post_impl.dcp"
    pl_path = f"./vivado/output_dir/{instance}/optimized_placement.pl"
    placer.init_placement(dcp_path, pl_path)

    placer.set_alpha(30)
    placer.set_beta(30)

    # Connectivity statistics
    net_sizes = [len(sites) for sites in placer.net_manager.net_to_sites.values()]
    stats = {
        'instance': instance,
        'logic_inst': placer.instances['logic'].num,
        'io_inst': placer.instances['io'].num,
        'total_nets': len(placer.net_manager.nets),
        'min_conn': min(net_sizes) if net_sizes else 0,
        'max_conn': max(net_sizes) if net_sizes else 0,
        'avg_conn': sum(net_sizes) / len(net_sizes) if net_sizes else 0.0,
    }

    # Print result table — test succeeds if this prints without error
    print()
    print("=" * 70)
    print(f"{'Instance':<20} | {'Logic':<8} | {'IO':<6} | {'Nets':<6} | "
          f"{'Min':<6} | {'Max':<6} | {'Avg':<8}")
    print("-" * 70)
    print(f"{stats['instance']:<20} | {stats['logic_inst']:<8} | "
          f"{stats['io_inst']:<6} | {stats['total_nets']:<6} | "
          f"{stats['min_conn']:<6} | {stats['max_conn']:<6} | "
          f"{stats['avg_conn']:<8.2f}")
    print("=" * 70)
    print()

    return stats


def test_extract_placer_info():
    """Placer initialisation and info extraction must complete without error."""
    stats = run_extract_placer_info()
    assert stats['logic_inst'] > 0, f"No logic instances found: {stats}"
    assert stats['io_inst'] > 0, f"No IO instances found: {stats}"
    assert stats['total_nets'] > 0, f"No nets found: {stats}"
    # Print success confirmation
    print(f"OK — {stats['instance']}: {stats['logic_inst']} logic, "
          f"{stats['io_inst']} io, {stats['total_nets']} nets")


if __name__ == "__main__":
    run_extract_placer_info()
    print("PASSED")
    sys.exit(0)

"""
Standalone Timing Analysis Script.

Analyzes timing for FPGA placement results from the FEM framework.
Can also parse Vivado timing reports for comparison.

Usage:
    # Analyze FEM placement result
    python scripts/analyze_timing.py --instance c2670 --coords result/c2670/placement_coords.pt

    # Parse Vivado timing report
    python scripts/analyze_timing.py --instance c2670 --vivado-report vivado/output_dir/c2670/timing_summary.rpt

    # Full analysis (FEM + Vivado)
    python scripts/analyze_timing.py --instance c2670 --coords result/c2670/placement_coords.pt \\
        --vivado-report vivado/output_dir/c2670/timing_summary.rpt

    # Compare multiple instances
    python scripts/analyze_timing.py --all --vivado-dir vivado/output_dir
"""

import sys
import os
import argparse
import json
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fem_placer.timing_analyzer import (
    TimingAnalyzer,
    TimingSummary,
    analyze_placement_timing,
    parse_vivado_timing,
)
from fem_placer.logger import SET_LEVEL, INFO, WARNING


def analyze_instance_framework(
    instance_name: str,
    result_dir: str = 'result',
    clock_period_ns: float = 5.0,
):
    """
    Analyze timing for a specific instance using saved placement data.

    This requires that init_params.json was previously saved.
    """
    params_file = os.path.join(result_dir, instance_name, 'init_params.json')
    coords_file = os.path.join(result_dir, instance_name, 'placement_coords.pt')
    io_coords_file = os.path.join(result_dir, instance_name, 'placement_io_coords.pt')

    if not os.path.exists(params_file):
        WARNING(f"init_params.json not found for {instance_name} at {params_file}")
        WARNING("Run a placement test first to generate placement results.")
        return None

    # Load placement coordinates
    if not os.path.exists(coords_file):
        WARNING(f"Placement coordinates not found at {coords_file}")
        return None

    logic_coords = torch.load(coords_file, map_location='cpu')

    io_coords = None
    if os.path.exists(io_coords_file):
        io_coords = torch.load(io_coords_file, map_location='cpu')

    # Load parameters to reconstruct placer context
    with open(params_file, 'r') as f:
        data = json.load(f)

    params = data['params']

    # For framework timing analysis, we need the net_manager topology.
    # If we don't have a full placer object, we use a simplified analysis
    # based on wirelength estimation from the coupling matrix.

    # Try to reconstruct coupling matrix
    coupling_matrix = None
    if 'coupling_matrix' in data.get('tensors', {}):
        coupling_matrix = torch.tensor(
            data['tensors']['coupling_matrix'],
            dtype=torch.float32,
        )

    analyzer = TimingAnalyzer(clock_period=clock_period_ns * 1e-9)

    # Simplified analysis using coupling matrix directly
    # Estimate wire delays from placement coordinates
    num_inst = logic_coords.shape[0]
    if coupling_matrix is not None:
        # Compute pairwise distances for connected instances
        connected_pairs = torch.nonzero(coupling_matrix > 0)
        wire_delays = {}
        total_wire_delay = 0.0
        max_wire_delay = 0.0

        for idx in range(connected_pairs.shape[0]):
            i, j = connected_pairs[idx][0].item(), connected_pairs[idx][1].item()
            if i < j and i < num_inst and j < num_inst:
                dist = torch.sum(torch.abs(logic_coords[i] - logic_coords[j])).item()
                delay = dist * analyzer.wire_delay_per_site
                wire_delays[f"net_{i}_{j}"] = delay
                total_wire_delay += delay
                max_wire_delay = max(max_wire_delay, delay)

        # Estimate logic depth
        logic_depth = params.get('logic_depth', 8)
        avg_cell_delay = 80e-12  # ~80 ps per LUT

        # Critical path estimate
        sorted_delays = sorted(wire_delays.values(), reverse=True)
        num_critical = min(int(logic_depth), len(sorted_delays))
        critical_wire_delay = sum(sorted_delays[:num_critical]) if sorted_delays else 0
        total_cell_delay = logic_depth * avg_cell_delay
        ff_overhead = analyzer.ff_clk2q_time + analyzer.ff_setup_time

        critical_path_delay = total_cell_delay + critical_wire_delay + ff_overhead
        clock_period = clock_period_ns * 1e-9
        wns = clock_period - critical_path_delay

        # TNS estimate
        tns = 0.0
        num_violations = 0
        for delay in sorted_delays:
            path_delay = avg_cell_delay + delay + ff_overhead
            slack = clock_period - path_delay
            if slack < 0:
                tns += slack
                num_violations += 1

        fmax = 1.0 / critical_path_delay if critical_path_delay > 0 else 0.0

        top_violated = []
        for name, delay in sorted(wire_delays.items(), key=lambda x: x[1], reverse=True)[:20]:
            slack = clock_period - (avg_cell_delay + delay + ff_overhead)
            if slack < 0:
                top_violated.append((name, slack))

        result = TimingSummary(
            wns=wns,
            tns=tns,
            fmax=fmax / 1e6,
            estimated_period=critical_path_delay,
            critical_path_delay=critical_path_delay,
            logic_depth=int(logic_depth),
            num_paths=len(wire_delays),
            num_violations=num_violations,
            wire_delays=wire_delays,
            top_violated_nets=top_violated,
        )
        return result

    return None


def analyze_all_instances(result_dir: str = 'result', vivado_dir: str = 'vivado/output_dir'):
    """Analyze timing for all instances with available data."""
    results = []

    # Scan result directory for instances
    if os.path.exists(result_dir):
        for inst_name in sorted(os.listdir(result_dir)):
            inst_path = os.path.join(result_dir, inst_name)
            if not os.path.isdir(inst_path):
                continue

            timing = analyze_instance_framework(inst_name, result_dir)
            if timing is not None:
                # Check Vivado timing
                vvd_wns = 'N/A'
                vvd_fmax = 'N/A'
                vvd_path = os.path.join(vivado_dir, inst_name, 'timing_metrics.txt')
                if os.path.exists(vvd_path):
                    try:
                        with open(vvd_path, 'r') as f:
                            for line in f:
                                if 'WNS:' in line:
                                    vvd_wns = line.split(':', 1)[1].strip()
                                if 'Fmax' in line:
                                    vvd_fmax = line.split(':', 1)[1].strip().replace(' MHz', '')
                    except Exception:
                        pass

                results.append({
                    'instance': inst_name,
                    'fem_wns_ns': f"{timing.wns*1e9:.3f}",
                    'fem_fmax_mhz': f"{timing.fmax:.1f}",
                    'fem_logic_depth': timing.logic_depth,
                    'vivado_wns': vvd_wns,
                    'vivado_fmax': vvd_fmax,
                })

    # Print results table
    if results:
        print()
        print("=" * 90)
        print("  Timing Analysis Summary (all instances)")
        print("=" * 90)
        print(f"{'Instance':<20} {'FEM WNS(ns)':<16} {'FEM Fmax(MHz)':<16} {'Logic Depth':<12} {'Vivado WNS':<14} {'Vivado Fmax':<14}")
        print("-" * 90)
        for r in results:
            print(f"{r['instance']:<20} {r['fem_wns_ns']:<16} {r['fem_fmax_mhz']:<16} {r['fem_logic_depth']:<12} {r['vivado_wns']:<14} {r['vivado_fmax']:<14}")
        print("=" * 90)
    else:
        print("No timing results available.")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Timing Analysis for FPGA Placement"
    )
    parser.add_argument('--instance', type=str, default=None,
                        help='Instance name to analyze')
    parser.add_argument('--coords', type=str, default=None,
                        help='Path to placement coordinates .pt file')
    parser.add_argument('--io-coords', type=str, default=None,
                        help='Path to IO placement coordinates .pt file')
    parser.add_argument('--vivado-report', type=str, default=None,
                        help='Path to Vivado timing_summary.rpt')
    parser.add_argument('--vivado-dir', type=str, default='vivado/output_dir',
                        help='Vivado output directory')
    parser.add_argument('--result-dir', type=str, default='result',
                        help='Result directory with placement data')
    parser.add_argument('--clock-period', type=float, default=5.0,
                        help='Target clock period in ns')
    parser.add_argument('--all', action='store_true',
                        help='Analyze all available instances')
    parser.add_argument('--detailed', action='store_true',
                        help='Print detailed timing report')
    parser.add_argument('--save', type=str, default=None,
                        help='Save timing results to JSON file')

    args = parser.parse_args()

    if args.all:
        results = analyze_all_instances(args.result_dir, args.vivado_dir)
        if args.save and results:
            with open(args.save, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.save}")
        return

    if args.instance is not None:
        # Analyze from saved parameters
        timing = analyze_instance_framework(
            args.instance, args.result_dir, args.clock_period
        )
        if timing is not None:
            print(timing.format_report())

    # Parse Vivado report if requested
    if args.vivado_report:
        vvd_timing = parse_vivado_timing(args.vivado_report)
        if vvd_timing is not None:
            print("\n" + "=" * 65)
            print("  Vivado Timing Report")
            print("=" * 65)
            print(f"  WNS: {vvd_timing.wns*1e9:.3f} ns")
            print(f"  TNS: {vvd_timing.tns*1e9:.3f} ns")
            print(f"  Violating Paths: {vvd_timing.num_violations} / {vvd_timing.num_paths}")
        else:
            print(f"\nNo valid timing data in {args.vivado_report}")
            print("(This is expected if no clock constraint was defined during Vivado run)")

    # Analyze from explicit coordinate files
    if args.coords is not None and os.path.exists(args.coords):
        logic_coords = torch.load(args.coords, map_location='cpu')
        io_coords = None
        if args.io_coords and os.path.exists(args.io_coords):
            io_coords = torch.load(args.io_coords, map_location='cpu')

        # Without net_manager context, use simplified analysis
        analyzer = TimingAnalyzer(clock_period=args.clock_period * 1e-9)
        print(f"\nPlacement dimensions: {logic_coords.shape}")
        print(f"  X range: [{logic_coords[:, 0].min():.1f}, {logic_coords[:, 0].max():.1f}]")
        print(f"  Y range: [{logic_coords[:, 1].min():.1f}, {logic_coords[:, 1].max():.1f}]")
        print(f"  Average wire delay per site: {analyzer.wire_delay_per_site*1e12:.2f} ps")
        print("\n(Full timing analysis requires netlist topology from init_params.json)")


if __name__ == '__main__':
    main()

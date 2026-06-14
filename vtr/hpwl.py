#!/usr/bin/env python3
"""
Compute Half-Perimeter Wirelength (HPWL) from VPR placement and XML netlist.
Automatically distinguishes nets that involve top-level I/O ports using
the __top_input__ / __top_output__ markers present in the XML netlist.

Includes cross-platform normalization metrics (Footprint and Grid Perimeter Scaling)
with a robust XML-attribute inspection parser to completely exclude physical 
top-level I/O pad instances from structural normalizations.
"""

import argparse
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
import os
import math

def parse_place(filename):
    """Parse VPR .place file -> dict {block_name: (x, y)} and grid dimensions."""
    coords = {}
    max_x = max_y = 0
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            name = parts[0]
            try:
                x = int(parts[1])
                y = int(parts[2])
            except ValueError:
                continue
            coords[name] = (x, y)
            if x > max_x: max_x = x
            if y > max_y: max_y = y
    width = max_x + 1
    height = max_y + 1
    return coords, width, height

def parse_netlist_xml(filename):
    """
    Parse VPR XML .net file -> 
    Returns:
      - nets: dict net_name -> set of block names
      - io_blocks: set of block names explicitly identified as IO pads
    """
    tree = ET.parse(filename)
    root = tree.getroot()
    nets = defaultdict(set)
    io_blocks = set()

    def add_signals(signal_str, block_name):
        if not signal_str:
            return
        for sig in signal_str.split():
            sig = sig.strip()
            if sig and sig != "open":
                nets[sig].add(block_name)

    # Track pseudo boundaries
    for tag, pseudo in [('inputs', '__top_input__'),
                        ('outputs', '__top_output__'),
                        ('clocks', '__top_clock__')]:
        elem = root.find(tag)
        if elem is not None and elem.text:
            for sig in elem.text.split():
                sig = sig.strip()
                if sig:
                    nets[sig].add(pseudo)

    # Inspect structural blocks and flag physical IO blocks
    for block in root.findall('block'):
        block_name = block.get('name')
        block_type = block.get('instance', '')
        
        if block_name is None:
            continue
            
        # If the block instance starts with 'io[' or is classified as an IO hardware macro, flag it
        if block_type.startswith('io[') or 'io' in block_type.lower():
            io_blocks.add(block_name)
            
        for port in block.findall('.//port'):
            if port.text:
                add_signals(port.text, block_name)
                
    return nets, io_blocks

def compute_hpwl(coords, nets, io_blocks, width, height, exclude_io=False):
    """
    Compute Site-Level HPWL statistics.
    Filters out blocks identified as IO pads from the placement canvas totals.
    """
    total_all = 0
    lengths_all = []
    missing_blocks = set()
    nets_with_io = 0
    
    total_pins_all = 0
    total_pins_minus_one_all = 0
    num_logic_nets = 0

    # Define valid instances based on the strict IO block classification set
    if exclude_io:
        valid_instances = {b for b in coords if b not in io_blocks}
    else:
        valid_instances = set(coords.keys())

    for net_name, blocks in nets.items():
        xs, ys = [], []
        has_io = False
        unique_sites = set()
        
        # Check if the net touches an abstract port or an identified physical IO block
        for blk in blocks:
            if blk.startswith('__top__') or blk in io_blocks:
                has_io = True
                if exclude_io:
                    continue
            
            if blk.startswith('__top_'):
                has_io = True
                if exclude_io:
                    continue
                continue

            if exclude_io and blk in io_blocks:
                continue

            if blk in coords:
                x, y = coords[blk]
                xs.append(x)
                ys.append(y)
                unique_sites.add((x, y))
            elif not blk.startswith('__top_'):
                missing_blocks.add(blk)

        unmapped_io = 1 if (has_io and not exclude_io) else 0
        physical_pin_count = len(unique_sites) + unmapped_io

        if physical_pin_count > 0:
            total_pins_all += physical_pin_count
            total_pins_minus_one_all += (physical_pin_count - 1)

        if has_io:
            nets_with_io += 1
        else:
            if physical_pin_count >= 2:
                num_logic_nets += 1

        if physical_pin_count < 2:
            lengths_all.append(0)
            continue

        # BOUNDARY INFERENCE
        if len(xs) == 1 and has_io and not exclude_io:
            lx, ly = xs[0], ys[0]
            io_x = 0 if lx < width / 2 else (width - 1)
            io_y = 0 if ly < height / 2 else (height - 1)
            xs.append(io_x)
            ys.append(io_y)

        if len(xs) < 2:
            lengths_all.append(0)
            continue

        hpwl = (max(xs) - min(xs)) + (max(ys) - min(ys))
        total_all += hpwl
        lengths_all.append(hpwl)

    return {
        'total_all': total_all,
        'per_net_all': lengths_all,
        'num_nets_all': len([l for l in lengths_all if l > 0]),
        'num_nets_logic': num_logic_nets,
        'nets_with_io': nets_with_io,
        'missing_blocks': missing_blocks,
        'total_pins_all': total_pins_all,
        'total_pins_minus_one_all': total_pins_minus_one_all,
        'logic_instances_count': len(valid_instances)
    }

def write_report(f, place_file, net_file, coords, width, height, nets, results, exclude_io):
    """Write all output to the given file handle."""
    f.write(f"HPWL Report generated from:\n")
    f.write(f"  Placement: {place_file}\n")
    f.write(f"  Netlist:   {net_file}\n")
    f.write(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"  Mode:      {'IO Excluded' if exclude_io else 'Standard Inclusion'}\n\n")

    f.write("--- Placement Grid ---\n")
    f.write(f"Grid dimensions: {width} x {height} (area = {width*height} sites)\n")
    f.write(f"Number of total placed blocks: {len(coords)}\n")
    f.write(f"Number of logic-only blocks: {results['logic_instances_count']}\n\n")

    f.write("--- HPWL Results ---\n")
    f.write(f"Total HPWL evaluated: {results['total_all']}\n")
    f.write(f"Number of active scaling nets: {results['num_nets_all']}\n")
    
    print_instances = results['logic_instances_count']
    if print_instances > 0:
        grid_side_proxy = math.sqrt(print_instances)
        norm_footprint = results['total_all'] / (print_instances * grid_side_proxy)
        f.write(f"Method 1: Footprint-Scaled HPWL: {norm_footprint:.6f}\n")
    
    max_perimeter = width + height
    if max_perimeter > 0:
        norm_perimeter = results['total_all'] / max_perimeter
        f.write(f"Method 2: Grid-Perimeter Normalized HPWL: {norm_perimeter:.6f}\n")

def main():
    parser = argparse.ArgumentParser(description='Compute HPWL from VPR placement and XML netlist.')
    parser.add_argument('-s', '--src-dir', required=True, help='Source directory root containing circuit subfolders')
    parser.add_argument('-c', '--circuit', required=True, help='Circuit name / subfolder match identifier')
    parser.add_argument('-o', '--output-dir', default='.', help='Output tracking report file directory')
    parser.add_argument('--exclude-io', action='store_true', help='Exclude I/O pad elements from scaling normalizations')
    args = parser.parse_args()

    target_dir = os.path.join(args.src_dir, args.circuit)
    place_file = os.path.join(target_dir, f"{args.circuit}.place")
    net_file = os.path.join(target_dir, f"{args.circuit}.net")

    if not os.path.exists(place_file) or not os.path.exists(net_file):
        print(f"Error: Missing placement or netlist components in {target_dir}", file=sys.stderr)
        sys.exit(1)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_filename = f"vtr_pr_{args.circuit}_{timestamp}.txt"
    out_path = os.path.join(args.output_dir, out_filename)

    coords, width, height = parse_place(place_file)
    nets, io_blocks = parse_netlist_xml(net_file)
    
    results = compute_hpwl(coords, nets, io_blocks, width, height, exclude_io=args.exclude_io)

    with open(out_path, 'w') as f:
        write_report(f, place_file, net_file, coords, width, height, nets, results, args.exclude_io)

    num_instances = results['logic_instances_count']
    num_nets = results['num_nets_logic'] if args.exclude_io else len(nets)
    raw_hpwl = results['total_all']
    
    grid_side_proxy = math.sqrt(num_instances) if num_instances > 0 else 1.0
    norm_footprint = raw_hpwl / (num_instances * grid_side_proxy) if num_instances > 0 else 0.0
    max_perimeter = width + height
    norm_perimeter = raw_hpwl / max_perimeter if max_perimeter > 0 else 0.0

    print("\n" + "="*112)
    mode_str = "CIRCUIT CROSS-PLATFORM SUMMARY (IO EXCLUDED)" if args.exclude_io else "CIRCUIT CROSS-PLATFORM SUMMARY"
    print(f" {mode_str:^110}")
    print("="*112)
    header = f"| {'Circuit Name':<16} | {'Instances':<10} | {'Nets':<10} | {'Raw HPWL':<12} | {'Norm Footprint':<16} | {'Norm Perimeter':<16} |"
    print(header)
    print("|" + "-"*18 + "|" + "-"*12 + "|" + "-"*12 + "|" + "-"*14 + "|" + "-"*18 + "|" + "-"*18 + "|")
    row = f"| {args.circuit:<16} | {num_instances:<10} | {num_nets:<10} | {raw_hpwl:<12} | {norm_footprint:<16.6f} | {norm_perimeter:<16.6f} |"
    print(row)
    print("="*112 + "\n")

if __name__ == "__main__":
    main()
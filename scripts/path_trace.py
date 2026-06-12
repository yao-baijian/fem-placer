"""
Path Tracing from Verilog Netlist.

Traces FF-to-FF paths directly from the Verilog source file, independent of
RapidWright. This provides ground-truth path information for debugging the
timing analyzer's BEL-level path tracing.

Usage:
    python scripts/path_trace.py benchmarks/ISCAS89/s15850.v
    python scripts/path_trace.py benchmarks/vtr/bgm.v
"""

import re
import sys
import os
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple, Optional


# =============================================================================
# Verilog Netlist Parser
# =============================================================================

FF_MODULE_PREFIXES = ('dff', 'DFF', 'FD', 'FDRE', 'FDCE', 'FDPE', 'FDSE')

# Combinational gate types (ISCAS style)
GATE_TYPES = {
    'and', 'nand', 'or', 'nor', 'xor', 'xnor', 'not', 'buf',
    'AND', 'NAND', 'OR', 'NOR', 'XOR', 'XNOR', 'NOT', 'BUF',
}

# LUT-like primitives (VTR style)
LUT_TYPES = {'lut', 'LUT', 'LUT1', 'LUT2', 'LUT3', 'LUT4', 'LUT5', 'LUT6'}


def parse_verilog(filepath: str) -> dict:
    """
    Parse a Verilog file and return:
      - module_name: str
      - ff_insts: dict   output_wire -> inst_name
      - gate_insts: dict  output_wire -> (gate_type, [input_wires])
      - top_ports: set of port wire names
      - wires: set of all wire names
    """
    with open(filepath, 'r') as f:
        content = f.read()

    # Remove comments
    content = re.sub(r'//.*', '', content)
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)

    # Extract module name
    m = re.search(r'module\s+(\w+)', content)
    module_name = m.group(1) if m else 'unknown'

    # Extract port list
    port_match = re.search(r'module\s+\w+\s*\((.*?)\);', content, re.DOTALL)
    top_ports: Set[str] = set()
    if port_match:
        port_text = port_match.group(1)
        for p in re.findall(r'(\w+)', port_text):
            top_ports.add(p)

    # Extract wire declarations
    wire_set: Set[str] = set(top_ports)
    for w in re.findall(r'\bwire\s+([^;]+);', content):
        for name in re.findall(r'(\w+)', w):
            wire_set.add(name)

    # Extract all instance instantiations
    # Pattern: <module_name> <inst_name>(<pin_connections>);
    # Or: <module_name> #(...) <inst_name>(<pin_connections>);

    # Remove parameter assignments #(...) for easier parsing
    content_no_params = re.sub(r'#\s*\([^)]*\)', '', content)

    inst_pattern = re.compile(
        r'(\w+)\s+(\w+)\s*\(\s*((?:(?!\s*\w+\s*\().)*?)\s*\)\s*;',
        re.DOTALL
    )

    ff_insts: Dict[str, str] = {}  # output_wire -> inst_name
    gate_insts: Dict[str, Tuple[str, List[str]]] = {}  # output_wire -> (type, [inputs])

    # Find all module instantiations (non-primitive)
    # ISCAS style: <type> <name>(<in1>, <in2>, ..., <out>);
    # where FF: dff DFF_0(CK, Q, D);  gate: and AND_0(I1, I2, O);

    # Try ISCAS89 format first: <type> <name>(<args>);
    iscas_inst = re.finditer(r'(\w+)\s+(\w+)\s*\(([^;]*?)\)\s*;', content)

    for match in iscas_inst:
        cell_type = match.group(1)
        inst_name = match.group(2)
        args_str = match.group(3)

        # Parse arguments (comma-separated, possibly with .PORT(expr) format)
        if '(' not in args_str:
            # Positional arguments: ISCAS style
            args = [a.strip() for a in args_str.split(',') if a.strip()]
        else:
            # Named port connections: .PORT(expr)
            args = []
            for nm in re.finditer(r'\.(\w+)\s*\(([^)]*)\)', args_str):
                args.append(nm.group(2).strip())

        if not args:
            continue

        # Classify cell type
        if cell_type in GATE_TYPES or cell_type in LUT_TYPES or cell_type.startswith('LUT'):
            # Combinational: last arg is output, rest are inputs
            output_wire = args[-1]
            input_wires = args[:-1]
            gate_insts[output_wire] = (cell_type, input_wires)

        elif cell_type in ('dff', 'DFF') or cell_type.startswith('FD'):
            # Flip-flop: CK, Q, D  (or .CK(x), .Q(y), .D(z))
            q_wire = ''
            d_wire = ''
            if '(' not in args_str:
                # Positional: CK, Q, D
                if len(args) >= 3:
                    d_wire = args[2].strip()
                    q_wire = args[1].strip()
            else:
                # Named ports
                for nm in re.finditer(r'\.(\w+)\s*\(([^)]*)\)', args_str):
                    pname = nm.group(1).upper()
                    pval = nm.group(2).strip()
                    if pname == 'Q':
                        q_wire = pval
                    elif pname == 'D':
                        d_wire = pval
                    elif pname in ('CK', 'CLK'):
                        pass  # ignore clock
            if q_wire and d_wire:
                ff_insts[q_wire] = inst_name
                # The D input is connected to some net — we'll track it
                wire_set.add(q_wire)
                wire_set.add(d_wire)
        elif cell_type == 'not':
            # NOT: positional args are (input, output) in ISCAS
            output_wire = args[-1]
            input_wires = args[:-1]
            gate_insts[output_wire] = (cell_type, input_wires)

        # Add wires from arguments
        for a in args:
            a_clean = a.strip()
            if a_clean and re.match(r'^\w+$', a_clean):
                wire_set.add(a_clean)

    return {
        'module_name': module_name,
        'ff_insts': ff_insts,
        'gate_insts': gate_insts,
        'top_ports': top_ports,
        'wires': wire_set,
    }


# =============================================================================
# Path Tracing
# =============================================================================

def trace_ff_to_ff_paths(netlist: dict) -> List[List[str]]:
    """
    Trace all FF-to-FF paths through combinational logic.

    Returns a list of paths, where each path is [src_ff_output, ..., sink_ff_input].
    """
    # Build reverse graph: wire -> list of gate output wires that use this wire as input
    # Build forward graph: gate_output -> list of input wires
    # Actually, build a proper directed graph where nodes are wires.

    ff_outputs: Set[str] = set(netlist['ff_insts'].keys())
    # FF inputs (D pins) — these are the nets that connect to FF D pins
    # We need to find the D input net for each FF. We don't have it directly from
    # ff_insts which maps Q -> inst_name.
    # Let's build a reverse map: for each gate output, which gate drove it?

    # Forward: wire -> [wires driven from this wire through gates]
    # A gate output wire is driven by combinational logic from input wires.
    # So from a gate output, the inputs are known.
    # From a wire that is a gate input, the gate's output is driven from it.

    # Build: for each wire, what gate outputs does it feed into?
    wire_feeds: Dict[str, List[str]] = defaultdict(list)

    for out_wire, (gate_type, in_wires) in netlist['gate_insts'].items():
        for in_w in in_wires:
            if in_w in netlist['wires']:
                wire_feeds[in_w].append(out_wire)

    # Collect all FF D-input wires (need to search gate inputs for FF names)
    # Actually we need a different approach: trace forward from FF outputs.

    # Forward trace: start from each FF output, follow through gates until
    # reaching another FF's D input.

    # For each FF output wire, find all reachable sink FF D-input wires
    all_paths: List[List[str]] = []

    for ff_q, ff_name in netlist['ff_insts'].items():
        if ff_q not in netlist['wires']:
            continue

        visited: Set[str] = set()
        # BFS/DFS: from this wire, follow through gate chains
        stack: List[Tuple[str, List[str]]] = [(ff_q, [ff_q])]

        while stack:
            wire, path = stack.pop()
            if wire in visited and len(path) > 10:
                continue
            visited.add(wire)

            # Check if this wire is a FF D-input (i.e., feeds into a FF)
            # We identify this by checking if any FF has this as its D input
            # Since we only have Q->name mapping, we need D->name too
            # Let's check if any gate feeds into a wire that IS a FF D input.
            # Actually, we can check if this wire feeds into a gate whose
            # output is not a FF output and not in our forward map.

            # Check if this wire is a known FF output (other than start)
            if wire in ff_outputs and wire != ff_q:
                all_paths.append(path)
                continue

            # Follow forward through gates
            next_wires = wire_feeds.get(wire, [])
            for nw in next_wires:
                if nw not in visited:
                    stack.append((nw, path + [nw]))

    return all_paths


# =============================================================================
# Main
# =============================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python path_trace.py <verilog_file> [--paths] [--stats]")
        sys.exit(1)

    filepath = sys.argv[1]
    show_paths = '--paths' in sys.argv
    show_stats = '--stats' in sys.argv or not show_paths  # stats by default

    if not os.path.exists(filepath):
        print(f"Error: file not found: {filepath}")
        sys.exit(1)

    print(f"Parsing: {filepath}")
    netlist = parse_verilog(filepath)

    print(f"Module: {netlist['module_name']}")
    print(f"  Flip-flops: {len(netlist['ff_insts'])}")
    print(f"  Gates:      {len(netlist['gate_insts'])}")
    print(f"  Top ports:  {len(netlist['top_ports'])}")
    print(f"  Wires:      {len(netlist['wires'])}")

    if show_stats and netlist['ff_insts']:
        # Show FF details
        for q, name in list(netlist['ff_insts'].items())[:5]:
            print(f"    FF {name}: Q={q}")
        if len(netlist['ff_insts']) > 5:
            print(f"    ... and {len(netlist['ff_insts'])-5} more FFs")

        # Show gate details
        for out, (typ, ins) in list(netlist['gate_insts'].items())[:5]:
            print(f"    Gate {typ}: {ins} -> {out}")
        if len(netlist['gate_insts']) > 5:
            print(f"    ... and {len(netlist['gate_insts'])-5} more gates")

    # Trace paths
    print("\nTracing FF-to-FF paths...")
    paths = trace_ff_to_ff_paths(netlist)
    print(f"Found {len(paths)} paths")

    if show_paths:
        for i, path in enumerate(paths[:50]):
            print(f"  Path {i+1}: {' -> '.join(path)}")
        if len(paths) > 50:
            print(f"  ... and {len(paths)-50} more paths")

    # Path length distribution
    if show_stats and paths:
        lengths = [len(p) for p in paths]
        from collections import Counter
        dist = Counter(lengths)
        print(f"\nPath length distribution (#wires):")
        for length in sorted(dist):
            print(f"  {length} wires: {dist[length]} paths")

        max_path = max(paths, key=lambda p: len(p))
        print(f"\nLongest path ({len(max_path)} wires):")
        print(f"  {' -> '.join(max_path)}")


if __name__ == '__main__':
    main()

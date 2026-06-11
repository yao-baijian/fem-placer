# =============================================================================
# Timing Analysis Script
# =============================================================================
# Usage:
#   cd vivado
#   vivado -mode batch -source ../tcl/analyze_timing.tcl -tclargs <dcp_file> <output_dir> [clock_period_ns] [clock_port]
#
# Examples:
#   vivado -mode batch -source ../tcl/analyze_timing.tcl -tclargs output_dir/c2670/post_impl.dcp output_dir/c2670 5.0 clk
#   vivado -mode batch -source ../tcl/analyze_timing.tcl -tclargs output_dir/c5315/post_impl.dcp output_dir/c5315
#
# If clock_port is 'none', creates a virtual clock instead.
# =============================================================================

if { $argc < 2 } {
    puts "Usage: vivado -mode batch -source [file tail [info script]] -tclargs <dcp_file> <output_dir> \[clock_period_ns\] \[clock_port\]"
    puts ""
    puts "  <dcp_file>       : Path to the post-implementation DCP checkpoint"
    puts "  <output_dir>     : Directory for timing report outputs"
    puts "  [clock_period_ns]: Clock period in ns (default: 5.0)"
    puts "  [clock_port]     : Clock port name, or 'none' for virtual clock (default: auto-detect)"
    exit 1
}

set dcp_file    [lindex $argv 0]
set output_dir  [lindex $argv 1]

if { $argc > 2 } {
    set clock_period_ns [lindex $argv 2]
} else {
    set clock_period_ns 5.0
}

if { $argc > 3 } {
    set clock_port [lindex $argv 3]
} else {
    set clock_port "auto"
}

file mkdir $output_dir

puts "========================================"
puts "  Timing Analysis"
puts "========================================"
puts "  DCP File     : $dcp_file"
puts "  Output Dir   : $output_dir"
puts "  Clock Period : $clock_period_ns ns"
puts "  Clock Port   : $clock_port"
puts "========================================"

# Open the design
if { [catch {open_checkpoint $dcp_file} err] } {
    puts "Error: Failed to open checkpoint $dcp_file"
    puts "  $err"
    exit 1
}
puts "Checkpoint opened successfully."

# Determine clock source
set clock_created 0

if { $clock_port eq "none" } {
    # Create a virtual clock (for out-of-context designs)
    puts "Creating virtual clock..."
    create_clock -period $clock_period_ns -name virtual_clk
    set clock_created 1
} elseif { $clock_port eq "auto" } {
    # Try to auto-detect clock ports
    set clock_ports [get_ports -quiet -filter {DIRECTION == IN && NAME =~ "*clk*" || NAME =~ "*clock*" || NAME =~ "*CK*"}]
    
    if { [llength $clock_ports] > 0 } {
        set clk_port [lindex $clock_ports 0]
        puts "Auto-detected clock port: $clk_port"
        create_clock -period $clock_period_ns -name clk [get_ports $clk_port]
        set clock_created 1
    } else {
        # Try to find clock nets
        set clock_nets [get_nets -quiet -filter {PRIMITIVE_GROUP == CLOCK}]
        if { [llength $clock_nets] == 0 } {
            # Fallback: try common clock net names
            set clock_nets [get_nets -quiet -filter {NAME =~ "*clk*" && PRIMITIVE_GROUP != ""}]
        }
        if { [llength $clock_nets] > 0 } {
            set clk_net [lindex $clock_nets 0]
            puts "Auto-detected clock net: $clk_net"
            create_clock -period $clock_period_ns -name clk [get_nets $clk_net]
            set clock_created 1
        } else {
            # Try to find any BUFG or clock buffer
            set clock_bufs [get_cells -quiet -filter {PRIMITIVE_TYPE =~ "*.CLOCK.*" || REF_NAME =~ "BUFG*"}]
            if { [llength $clock_bufs] > 0 } {
                set clk_cell [lindex $clock_bufs 0]
                puts "Auto-detected clock buffer: $clk_cell"
                create_clock -period $clock_period_ns -name clk [get_pins $clk_cell/O]
                set clock_created 1
            } else {
                puts "Warning: No clock port/net detected. Creating virtual clock."
                create_clock -period $clock_period_ns -name virtual_clk
                set clock_created 1
            }
        }
    }
} else {
    # Use user-specified clock port
    set port_exists [get_ports -quiet $clock_port]
    if { [llength $port_exists] > 0 } {
        puts "Using specified clock port: $clock_port"
        create_clock -period $clock_period_ns -name clk [get_ports $clock_port]
    } else {
        # Try as a net
        set net_exists [get_nets -quiet $clock_port]
        if { [llength $net_exists] > 0 } {
            puts "Using specified clock net: $clock_port"
            create_clock -period $clock_period_ns -name clk [get_nets $clock_port]
        } else {
            puts "Warning: Specified clock port/net '$clock_port' not found. Creating virtual clock."
            create_clock -period $clock_period_ns -name virtual_clk
        }
    }
    set clock_created 1
}

puts "Clock constraint applied: period = $clock_period_ns ns"

# Generate timing reports
puts "Generating timing summary..."
report_timing_summary -file [file join $output_dir timing_summary.rpt] -delay_type min_max

puts "Generating detailed timing paths..."
report_timing -max_paths 100 -file [file join $output_dir timing_paths.rpt] -delay_type min_max -setup

puts "Generating critical paths..."
report_timing -max_paths 10 -nworst 10 -file [file join $output_dir critical_paths.rpt] -setup

# Extract WNS and TNS for easy parsing
set wns ""
set tns 0.0
set num_failing 0
set total_paths 0

set paths [get_timing_paths -max_paths 1 -nworst 1 -setup]
if { [llength $paths] > 0 } {
    set wns [get_property SLACK [lindex $paths 0]]
    puts "WNS (Worst Negative Slack): $wns ns"
} else {
    puts "WNS: No timing paths found"
}

foreach path [get_timing_paths -max_paths 10000 -nworst 10000 -setup] {
    set slack [get_property SLACK $path]
    incr total_paths
    if { $slack < 0 } {
        set tns [expr {$tns + $slack}]
        incr num_failing
    }
}
puts "TNS (Total Negative Slack): $tns ns"
puts "Failing Paths: $num_failing / $total_paths"

# Compute Fmax
if { $wns != "" } {
    set fmax [expr {1000.0 / ($clock_period_ns - $wns)}]
    puts "Estimated Fmax: $fmax MHz"
} else {
    set fmax "N/A"
    puts "Estimated Fmax: N/A"
}

# Save timing metrics to text file
set fp [open [file join $output_dir timing_metrics.txt] w]
puts $fp "Timing Metrics for: $dcp_file"
puts $fp "Clock Period: $clock_period_ns ns"
puts $fp "========================================"
if { $wns != "" } {
    puts $fp "WNS: $wns"
} else {
    puts $fp "WNS: N/A"
}
puts $fp "TNS: $tns"
puts $fp "Failing Paths: $num_failing"
puts $fp "Total Paths: $total_paths"
puts $fp "Fmax (MHz): $fmax"
puts $fp "========================================"
close $fp

puts ""
puts "========================================"
puts "  Timing Analysis Complete"
puts "========================================"
puts "  WNS        : $wns ns"
puts "  TNS        : $tns ns"
puts "  Failing    : $num_failing / $total_paths"
puts "  Fmax       : $fmax MHz"
puts "  Reports    : $output_dir/"
puts "========================================"

quit

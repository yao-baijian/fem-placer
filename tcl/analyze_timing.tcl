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

proc run_timing_analysis {dcp_file output_dir clock_period_ns clock_port} {
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
        return
    }
    puts "Checkpoint opened successfully."

    # Check if the design already has clocks (from the original implementation)
    set existing_clocks [get_clocks -quiet]
    if { [llength $existing_clocks] > 0 } {
        puts "Using existing clock(s) from the checkpoint: [join [get_property NAME $existing_clocks] {, }]"
        set clk_name [lindex $existing_clocks 0]
        set orig_period [get_property PERIOD [get_clocks $clk_name]]
        if { $orig_period != $clock_period_ns } {
            puts "Overriding clock period from ${orig_period} to ${clock_period_ns} ns"
            set_property PERIOD $clock_period_ns [get_clocks $clk_name]
        }
    } else {
        # No existing clocks — try to auto-detect a clock source
        set clk_target ""
        if { $clock_port ne "none" && $clock_port ne "auto" } {
            # User specified a specific port name
            set port_exists [get_ports -quiet $clock_port]
            if { [llength $port_exists] > 0 } {
                set clk_target [get_ports $clock_port]
            } else {
                set net_exists [get_nets -quiet $clock_port]
                if { [llength $net_exists] > 0 } {
                    set clk_target [get_nets $clock_port]
                }
            }
        }
        if { $clk_target eq "" } {
            # Auto-detect: search ports, then nets, then clock buffers
            set clk_ports [get_ports -quiet -filter {DIRECTION == IN && (NAME =~ "*clk*" || NAME =~ "*clock*" || NAME =~ "*CK*")}]
            if { [llength $clk_ports] > 0 } {
                set clk_target [get_ports [lindex $clk_ports 0]]
                puts "Auto-detected clock port: [lindex $clk_ports 0]"
            } else {
                set clk_nets [get_nets -quiet -filter {PRIMITIVE_GROUP == CLOCK}]
                if { [llength $clk_nets] > 0 } {
                    set clk_target [get_nets [lindex $clk_nets 0]]
                    puts "Auto-detected clock net: [lindex $clk_nets 0]"
                } else {
                    set clk_bufs [get_cells -quiet -filter {PRIMITIVE_TYPE =~ "*.CLOCK.*" || REF_NAME =~ "BUFG*"}]
                    if { [llength $clk_bufs] > 0 } {
                        set clk_target [get_pins [lindex $clk_bufs 0]/O]
                        puts "Auto-detected clock buffer: [lindex $clk_bufs 0]"
                    }
                }
            }
        }
        if { $clk_target ne "" } {
            create_clock -period $clock_period_ns -name override_clk $clk_target
            puts "Created clock 'override_clk' with period ${clock_period_ns} ns"
        } else {
            puts "Warning: No clock source found. Timing analysis may be limited."
        }
    }

    puts "Clock constraint applied: period = $clock_period_ns ns"

    # Generate timing reports
    puts "Generating timing summary..."
    report_timing_summary -file [file join $output_dir timing_summary.rpt] -delay_type min_max

    puts "Generating detailed timing paths..."
    report_timing -max_paths 100 -file [file join $output_dir timing_paths.rpt] -delay_type min_max

    puts "Generating critical paths..."
    report_timing -max_paths 10 -nworst 10 -file [file join $output_dir critical_paths.rpt] -setup

    # Extract WNS and TNS from report_timing output (more reliable than get_timing_paths)
    set wns ""
    set tns ""
    set num_failing ""
    set total_paths ""

    set timing_output [report_timing -max_paths 1 -nworst 1 -setup -return_string]
    # Parse WNS from lines like: "Slack (VIOLATED) :        -0.123ns" or "Slack (MET) :         4.478ns"
    set slack_lines [regexp -all -inline -line {^\s*Slack\s*:?\s*(-?[0-9.]+)} $timing_output]
    if { [llength $slack_lines] >= 2 } {
        set wns [lindex $slack_lines 1]
    } else {
        # Try alternate format: "WNS(ns):  4.478"
        set slack_lines [regexp -all -inline -line {WNS.*?(-?[0-9.]+)} $timing_output]
        if { [llength $slack_lines] >= 2 } {
            set wns [lindex $slack_lines 1]
        }
    }

    if { $wns ne "" } {
        puts "WNS (Worst Negative Slack): $wns ns"
        set fmax [expr {1000.0 / ($clock_period_ns - $wns)}]
        puts "Estimated Fmax: $fmax MHz"
    } else {
        set wns "N/A"
        set fmax "N/A"
        puts "WNS: N/A (no timing paths found)"
        puts "Estimated Fmax: N/A"
    }

    # Also get total path count and TNS from a broader report
    set summary_output [report_timing_summary -return_string -delay_type min_max]
    # Parse "Total Paths: 1171" etc.
    set total_match [regexp -all -inline -line {Total\s+Paths\s*:\s*(\d+)} $summary_output]
    if { [llength $total_match] >= 2 } {
        set total_paths [lindex $total_match 1]
    } else {
        set total_paths "N/A"
    }
    set failing_match [regexp -all -inline -line {Failing\s+Paths\s*:\s*(\d+)} $summary_output]
    if { [llength $failing_match] >= 2 } {
        set num_failing [lindex $failing_match 1]
    } else {
        set num_failing "N/A"
    }
    # Parse TNS from timing summary
    set tns_lines [regexp -all -inline -line {TNS.*?(-?[0-9.]+)} $summary_output]
    if { [llength $tns_lines] >= 2 } {
        set tns [lindex $tns_lines 1]
    } else {
        set tns "N/A"
    }

    puts "TNS (Total Negative Slack): $tns ns"
    puts "Failing Paths: $num_failing / $total_paths"

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

    close_project
}

# =============================================================================
# Main entry point
# =============================================================================
if { $argc < 1 } {
    puts "Usage: vivado -mode batch -source [file tail [info script]] -tclargs <dcp_file> <output_dir> \[clock_period_ns\] \[clock_port\]"
    puts "       vivado -mode batch -source [file tail [info script]] -tclargs -scan <base_output_dir> \[clock_period_ns\] \[clock_port\]"
    puts ""
    puts "  Single mode:"
    puts "    <dcp_file>       : Path to the post-implementation DCP checkpoint"
    puts "    <output_dir>     : Directory for timing report outputs"
    puts "    [clock_period_ns]: Clock period in ns (default: 5.0)"
    puts "    [clock_port]     : Clock port name, or 'none' for virtual clock (default: auto-detect)"
    puts ""
    puts "  Scan mode:"
    puts "    -scan <base_output_dir> : Scan all subdirectories under base_output_dir for post_impl.dcp"
    puts "    [clock_period_ns]       : Clock period in ns (default: 5.0)"
    puts "    [clock_port]            : Clock port name, or 'none' for virtual clock (default: auto-detect)"
    puts ""
    puts "  Examples:"
    puts "    vivado -mode batch -source ../tcl/analyze_timing.tcl -tclargs output_dir/c2670/post_impl.dcp output_dir/c2670 5.0 clk"
    puts "    vivado -mode batch -source ../tcl/analyze_timing.tcl -tclargs -scan output_dir 5.0 clk"
    exit 1
}

if { [lindex $argv 0] eq "-scan" } {
    # ===== Scan mode: process all instances under base_output_dir =====
    if { $argc < 2 } {
        puts "Error: -scan mode requires <base_output_dir>"
        exit 1
    }
    set base_dir [lindex $argv 1]
    
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

    puts "========================================"
    puts "  Scan Mode"
    puts "========================================"
    puts "  Base Dir     : $base_dir"
    puts "  Clock Period : $clock_period_ns ns"
    puts "  Clock Port   : $clock_port"
    puts "========================================"
    puts ""

    set instance_dirs [glob -nocomplain -type d [file join $base_dir *]]
    set count 0
    
    foreach dir $instance_dirs {
        set dcp_file [file join $dir post_impl.dcp]
        if { [file exists $dcp_file] } {
            set instance_name [file tail $dir]
            puts ""
            puts "########################################################################"
            puts "#  [clock format [clock seconds]] - Processing instance: $instance_name"
            puts "########################################################################"
            puts ""
            run_timing_analysis $dcp_file $dir $clock_period_ns $clock_port
            incr count
        }
    }

    puts ""
    puts "========================================"
    puts "  Scan Complete: $count instances processed"
    puts "========================================"

} else {
    # ===== Single mode: process one DCP file =====
    if { $argc < 2 } {
        puts "Error: Single mode requires <dcp_file> and <output_dir>"
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

    run_timing_analysis $dcp_file $output_dir $clock_period_ns $clock_port
}

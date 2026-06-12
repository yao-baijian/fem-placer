# =============================================================================
# ISPD Implementation Script with Timing Constraints (out_of_context mode)
# =============================================================================
# Same as ispd.synth.tcl but with:
#   - out_of_context mode (I/O buffers removed post-open for internal timing)
#   - create_clock and timing reports
# Output goes to timing_output_dir/ instead of output_dir/
# =============================================================================
# Usage:
#   cd vivado
#   vivado -mode batch -source ../tcl/ispd.synth.timing.tcl
# =============================================================================

set synth_dcp "D:/Project/fem-placer-latest/fem-placer/benchmarks/ISPD/FPGA-example2/design.dcp"
set output_dir "D:/Project/fem-placer-latest/fem-placer/vivado/timing_output_dir/FPGA-example2"
set impl_dcp [file join $output_dir "post_impl.dcp"]
set clock_period_ns 5.0

file mkdir $output_dir

open_checkpoint $synth_dcp

# Remove I/O buffers to make the pre-synthesized DCP out_of_context-like.
# This allows proper evaluation of internal timing without I/O pad delays.
puts "Removing I/O buffers for out_of_context-style timing analysis..."
foreach ibuf [get_cells -hier -quiet -filter {PRIMITIVE_TYPE =~ "*.IO.IBUF*"}] {
    set in_pin  [get_pins -quiet -of $ibuf -filter {DIRECTION == IN}]
    set out_pin [get_pins -quiet -of $ibuf -filter {DIRECTION == OUT}]
    set port    [get_ports -quiet -of $in_pin]
    set net     [get_nets -quiet -of $out_pin]
    if { $port ne "" && $net ne "" } {
        disconnect_net -quiet -net [get_nets -of $in_pin] -objects $ibuf
        disconnect_net -quiet -net $net -objects $ibuf
        connect_net -quiet -net $net -objects $port
        remove_cell -quiet $ibuf
    }
}
foreach obuf [get_cells -hier -quiet -filter {PRIMITIVE_TYPE =~ "*.IO.OBUF*"}] {
    set in_pin  [get_pins -quiet -of $obuf -filter {DIRECTION == IN}]
    set out_pin [get_pins -quiet -of $obuf -filter {DIRECTION == OUT}]
    set port    [get_ports -quiet -of $out_pin]
    set net     [get_nets -quiet -of $in_pin]
    if { $port ne "" && $net ne "" } {
        disconnect_net -quiet -net $net -objects $obuf
        disconnect_net -quiet -net [get_nets -of $out_pin] -objects $obuf
        connect_net -quiet -net $net -objects $port
        remove_cell -quiet $obuf
    }
}
puts "I/O buffers removed."

# Apply clock constraint
puts "Applying clock constraint: period = ${clock_period_ns} ns..."
set clk_ports [get_ports -quiet -filter {DIRECTION == IN && (NAME =~ "*clk*" || NAME =~ "*clock*" || NAME =~ "*CK*")}]
if { [llength $clk_ports] > 0 } {
    create_clock -period $clock_period_ns -name sys_clk [get_ports [lindex $clk_ports 0]]
    set has_clock 1
    puts "Created clock on port: [lindex $clk_ports 0]"
} elseif { [llength [get_ports -quiet clk]] > 0 } {
    create_clock -period $clock_period_ns -name sys_clk [get_ports clk]
    set has_clock 1
    puts "Created clock on port: clk"
} else {
    puts "Error: No clock port found. Exiting."
    exit 1
}

opt_design -directive Explore

set place_start [clock seconds]
place_design -directive SSI_SpreadLogic_high
set place_end [clock seconds]
set place_time [expr {$place_end - $place_start}]

set fp [open [file join $output_dir "place_time.txt"] w]
puts $fp $place_time
close $fp

route_design -directive NoTimingRelaxation

write_checkpoint -force $impl_dcp
set edif_file [file join $output_dir "post_impl.edf"]
write_edif -force $edif_file

# Timing and analysis reports
if { $has_clock } {
    report_timing_summary -file [file join $output_dir "timing_summary.rpt"] -delay_type min_max
    report_timing -max_paths 10 -nworst 10 -file [file join $output_dir "critical_paths.rpt"] -setup
}
report_utilization -file [file join $output_dir "utilization.rpt"] -hierarchical
report_design_analysis -file [file join $output_dir "design_analysis.rpt"]
report_route_status -file [file join $output_dir "route_status.rpt"]
report_power -file [file join $output_dir "power.rpt"]

# Extract timing metrics
set wns "N/A"
set tns "N/A"
set fmax "N/A"

if { $has_clock } {
    set rpt_file [file join $output_dir timing_summary.rpt]
    if { [file exists $rpt_file] } {
        set fp [open $rpt_file r]
        set content [read $fp]
        close $fp
        foreach line [split $content \n] {
            if { [regexp {WNS.*?:\s*(-?[0-9.]+)} $line -> val] } { set wns $val }
            if { [regexp {TNS.*?:\s*(-?[0-9.]+)} $line -> val] } { set tns $val }
        }
        if { $wns ne "N/A" } {
            set fmax [format "%.1f" [expr {1000.0 / ($clock_period_ns - $wns)}]]
        }
    }
}

set fp [open [file join $output_dir "timing_metrics.txt"] w]
puts $fp "WNS: $wns"
puts $fp "TNS: $tns"
puts $fp "Fmax (MHz): $fmax"
close $fp

puts "Timing metrics: WNS=$wns ns, Fmax=${fmax} MHz"
puts "Implementation with timing complete."

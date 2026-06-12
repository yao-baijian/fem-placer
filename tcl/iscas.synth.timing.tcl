# =============================================================================
# ISCAS Synthesis Script with Timing Constraints (out_of_context mode)
# =============================================================================
# Same as iscas.synth.tcl but with:
#   - out_of_context synthesis (no I/O buffers, no pin placement)
#   - create_clock and timing reports
# Output goes to timing_output_dir/ instead of output_dir/
# =============================================================================
# Usage:
#   cd vivado
#   vivado -mode batch -source ../tcl/iscas.synth.timing.tcl
# =============================================================================

set benchmarks {
    s1488 D:/Project/fem-placer-latest/fem-placer/benchmarks/ISCAS89/s1488.v
    s5378 D:/Project/fem-placer-latest/fem-placer/benchmarks/ISCAS89/s5378.v
    s9234 D:/Project/fem-placer-latest/fem-placer/benchmarks/ISCAS89/s9234.v
    s15850 D:/Project/fem-placer-latest/fem-placer/benchmarks/ISCAS89/s15850.v
}

set part_name {xcvu065-ffvc1517-1-i}
set base_output_dir {timing_output_dir}
set clock_period_ns 5.0

foreach {top_module rtl_file} $benchmarks {
    puts "========================================"
    puts "Processing benchmark (with timing): $top_module"
    puts "========================================"
    
    set output_dir [file join $base_output_dir $top_module]
    puts "Creating output directory: $output_dir"
    file mkdir $output_dir
    
    set temp_project_dir [file join ./temp_projects ${top_module}_timing]
    file mkdir $temp_project_dir
    
    create_project -part $part_name -force ${top_module}_timing $temp_project_dir
    add_files -norecurse $rtl_file
    set_property top $top_module [current_fileset]
    
    # Synthesis (out_of_context mode: no I/O buffers inserted, no pin placement — allows
    # proper evaluation of internal timing without I/O constraints)
    puts "Running synthesis for $top_module (out_of_context)..."
    synth_design -mode out_of_context -top $top_module -part $part_name -flatten_hierarchy rebuilt
    write_checkpoint -force [file join $output_dir post_synth.dcp]
    
    # Optimization
    puts "Running optimization for $top_module..."
    opt_design
    
    # Apply clock constraint before placement for timing-driven P&R
    puts "Applying clock constraint: period = ${clock_period_ns} ns..."
    set clk_ports [get_ports -quiet -filter {DIRECTION == IN && (NAME =~ "*clk*" || NAME =~ "*clock*" || NAME =~ "*CK*")}]
    if { [llength $clk_ports] > 0 } {
        create_clock -period $clock_period_ns -name sys_clk [get_ports [lindex $clk_ports 0]]
        set has_clock 1
        puts "Created clock on port: [lindex $clk_ports 0]"
    } else {
        puts "Error: No clock port found for $top_module. Skipping this instance."
        close_project
        continue
    }
    
    # Placement
    puts "Running placement for $top_module..."
    set place_start [clock seconds]
    place_design
    set place_end [clock seconds]
    set place_time [expr {$place_end - $place_start}]

    set fp [open [file join $output_dir "place_time.txt"] w]
    puts $fp $place_time
    close $fp
    
    # Routing
    puts "Running routing for $top_module..."
    route_design
    write_checkpoint -force [file join $output_dir post_impl.dcp]
    
    # Reports
    puts "Generating reports for $top_module..."
    if { $has_clock } {
        report_timing_summary -file [file join $output_dir timing_summary.rpt] -delay_type min_max
        report_timing -max_paths 10 -nworst 10 -file [file join $output_dir critical_paths.rpt] -setup
    }
    report_design_analysis -file [file join $output_dir placement_analysis.rpt] -name placement_analysis
    report_route_status -file [file join $output_dir route_status.rpt]
    
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
    
    puts "Timing metrics for $top_module: WNS=$wns ns, Fmax=${fmax} MHz"
    puts "Completed processing $top_module"
    puts ""
    
    close_project
}

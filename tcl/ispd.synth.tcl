set synth_dcp "D:/Project/fem-placer-latest/fem-placer/benchmarks/ISPD/FPGA-example1/design.dcp"  ;# 综合后的DCP
set output_dir "D:/Project/fem-placer-latest/fem-placer/vivado/output_dir/FPGA-example1"         ;# 输出目录
set impl_dcp [file join $output_dir "post_impl.dcp"]                                  ;# 实现后的DCP

file mkdir $output_dir

open_checkpoint $synth_dcp

opt_design -directive Explore ;# 或 AlternateArea

set place_start [clock seconds]
place_design -directive SSI_SpreadLogic_high
set place_end [clock seconds]
set place_time [expr {$place_end - $place_start}]

set fp [open [file join $output_dir "place_time.txt"] w]
puts $fp $place_time
close $fp

# phys_opt_design -force_replication 
# 降低扇出

route_design -directive NoTimingRelaxation

write_checkpoint -force $impl_dcp
set edif_file [file join $output_dir "post_impl.edf"]
write_edif -force $edif_file

report_route_status

# report_timing_summary -file [file join $output_dir "timing_summary.rpt"] -delay_type min_max
# report_timing -max_paths 10 -file [file join $output_dir "timing_paths.rpt"]
# report_utilization -file [file join $output_dir "utilization.rpt"] -hierarchical
# report_design_analysis -file [file join $output_dir "design_analysis.rpt"]
# report_route_status -file [file join $output_dir "route_status.rpt"]
# report_power -file [file join $output_dir "power.rpt"]
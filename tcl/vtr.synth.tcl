
set benchmarks {
    bgm D:/Project/fem-placer-latest/fem-placer/benchmarks/vtr/bgm.v
    RLE_BlobMerging D:/Project/fem-placer-latest/fem-placer/benchmarks/vtr/blob_merge.v
    paj_boundtop_hierarchy_no_mem D:/Project/fem-placer-latest/fem-placer/benchmarks/vtr/boundtop.v
    memory_controller D:/Project/fem-placer-latest/fem-placer/benchmarks/vtr/ch_intrinsics.v
    diffeq_paj_convert D:/Project/fem-placer-latest/fem-placer/benchmarks/vtr/diffeq1.v
    diffeq_f_systemC D:/Project/fem-placer-latest/fem-placer/benchmarks/vtr/diffeq2.v
    LU8PEEng D:/Project/fem-placer-latest/fem-placer/benchmarks/vtr/LU8PEEng.v
    LU32PEEng D:/Project/fem-placer-latest/fem-placer/benchmarks/vtr/LU32PEEng.v
    mcml D:/Project/fem-placer-latest/fem-placer/benchmarks/vtr/mcml.v
    mkPktMerge D:/Project/fem-placer-latest/fem-placer/benchmarks/vtr/mkPktMerge.v
    mkSMAdapter4B D:/Project/fem-placer-latest/fem-placer/benchmarks/vtr/mkSMAdapter4B.v
    or1200_flat D:/Project/fem-placer-latest/fem-placer/benchmarks/vtr/or1200.v
    paj_raygentop_hierarchy_no_mem D:/Project/fem-placer-latest/fem-placer/benchmarks/vtr/raygentop.v
    sha1 D:/Project/fem-placer-latest/fem-placer/benchmarks/vtr/sha.v
    sv_chip0_hierarchy_no_mem D:/Project/fem-placer-latest/fem-placer/benchmarks/vtr/stereovision0.v
    sv_chip1_hierarchy_no_mem D:/Project/fem-placer-latest/fem-placer/benchmarks/vtr/stereovision1.v
    sv_chip2_hierarchy_no_mem D:/Project/fem-placer-latest/fem-placer/benchmarks/vtr/stereovision2.v
    sv_chip3_hierarchy_no_mem D:/Project/fem-placer-latest/fem-placer/benchmarks/vtr/stereovision3.v
}

# bgm /home/byao/Desktop/fem_rev/fem/benchmarks/vtr/verilog/bgm.v
# RLE_BlobMerging /home/byao/Desktop/fem_rev/fem/benchmarks/vtr/verilog/blob_merge.v
# paj_boundtop_hierarchy_no_mem /home/byao/Desktop/fem_rev/fem/benchmarks/vtr/verilog/boundtop.v
# memory_controller /home/byao/Desktop/fem_rev/fem/benchmarks/vtr/verilog/ch_intrinsics.v
# diffeq_paj_convert /home/byao/Desktop/fem_rev/fem/benchmarks/vtr/verilog/diffeq1.v
# diffeq_f_systemC /home/byao/Desktop/fem_rev/fem/benchmarks/vtr/verilog/diffeq2.v
# LU8PEEng /home/byao/Desktop/fem_rev/fem/benchmarks/vtr/verilog/LU8PEEng.v
# LU32PEEng /home/byao/Desktop/fem_rev/fem/benchmarks/vtr/verilog/LU32PEEng.v
# mcml /home/byao/Desktop/fem_rev/fem/benchmarks/vtr/verilog/mcml.v
# mkDelayWorker32B /home/byao/Desktop/fem_rev/fem/benchmarks/vtr/verilog/mkDelayWorker32B.v
# mkPktMerge /home/byao/Desktop/fem_rev/fem/benchmarks/vtr/verilog/mkPktMerge.v
# mkSMAdapter4B /home/byao/Desktop/fem_rev/fem/benchmarks/vtr/verilog/mkSMAdapter4B.v
# or1200_flat /home/byao/Desktop/fem_rev/fem/benchmarks/vtr/verilog/or1200.v
# paj_raygentop_hierarchy_no_mem /home/byao/Desktop/fem_rev/fem/benchmarks/vtr/verilog/raygentop.v
# sha1 /home/byao/Desktop/fem_rev/fem/benchmarks/vtr/verilog/sha.v
# sv_chip0_hierarchy_no_mem /home/byao/Desktop/fem_rev/fem/benchmarks/vtr/verilog/stereovision0.v
# sv_chip1_hierarchy_no_mem /home/byao/Desktop/fem_rev/fem/benchmarks/vtr/verilog/stereovision1.v
# sv_chip2_hierarchy_no_mem /home/byao/Desktop/fem_rev/fem/benchmarks/vtr/verilog/stereovision2.v
# sv_chip3_hierarchy_no_mem /home/byao/Desktop/fem_rev/fem/benchmarks/vtr/verilog/stereovision3.v

set part_name {xcvu065-ffvc1517-1-i}
set base_output_dir {output_dir}

# Process each benchmark
foreach {top_module rtl_file} $benchmarks {
    puts "========================================"
    puts "Processing benchmark: $top_module"
    puts "========================================"
    
    # Create output directory
    set output_dir [file join $base_output_dir $top_module]
    puts "Creating output directory: $output_dir"
    file mkdir $output_dir
    
    # Create temp project directory
    set temp_project_dir [file join ./temp_projects $top_module]
    file mkdir $temp_project_dir
    
    # Create project
    create_project -part $part_name -force $top_module $temp_project_dir
    add_files -norecurse $rtl_file D:/Project/fem-placer-latest/fem-placer/benchmarks/vtr/vtr_primitives.v
    set_property top $top_module [current_fileset]
    
    # Synthesis
    puts "Running synthesis for $top_module..."
    synth_design -top $top_module -part $part_name -flatten_hierarchy rebuilt
    write_checkpoint -force [file join $output_dir post_synth.dcp]
    
    # Optimization
    puts "Running optimization for $top_module..."
    opt_design
    
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
    report_timing_summary -file [file join $output_dir timing_summary.rpt] -delay_type min_max
    report_design_analysis -file [file join $output_dir placement_analysis.rpt] -name placement_analysis
    report_route_status -file [file join $output_dir route_status.rpt]
    
    puts "Completed processing $top_module"
    puts ""
    
    # Close project
    close_project
}

# 1. Define your real core clock running at 200MHz (5.0 ns)
create_clock -name clk -period 5.000 [get_ports clk]

# 2. Define a VIRTUAL clock (no physical port attached) for the boundaries
create_clock -name v_io_clk -period 5.000

# 3. Constrain all inputs/outputs relative to the virtual clock domain
set_input_delay -clock v_io_clk -max 0.0 [all_inputs]
set_output_delay -clock v_io_clk -max 0.0 [all_outputs]
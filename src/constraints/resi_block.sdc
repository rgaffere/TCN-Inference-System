create_clock -name clk -period 5.000 [get_ports clk]

set data_inputs [remove_from_collection [all_inputs] [get_ports clk]]
set data_inputs [remove_from_collection $data_inputs [get_ports rst_n]]

set_input_delay 0.2 -clock clk $data_inputs
set_output_delay 0.2 -clock clk [all_outputs]

set_false_path -from [get_ports rst_n]
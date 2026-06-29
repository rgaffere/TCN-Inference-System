/*
Author: Ryan G
Description: Vectorized ReLU module for TCN
*/

module ReLU_array #(
    parameter int DATA_LEN = 32,
    parameter int NUM_CHANNELS = 16
)(
    input var logic signed [DATA_LEN - 1: 0] data_in [0: NUM_CHANNELS - 1],
    output var logic signed [DATA_LEN - 1: 0] data_out [0: NUM_CHANNELS - 1]
);
    genvar i;
    generate
        for(i = 0; i < NUM_CHANNELS; i++) begin : gen_relu_lane
            assign data_out[i] = data_in[i][DATA_LEN-1] ? '0 : data_in[i];
        end
    endgenerate
endmodule
/*
Author: Ryan G
Description: ReLU module for TCN
*/

module ReLU #(
    parameter int DATA_LEN = 8
)(
    input logic signed [DATA_LEN - 1: 0] data_in,
    output logic signed [DATA_LEN - 1: 0] data_out
);

    assign data_out = data_in[DATA_LEN-1] ? '0 : data_in;
    
endmodule
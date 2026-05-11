/*
Author: Ryan G
Date: May 10th, 2026
Description: Single-cycle combinational MAC datapath
Notes: INT8 input with INT32 output

MAC is going to stay dumb. Pipelining and such occurs at a higher level.

Also, we already decided to sacrifice space for computation speed since this is an accelerator so high area is ok.
Note from ryan in the future: im going to bite these words
*/
module MAC_combi #(
    parameter int IN_LEN = 8,
    localparam int PROD_LEN = 2 * IN_LEN,
    localparam int OUT_LEN = 4 * IN_LEN
)(
    input logic signed [IN_LEN - 1:0] x,
    input logic signed [IN_LEN - 1:0] weight,
    input logic signed [OUT_LEN - 1: 0] acc_in,
    output logic signed [OUT_LEN - 1: 0] acc_out
);
    logic signed [PROD_LEN - 1:0] product;

    assign product = x * weight;
    assign acc_out = acc_in + {{PROD_LEN{product[PROD_LEN-1]}}, product};
endmodule
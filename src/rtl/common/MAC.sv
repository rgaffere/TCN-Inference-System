/*
By Ryan
Date: May 10th, 2026
Description: A MAC unit with a two stage pipeline
Stage 1: MULTIPLY
Stage 2: Accumulate

NOTES:

INT8 X INT8 = INT16
INT16 product is sign-extended to INT32 before accumulation
*/

module MAC #(
    parameter int IN_LEN = 8,
    localparam int PROD_LEN = 2 * IN_LEN,
    localparam int ACC_LEN = 4 * IN_LEN
)(
    input logic clk, rst_n,

    input logic valid_in,
    input logic signed [IN_LEN - 1: 0] x, w,
    input logic signed [ACC_LEN - 1: 0] acc_in,

    output logic valid_out,
    output logic signed [ACC_LEN - 1: 0] acc_out
);
    logic signed [PROD_LEN - 1: 0] mul_reg;
    logic signed [ACC_LEN - 1: 0] acc_in_reg;
    logic valid_s1;

    // Stage 1: Multiply
    always_ff @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            mul_reg <= '0;
            acc_in_reg <= '0;
            valid_s1 <= 1'b0;
        end else begin
            mul_reg <= x * w;
            acc_in_reg <= acc_in;
            valid_s1 <= valid_in;
        end
    end

    // Stage 2: Accumulate
    always_ff @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            acc_out <= '0;
            valid_out <= 1'b0;
        end else begin
            acc_out <= {{PROD_LEN{mul_reg[PROD_LEN-1]}}, mul_reg} + acc_in_reg;
            valid_out <= valid_s1;
        end
    end
endmodule
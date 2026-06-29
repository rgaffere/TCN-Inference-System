/*
By Ryan
Date: May 10th, 2026
Description: A pipelined mac with states

NOTES:

INT8 X INT8 = INT16
INT16 product is sign-extended to INT32 before accumulation
*/

module conv_MAC #(
    parameter int IN_LEN = 8,
    parameter int KERNEL_LEN = 3,
    localparam int PROD_LEN = 2 * IN_LEN,
    localparam int ACC_LEN = 4 * IN_LEN
)(
    input logic clk, rst_n,

    input logic valid_in,
    input logic signed [IN_LEN - 1: 0] x, w,
    input logic signed [ACC_LEN - 1: 0] bias,

    output logic valid_out,
    output logic signed [ACC_LEN - 1: 0] acc_out
);
    logic do_acc;
    logic [$clog2(KERNEL_LEN) - 1: 0] tap_idx;
    logic signed [PROD_LEN - 1: 0] prod_reg;
    logic signed [ACC_LEN - 1: 0] acc_reg;

    // This is for signed bit extension
    logic signed [ACC_LEN - 1:0] prod_ext;

    assign prod_ext = $signed(prod_reg);

    always_ff @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            tap_idx <= '0;
            valid_out <= 1'b0;
            do_acc <= 1'b0;
            prod_reg <= '0;
            acc_reg <= '0;
            acc_out <= '0;
        end else begin
            if(valid_in) begin
                prod_reg <= x * w;
                do_acc <= 1'b1;
            end else begin
                do_acc <= 1'b0;
            end

            if(do_acc) begin
                // Increment index to get ready for next multiplication
                if(tap_idx == KERNEL_LEN - 1) begin
                    acc_out <= acc_reg + prod_ext;
                    acc_reg <= bias;
                    tap_idx <= '0;
                    valid_out <= 1'b1;
                end else begin
                    acc_reg <= acc_reg + prod_ext;
                    tap_idx <= tap_idx + 1'b1;
                    valid_out <= 1'b0;
                end
            end else begin
                valid_out <= 1'b0;
                acc_reg <= bias;
            end
        end
    end
endmodule
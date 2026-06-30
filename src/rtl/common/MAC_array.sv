/*
By Ryan
Date: May 10th, 2026
Description: Vectorized conv_MAC array for NUM_CHANNELS parallel channels

NOTES:
INT8 x INT8 = INT16
INT16 product is sign-extended to INT32 before accumulation
*/

module MAC_array #(
    parameter int IN_LEN = 8,
    parameter int KERNEL_LEN = 3,
    parameter int NUM_CHANNELS = 16,

    localparam int PROD_LEN = 2 * IN_LEN,
    localparam int ACC_LEN = 4 * IN_LEN
)(
    input logic clk,
    input logic rst_n,

    input logic valid_in,

    input logic signed [IN_LEN - 1: 0] x [0: NUM_CHANNELS - 1],
    input logic signed [IN_LEN - 1: 0] w [0: NUM_CHANNELS - 1],
    input logic signed [ACC_LEN - 1: 0] bias [0: NUM_CHANNELS - 1],

    output logic valid_out,
    output logic signed [ACC_LEN - 1: 0] acc_out [0: NUM_CHANNELS - 1]
);

    logic do_acc;
    logic [$clog2(KERNEL_LEN) - 1: 0] tap_idx;

    logic signed [PROD_LEN - 1: 0] prod_reg [0: NUM_CHANNELS - 1];
    logic signed [ACC_LEN - 1: 0] acc_reg [0: NUM_CHANNELS - 1];
    logic signed [ACC_LEN - 1: 0] prod_ext [0: NUM_CHANNELS - 1];

    genvar i;
    generate
        for (i = 0; i < NUM_CHANNELS; i++) begin : gen_mac_lane

            assign prod_ext[i] = $signed(prod_reg[i]);

            always_ff @(posedge clk) begin
                prod_reg[i] <= x[i] * w[i];

                if (do_acc) begin
                    if (tap_idx == KERNEL_LEN - 1) begin
                        acc_out[i] <= acc_reg[i] + prod_ext[i];
                        acc_reg[i] <= bias[i];
                    end else begin
                        acc_reg[i] <= acc_reg[i] + prod_ext[i];
                    end
                end else begin
                    acc_reg[i] <= bias[i];
                end
            end

        end
    endgenerate

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            do_acc <= 1'b0;
            tap_idx <= '0;
            valid_out <= 1'b0;
        end else begin

            do_acc <= valid_in;

            if (do_acc) begin
                if (tap_idx == KERNEL_LEN - 1) begin
                    tap_idx <= '0;
                    valid_out <= 1'b1;
                end else begin
                    tap_idx <= tap_idx + 1'b1;
                    valid_out <= 1'b0;
                end
            end else begin
                tap_idx <= '0;
                valid_out <= 1'b0;
            end
        end
    end

endmodule

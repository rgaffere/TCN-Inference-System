module resi_block #(
    parameter int NUM_CHANNELS = 16,
    parameter int W_BIT_WIDTH = 8,
    parameter int B_BIT_WIDTH = 32,
    parameter int KERNEL_LEN = 3
)(
    input logic clk, rst_n, valid_in,
    input var logic signed [W_BIT_WIDTH - 1: 0] weights1 [0: NUM_CHANNELS - 1][0: KERNEL_LEN - 1],
    input var logic signed [W_BIT_WIDTH - 1: 0] weights2 [0: NUM_CHANNELS - 1][0: KERNEL_LEN - 1],
    input var logic signed [B_BIT_WIDTH - 1: 0] bias1 [0: NUM_CHANNELS - 1],
    input var logic signed [B_BIT_WIDTH - 1: 0] bias2 [0: NUM_CHANNELS - 1],
    output logic valid_out
);
    // It looks like we need to go even higher upstream for weights and bias loading, possibly highest level since memory is its own system.

    // valid_in=hidden layer 1 go -> hidden layer 2 go -> resi -> valid out
    logic s0, s1, s2;

    logic [NUM_CHANNELS - 1: 0] vo1;
    logic [NUM_CHANNELS - 1: 0] vo2;

    // Generate the first hidden layer
    genvar i;
    generate
        for(i = 0; i < NUM_CHANNELS; i++) begin : gen_h1
            conv hidden_layer1 (
                .clk(clk),
                .rst_n(rst_n),
                .valid_in(valid_in),
                .weights(weights1[i]),
                .bias(bias1[i]),
                .valid_out(vo1[i])
            );
        end
    endgenerate

    // Generate the second hidden layer
    genvar j;
    generate
        for(j = 0; j < NUM_CHANNELS; j++) begin : gen_h2
            conv hidden_layer2 (
                .clk(clk),
                .rst_n(rst_n),
                .valid_in(s0),
                .weights(weights2[j]),
                .bias(bias2[j]),
                .valid_out(vo2[j])
            );
        end
    endgenerate

    always_ff @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            s0 <= 1'b0;
            s1 <= 1'b0;
            s2 <= 1'b0;
            valid_out <= 1'b0;
        end else begin
            s0 <= &vo1;
            s1 <= &vo2;
            if(s1) begin
                // do residual add
            end
            s2 <= s1;
            valid_out <= s2;
        end
    end
endmodule
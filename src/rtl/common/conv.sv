module conv #(
    parameter int DATA_LEN = 8,
    parameter int DEPTH = 512,
    parameter int KERNEL_LEN = 3,
    localparam int ACC_LEN = 32,
    localparam int ADDR_WIDTH = $clog2(DEPTH)
)(
    input logic clk, rst_n, valid_in,

    input var logic signed [DATA_LEN-1:0] weights [0:KERNEL_LEN-1],
    input logic signed [ACC_LEN-1:0] bias,

    output logic valid_out
);
    // We use s0 to introduce 1 cycle latency at start up for read and offsets to be ready
    logic s0, s1, s2, s3;

    // Now for some vars
    logic [ADDR_WIDTH - 1: 0] read_offsets[0: KERNEL_LEN - 1];
    logic signed [ACC_LEN - 1: 0] mac_out, relu_out;
    
    logic signed [DATA_LEN - 1: 0] x, w, write_in;
    logic [ADDR_WIDTH - 1: 0] read_offset;

    logic [$clog2(KERNEL_LEN) - 1: 0] tap_idx;

    // Init modules
    dilation_offsets init_do(
        .read_offsets(read_offsets)
    );

    // cache init_cache(
    //     .weights(weights),
    //     .bias(bias)
    // );

    conv_MAC init_mac(
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(s1),
        .x(x),
        .w(w),
        .bias(bias),
        .valid_out(s2),
        .acc_out(mac_out)
    );

    ReLU #(
        .DATA_LEN(32)
    ) init_relu (
        .data_in(mac_out),
        .data_out(relu_out)
    );

    quant init_quant(
        .data_in(relu_out),
        .data_out(write_in)
    );

    ring init_ring(
        .clk(clk),
        .rst_n(rst_n),
        .write_en(s3),
        .data_in(write_in),
        .read_offset(read_offset),
        .data_out(x)
    );

    assign w = weights[tap_idx];
    assign read_offset = read_offsets[tap_idx];
    assign valid_out = s3;

    always_ff @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            tap_idx <= '0;
            s0 <= 1'b0;
            s1 <= 1'b0;
            s3 <= 1'b0;
        end else begin
            s0 <= valid_in;
            s1 <= s0;
            s3 <= s2;

            if(tap_idx == KERNEL_LEN - 1) begin
                tap_idx <= '0;
            end else begin
                tap_idx <= tap_idx + 1;
            end
        end
    end
endmodule
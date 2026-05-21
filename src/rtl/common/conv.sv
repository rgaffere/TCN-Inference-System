module conv #(
    parameter int DATA_LEN = 8,
    parameter int DEPTH = 512,
    parameter int KERNEL_LEN = 3,
    localparam int ACC_LEN = 32,
    localparam int ADDR_WIDTH = $clog2(DEPTH)
)(
    input logic clk, rst_n, valid_in,
    output logic valid_out
);

    logic signed [DATA_LEN-1:0] x;
    logic signed [DATA_LEN-1:0] w;

    logic signed [ACC_LEN - 1:0] acc_in;
    logic signed [ACC_LEN - 1:0] acc_reg;
    logic signed [ACC_LEN - 1:0] acc_out;

    logic signed [ACC_LEN - 1:0] relu_out;
    logic signed [DATA_LEN-1:0] quant_out;

    logic [ADDR_WIDTH-1:0] read_offset;
    logic [ADDR_WIDTH-1:0] read_offsets [0:KERNEL_LEN-1];

    logic signed [DATA_LEN-1:0] weight_cache [0:KERNEL_LEN-1];
    logic signed [ACC_LEN - 1:0] bias_cache;

    logic [$clog2(KERNEL_LEN)-1:0] tap_idx;

    // Pipeline signals: load -> MAC -> relu -> requant -> write
    logic s0, s1, s2, s3, s4;
    assign valid_out = s4;

    // Add cache to store relevants weights and bias for convolution

    dilation_offsets init_do(
        .read_offsets(read_offsets)
    );

    // Initialize modules
    MAC init_mac(
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(s1),
        .x(x),
        .w(w),
        .acc_in(acc_in),
        .valid_out(s2),
        .acc_out(acc_reg)
    );

    ReLU #(.DATA_LEN(32)) init_relu(
        .data_in(acc_out),
        .data_out(relu_out)
    );

    //TODO
    requantize init_requantize(
        .data_in(relu_out),
        .data_out(quant_out)
    );

    ring init_ring(
        .clk(clk),
        .rst_n(rst_n),
        .write_en(s4),
        .data_in(quant_out),
        .read_offset(read_offset),
        .data_out(x)
    );

    // Some time mutiplexers

    // Weight MUX
    always_comb begin
        case (tap_idx)
            0 : w = weight_cache[0];
            1 : w = weight_cache[1];
            2 : w = weight_cache[2];
            default : w = 'x;
        endcase
    end

    // Tap select MUX
    always_comb begin
        case (tap_idx)
            0 : read_offset = read_offsets[0];
            1 : read_offset = read_offsets[1];
            2 : read_offset = read_offsets[2];
            default : read_offset = 'x;
        endcase
    end

    // Control logic
    always_ff @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            tap_idx <= '0;
            acc_reg <= '0;
            s0 <= 1'b0;
            s1 <= 1'b0;
            s3 <= 1'b0;
            s4 <= 1'b0;
        end else begin
            s0 <= valid_in;
            s1 <= s0;
            // Don't set s2 here since its contolled by the mac
            // Also don't set s3 until the conv is done, otherwise we have bad writes
            s4 <= s3;

            if (s1) begin
                if (tap_idx == KERNEL_LEN-1) begin
                    tap_idx <= '0;
                    // Use s3 to give time for relu and quantize
                    s3 <= s2;
                    acc_out <= acc_reg;
                end else begin
                    tap_idx <= tap_idx + 1'b1;    
                    s3 <= 1'b0;
                end    
            end
        end
    end

    // control for acc_in
    always_comb begin
        if (tap_idx == '0)
            acc_in = bias_cache;
        else
            acc_in = acc_reg;
    end
endmodule
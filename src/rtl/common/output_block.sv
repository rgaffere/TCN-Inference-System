module output_block #(
    parameter int IN_CHANNELS = 16,
    parameter int OUT_CHANNELS = 6,
    parameter int W_BIT_WIDTH = 8,
    parameter int B_BIT_WIDTH = 32,
    parameter int QUANT_SHIFT = 8,

    localparam int IN_CH_IDX_DEPTH = (IN_CHANNELS <= 1) ? 1 : $clog2(IN_CHANNELS)
)(
    input logic clk,
    input logic rst_n,
    input logic valid_in,

    input var logic signed [W_BIT_WIDTH - 1: 0] inputVals [0: IN_CHANNELS - 1],
    input var logic signed [W_BIT_WIDTH - 1: 0] weights [0: OUT_CHANNELS - 1][0: IN_CHANNELS - 1],
    input var logic signed [B_BIT_WIDTH - 1: 0] bias [0: OUT_CHANNELS - 1],

    output var logic signed [W_BIT_WIDTH - 1: 0] outputVals [0: OUT_CHANNELS - 1],
    output logic valid_out
);

    localparam logic [IN_CH_IDX_DEPTH - 1: 0] IN_CH_LAST = IN_CHANNELS - 1;
    localparam logic [IN_CH_IDX_DEPTH - 1: 0] IN_CH_ONE = 1;

    // Control
    logic mac_active;
    logic mac_start;
    logic mac_go;
    logic mac_done;

    logic [IN_CH_IDX_DEPTH - 1: 0] in_ch_idx;

    assign mac_start = valid_in && !mac_active;
    assign mac_go = mac_start | mac_active;

    // Hold the complete input vector while its channels are serialized
    logic signed [W_BIT_WIDTH - 1: 0] input_hold [0: IN_CHANNELS - 1];
    logic signed [W_BIT_WIDTH - 1: 0] selected_input;

    always_comb begin
        if (mac_start) begin
            selected_input = inputVals[0];
        end else begin
            selected_input = input_hold[in_ch_idx];
        end
    end

    // Six parallel output lanes, one serialized input channel per cycle
    logic signed [W_BIT_WIDTH - 1: 0] mac_x [0: OUT_CHANNELS - 1];
    logic signed [W_BIT_WIDTH - 1: 0] weight_tap [0: OUT_CHANNELS - 1];
    logic signed [B_BIT_WIDTH - 1: 0] mac_out [0: OUT_CHANNELS - 1];
    logic signed [W_BIT_WIDTH - 1: 0] quant_out [0: OUT_CHANNELS - 1];

    always_comb begin
        for (int out_ch = 0; out_ch < OUT_CHANNELS; out_ch++) begin
            mac_x[out_ch] = selected_input;
            weight_tap[out_ch] = weights[out_ch][in_ch_idx];
        end
    end

    MAC_array #(
        .IN_LEN(W_BIT_WIDTH),
        .KERNEL_LEN(IN_CHANNELS),
        .NUM_CHANNELS(OUT_CHANNELS)
    ) init_output_mac (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(mac_go),
        .x(mac_x),
        .w(weight_tap),
        .bias(bias),
        .valid_out(mac_done),
        .acc_out(mac_out)
    );

    // The output projection is linear, so there is no ReLU before quantization.
    quant_array #(
        .IN_LEN(B_BIT_WIDTH),
        .OUT_LEN(W_BIT_WIDTH),
        .SHIFT(QUANT_SHIFT),
        .NUM_CHANNELS(OUT_CHANNELS)
    ) init_output_quant (
        .data_in(mac_out),
        .data_out(quant_out)
    );

    // Input-channel sequencing
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mac_active <= 1'b0;
            in_ch_idx <= '0;

            for (int ch = 0; ch < IN_CHANNELS; ch++) begin
                input_hold[ch] <= '0;
            end
        end else begin
            if (mac_start) begin
                for (int ch = 0; ch < IN_CHANNELS; ch++) begin
                    input_hold[ch] <= inputVals[ch];
                end

                mac_active <= 1'b1;
                in_ch_idx <= IN_CH_ONE;
            end else if (mac_active) begin
                if (in_ch_idx == IN_CH_LAST) begin
                    mac_active <= 1'b0;
                    in_ch_idx <= '0;
                end else begin
                    in_ch_idx <= in_ch_idx + 1'b1;
                end
            end
        end
    end

    // Register the quantized result. mac_done is produced by MAC_array when
    // all IN_CHANNELS products have reached its accumulator.
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out <= 1'b0;

            for (int ch = 0; ch < OUT_CHANNELS; ch++) begin
                outputVals[ch] <= '0;
            end
        end else begin
            valid_out <= mac_done;

            if (mac_done) begin
                for (int ch = 0; ch < OUT_CHANNELS; ch++) begin
                    outputVals[ch] <= quant_out[ch];
                end
            end
        end
    end

`ifndef SYNTHESIS
    initial begin
        // MAC_array uses $clog2(KERNEL_LEN) internally, so it currently
        // requires a serialized accumulation length of at least two.
        if (IN_CHANNELS < 2)
            $error("output_block requires IN_CHANNELS >= 2 with the existing MAC_array");

        if (OUT_CHANNELS < 1)
            $error("output_block requires OUT_CHANNELS >= 1");

        // MAC_array fixes ACC_LEN at 4*IN_LEN.
        if (B_BIT_WIDTH != (4 * W_BIT_WIDTH))
            $error("B_BIT_WIDTH must equal 4*W_BIT_WIDTH for the existing MAC_array");
    end
`endif

endmodule

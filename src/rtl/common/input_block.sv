module input_block #(

    parameter int IN_CHANNELS = 6,
    parameter int OUT_CHANNELS = 16,
    parameter int W_BIT_WIDTH = 8,
    parameter int B_BIT_WIDTH = 32,
    parameter int KERNEL_LEN = 3,
    parameter int DILATION = 1,
    parameter int DEPTH = 512,
    parameter int NUM_RINGS = 4,
    parameter int CHANNELS_PER_RING = 4,
    parameter int QUANT_SHIFT = 8,

    localparam int ADDR_DEPTH = (DEPTH <= 1) ? 1 : $clog2(DEPTH),
    localparam int K_IDX_DEPTH = (KERNEL_LEN <= 1) ? 1 : $clog2(KERNEL_LEN),
    localparam int IN_CH_IDX_DEPTH = (IN_CHANNELS <= 1) ? 1 : $clog2(IN_CHANNELS),
    localparam int OUT_CH_IDX_DEPTH = (OUT_CHANNELS <= 1) ? 1 : $clog2(OUT_CHANNELS),
    localparam int IN_NUM_RINGS = (IN_CHANNELS + CHANNELS_PER_RING - 1) / CHANNELS_PER_RING,
    localparam int OUT_NUM_RINGS = NUM_RINGS,
    localparam int RING_WORD_WIDTH = CHANNELS_PER_RING * W_BIT_WIDTH
)(
    input logic clk,
    input logic rst_n,
    input logic valid_in,

    input var logic signed [W_BIT_WIDTH - 1: 0] inputVals [0: IN_CHANNELS - 1],

    input var logic signed [W_BIT_WIDTH - 1: 0] weights1 [0: OUT_CHANNELS - 1][0: IN_CHANNELS - 1][0: KERNEL_LEN - 1],
    input var logic signed [W_BIT_WIDTH - 1: 0] weights2 [0: OUT_CHANNELS - 1][0: OUT_CHANNELS - 1][0: KERNEL_LEN - 1],
    input var logic signed [W_BIT_WIDTH - 1: 0] residual_weights [0: OUT_CHANNELS - 1][0: IN_CHANNELS - 1],

    input var logic signed [B_BIT_WIDTH - 1: 0] bias1 [0: OUT_CHANNELS - 1],
    input var logic signed [B_BIT_WIDTH - 1: 0] bias2 [0: OUT_CHANNELS - 1],
    input var logic signed [B_BIT_WIDTH - 1: 0] residual_bias [0: OUT_CHANNELS - 1],

    output var logic signed [W_BIT_WIDTH - 1: 0] outputVals [0: OUT_CHANNELS - 1],
    output logic valid_out
);

    localparam logic [K_IDX_DEPTH - 1: 0] K_LAST = KERNEL_LEN - 1;
    localparam logic [IN_CH_IDX_DEPTH - 1: 0] IN_CH_LAST = IN_CHANNELS - 1;
    localparam logic [OUT_CH_IDX_DEPTH - 1: 0] OUT_CH_LAST = OUT_CHANNELS - 1;
    localparam logic [IN_CH_IDX_DEPTH - 1: 0] IN_CH_ONE = {{(IN_CH_IDX_DEPTH - 1){1'b0}}, 1'b1};
    localparam logic [OUT_CH_IDX_DEPTH - 1: 0] OUT_CH_ONE = {{(OUT_CH_IDX_DEPTH - 1){1'b0}}, 1'b1};
    localparam logic [ADDR_DEPTH - 1: 0] DILATION_ADDR = DILATION;

    // Control
    logic s0, s1, s2;
    logic residual_s0;

    logic mac1go, mac2go, residual_mac_go;
    logic mac1_active, mac2_active;
    logic mac1_start, mac2_start;

    logic [ADDR_DEPTH - 1: 0] currOffset1;
    logic [ADDR_DEPTH - 1: 0] currOffset2;

    logic [K_IDX_DEPTH - 1: 0] tap1_idx;
    logic [K_IDX_DEPTH - 1: 0] tap2_idx;

    logic [IN_CH_IDX_DEPTH - 1: 0]  in_ch1_idx;
    logic [OUT_CH_IDX_DEPTH - 1: 0] in_ch2_idx;

    assign mac1_start = valid_in && !mac1_active;
    assign mac2_start = s1 && !mac2_active;

    assign mac1go = mac1_start | mac1_active;
    assign mac2go = mac2_start | mac2_active;

    assign residual_mac_go = mac1go && (tap1_idx == '0);

    // MAC datapaths
    logic signed [W_BIT_WIDTH - 1: 0] macX1 [0: OUT_CHANNELS - 1];
    logic signed [W_BIT_WIDTH - 1: 0] macX2 [0: OUT_CHANNELS - 1];
    logic signed [W_BIT_WIDTH - 1: 0] residualMacX [0: OUT_CHANNELS - 1];

    logic signed [W_BIT_WIDTH - 1: 0] w1_tap [0: OUT_CHANNELS - 1];
    logic signed [W_BIT_WIDTH - 1: 0] w2_tap [0: OUT_CHANNELS - 1];
    logic signed [W_BIT_WIDTH - 1: 0] residual_w_tap [0: OUT_CHANNELS - 1];

    logic signed [B_BIT_WIDTH - 1: 0] macOut1 [0: OUT_CHANNELS - 1];
    logic signed [B_BIT_WIDTH - 1: 0] reluOut1 [0: OUT_CHANNELS - 1];
    logic signed [W_BIT_WIDTH - 1: 0] quantOut1 [0: OUT_CHANNELS - 1];
    logic signed [W_BIT_WIDTH - 1: 0] quantOut1_reg [0: OUT_CHANNELS - 1];

    logic signed [B_BIT_WIDTH - 1: 0] macOut2 [0: OUT_CHANNELS - 1];
    logic signed [B_BIT_WIDTH - 1: 0] reluOut2 [0: OUT_CHANNELS - 1];
    logic signed [W_BIT_WIDTH - 1: 0] quantOut2 [0: OUT_CHANNELS - 1];

    logic signed [B_BIT_WIDTH - 1: 0] residualMacOut [0: OUT_CHANNELS - 1];
    logic signed [W_BIT_WIDTH - 1: 0] residualQuantOut [0: OUT_CHANNELS - 1];

    MAC_array #(
        .IN_LEN(W_BIT_WIDTH),
        .KERNEL_LEN(KERNEL_LEN * IN_CHANNELS),
        .NUM_CHANNELS(OUT_CHANNELS)
    ) init_mac1 (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(mac1go),
        .x(macX1),
        .w(w1_tap),
        .bias(bias1),
        .valid_out(s0),
        .acc_out(macOut1)
    );

    ReLU_array #(
        .DATA_LEN(B_BIT_WIDTH),
        .NUM_CHANNELS(OUT_CHANNELS)
    ) init_relu1 (
        .data_in(macOut1),
        .data_out(reluOut1)
    );

    quant_array #(
        .IN_LEN(B_BIT_WIDTH),
        .OUT_LEN(W_BIT_WIDTH),
        .SHIFT(QUANT_SHIFT),
        .NUM_CHANNELS(OUT_CHANNELS)
    ) init_quant1 (
        .data_in(reluOut1),
        .data_out(quantOut1)
    );

    MAC_array #(
        .IN_LEN(W_BIT_WIDTH),
        .KERNEL_LEN(KERNEL_LEN * OUT_CHANNELS),
        .NUM_CHANNELS(OUT_CHANNELS)
    ) init_mac2 (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(mac2go),
        .x(macX2),
        .w(w2_tap),
        .bias(bias2),
        .valid_out(s2),
        .acc_out(macOut2)
    );

    ReLU_array #(
        .DATA_LEN(B_BIT_WIDTH),
        .NUM_CHANNELS(OUT_CHANNELS)
    ) init_relu2 (
        .data_in(macOut2),
        .data_out(reluOut2)
    );

    quant_array #(
        .IN_LEN(B_BIT_WIDTH),
        .OUT_LEN(W_BIT_WIDTH),
        .SHIFT(QUANT_SHIFT),
        .NUM_CHANNELS(OUT_CHANNELS)
    ) init_quant2 (
        .data_in(reluOut2),
        .data_out(quantOut2)
    );

    // Learned 1x1 residual projection. no ReLU here.
    MAC_array #(
        .IN_LEN(W_BIT_WIDTH),
        .KERNEL_LEN(IN_CHANNELS),
        .NUM_CHANNELS(OUT_CHANNELS)
    ) init_residual_mac (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(residual_mac_go),
        .x(residualMacX),
        .w(residual_w_tap),
        .bias(residual_bias),
        .valid_out(residual_s0),
        .acc_out(residualMacOut)
    );

    quant_array #(
        .IN_LEN(B_BIT_WIDTH),
        .OUT_LEN(W_BIT_WIDTH),
        .SHIFT(QUANT_SHIFT),
        .NUM_CHANNELS(OUT_CHANNELS)
    ) init_quant_residual (
        .data_in(residualMacOut),
        .data_out(residualQuantOut)
    );

    // Ring buffers
    logic [RING_WORD_WIDTH - 1: 0] ringIn1  [0: IN_NUM_RINGS - 1];
    logic [RING_WORD_WIDTH - 1: 0] ringOut1 [0: IN_NUM_RINGS - 1];

    logic [RING_WORD_WIDTH - 1: 0] ringIn2  [0: OUT_NUM_RINGS - 1];
    logic [RING_WORD_WIDTH - 1: 0] ringOut2 [0: OUT_NUM_RINGS - 1];

    logic signed [W_BIT_WIDTH - 1: 0] ringVals1 [0: IN_CHANNELS - 1];
    logic signed [W_BIT_WIDTH - 1: 0] ringVals2 [0: OUT_CHANNELS - 1];

    genvar in_bank, in_lane;
    generate
        for (in_bank = 0; in_bank < IN_NUM_RINGS; in_bank++) begin : gen_input_ring_banks
            for (in_lane = 0; in_lane < CHANNELS_PER_RING; in_lane++) begin : gen_input_ring_lanes
                localparam int CH = in_bank * CHANNELS_PER_RING + in_lane;
                localparam int LO = in_lane * W_BIT_WIDTH;

                if (CH < IN_CHANNELS) begin : gen_valid_input_lane
                    assign ringIn1[in_bank][LO +: W_BIT_WIDTH] = $unsigned(inputVals[CH]);
                    assign ringVals1[CH] = $signed(ringOut1[in_bank][LO +: W_BIT_WIDTH]);
                end else begin : gen_padded_input_lane
                    assign ringIn1[in_bank][LO +: W_BIT_WIDTH] = '0;
                end
            end

            ring #(
                .DEPTH(DEPTH),
                .DATA_LEN(W_BIT_WIDTH)
            ) ring_bank1 (
                .clk(clk),
                .rst_n(rst_n),
                .write_en(mac1_start),
                .data_in(ringIn1[in_bank]),
                .read_offset(currOffset1),
                .data_out(ringOut1[in_bank])
            );
        end
    endgenerate

    genvar out_bank, out_lane;
    generate
        for (out_bank = 0; out_bank < OUT_NUM_RINGS; out_bank++) begin : gen_output_ring_banks
            for (out_lane = 0; out_lane < CHANNELS_PER_RING; out_lane++) begin : gen_output_ring_lanes
                localparam int CH = out_bank * CHANNELS_PER_RING + out_lane;
                localparam int LO = out_lane * W_BIT_WIDTH;

                if (CH < OUT_CHANNELS) begin : gen_valid_output_lane
                    assign ringIn2[out_bank][LO +: W_BIT_WIDTH] = $unsigned(quantOut1_reg[CH]);
                    assign ringVals2[CH] = $signed(ringOut2[out_bank][LO +: W_BIT_WIDTH]);
                end else begin : gen_padded_output_lane
                    assign ringIn2[out_bank][LO +: W_BIT_WIDTH] = '0;
                end
            end

            ring #(
                .DEPTH(DEPTH),
                .DATA_LEN(W_BIT_WIDTH)
            ) ring_bank2 (
                .clk(clk),
                .rst_n(rst_n),
                .write_en(s1),
                .data_in(ringIn2[out_bank]),
                .read_offset(currOffset2),
                .data_out(ringOut2[out_bank])
            );
        end
    endgenerate

    // Conv1 and residual-projection input selection
    logic signed [W_BIT_WIDTH - 1: 0] input_hold [0: IN_CHANNELS - 1];
    logic signed [W_BIT_WIDTH - 1: 0] selectedIn1;
    logic signed [W_BIT_WIDTH - 1: 0] selectedIn2;

    always_comb begin
        if (tap1_idx == '0) begin
            if (mac1_start)
                selectedIn1 = inputVals[in_ch1_idx];
            else
                selectedIn1 = input_hold[in_ch1_idx];
        end else begin
            selectedIn1 = ringVals1[in_ch1_idx];
        end

        if (tap2_idx == '0)
            selectedIn2 = quantOut1_reg[in_ch2_idx];
        else
            selectedIn2 = ringVals2[in_ch2_idx];
    end

    always_comb begin
        for (int out_ch = 0; out_ch < OUT_CHANNELS; out_ch++) begin
            macX1[out_ch] = selectedIn1;
            macX2[out_ch] = selectedIn2;
            residualMacX[out_ch] = selectedIn1;

            w1_tap[out_ch] = weights1[out_ch][in_ch1_idx][tap1_idx];
            w2_tap[out_ch] = weights2[out_ch][in_ch2_idx][tap2_idx];
            residual_w_tap[out_ch] = residual_weights[out_ch][in_ch1_idx];
        end
    end

    // MAC index control
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mac1_active <= 1'b0;
            tap1_idx <= '0;
            in_ch1_idx <= '0;
        end else begin
            if (mac1_start) begin
                mac1_active <= 1'b1;
                tap1_idx <= '0;
                in_ch1_idx <= IN_CH_ONE;
            end else if (mac1_active) begin
                if (in_ch1_idx == IN_CH_LAST) begin
                    in_ch1_idx <= '0;

                    if (tap1_idx == K_LAST) begin
                        mac1_active <= 1'b0;
                        tap1_idx <= '0;
                    end else begin
                        tap1_idx <= tap1_idx + 1'b1;
                    end
                end else begin
                    in_ch1_idx <= in_ch1_idx + 1'b1;
                end
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mac2_active <= 1'b0;
            tap2_idx <= '0;
            in_ch2_idx <= '0;
        end else begin
            if (mac2_start) begin
                mac2_active <= 1'b1;
                tap2_idx <= '0;
                in_ch2_idx <= OUT_CH_ONE;
            end else if (mac2_active) begin
                if (in_ch2_idx == OUT_CH_LAST) begin
                    in_ch2_idx <= '0;

                    if (tap2_idx == K_LAST) begin
                        mac2_active <= 1'b0;
                        tap2_idx <= '0;
                    end else begin
                        tap2_idx <= tap2_idx + 1'b1;
                    end
                end else begin
                    in_ch2_idx <= in_ch2_idx + 1'b1;
                end
            end
        end
    end

    // Ring read-offset/prefetch control
    always_comb begin
        if (mac1_start) begin
            // The ring head advances on this write edge.
            currOffset1 = ADDR_DEPTH'(DILATION - 1);
        end else if (tap1_idx == '0) begin
            currOffset1 = DILATION_ADDR;
        end else if ((in_ch1_idx == IN_CH_LAST) && (tap1_idx < K_LAST)) begin
            // Prefetch the next delayed tap on the last channel of this tap.
            currOffset1 = ADDR_DEPTH'(tap1_idx + 1'b1) * DILATION_ADDR;
        end else begin
            currOffset1 = ADDR_DEPTH'(tap1_idx) * DILATION_ADDR;
        end

        if (mac2_start) begin
            currOffset2 = ADDR_DEPTH'(DILATION - 1);
        end else if (tap2_idx == '0) begin
            currOffset2 = DILATION_ADDR;
        end else if ((in_ch2_idx == OUT_CH_LAST) && (tap2_idx < K_LAST)) begin
            currOffset2 = ADDR_DEPTH'(tap2_idx + 1'b1) * DILATION_ADDR;
        end else begin
            currOffset2 = ADDR_DEPTH'(tap2_idx) * DILATION_ADDR;
        end
    end

    // Projected residual alignment
    localparam int RESI_STAGES = 3;
    logic signed [W_BIT_WIDTH - 1: 0] residual_reg [0: RESI_STAGES - 1][0: OUT_CHANNELS - 1];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int ch = 0; ch < IN_CHANNELS; ch++)
                input_hold[ch] <= '0;

            for (int stage = 0; stage < RESI_STAGES; stage++) begin
                for (int ch = 0; ch < OUT_CHANNELS; ch++) begin
                    residual_reg[stage][ch] <= '0;
                end
            end
        end else begin
            if (mac1_start) begin
                for (int ch = 0; ch < IN_CHANNELS; ch++) begin
                    input_hold[ch] <= inputVals[ch];
                end
            end

            if (residual_s0) begin
                for (int ch = 0; ch < OUT_CHANNELS; ch++) begin
                    residual_reg[0][ch] <= residualQuantOut[ch];
                end
            end

            // Preserve the projected residual before Conv1 accepts another input.
            if (mac1_active && (tap1_idx == K_LAST) && (in_ch1_idx == IN_CH_LAST)) begin
                for (int ch = 0; ch < OUT_CHANNELS; ch++) begin
                    residual_reg[1][ch] <= residual_reg[0][ch];
                end
            end

            // Align the residual slot with the sample currently inside Conv2.
            if (mac2_active && (tap2_idx == '0) && (in_ch2_idx == OUT_CH_ONE)) begin
                for (int ch = 0; ch < OUT_CHANNELS; ch++) begin
                    residual_reg[2][ch] <= residual_reg[1][ch];
                end
            end
        end
    end

    // Residual addition and output registration
    logic signed [W_BIT_WIDTH: 0] residual_sum [0: OUT_CHANNELS - 1];

    localparam logic signed [W_BIT_WIDTH: 0] SAT_MAX = $signed({2'b00, {(W_BIT_WIDTH - 1){1'b1}}});
    localparam logic signed [W_BIT_WIDTH: 0] SAT_MIN = $signed({2'b11, {(W_BIT_WIDTH - 1){1'b0}}});

    always_comb begin
        for (int ch = 0; ch < OUT_CHANNELS; ch++) begin
            residual_sum[ch] = $signed({quantOut2[ch][W_BIT_WIDTH - 1], quantOut2[ch]}) +
                $signed({residual_reg[RESI_STAGES - 1][ch][W_BIT_WIDTH - 1], residual_reg[RESI_STAGES - 1][ch]});
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s1 <= 1'b0;
            valid_out <= 1'b0;

            for (int ch = 0; ch < OUT_CHANNELS; ch++) begin
                quantOut1_reg[ch] <= '0;
                outputVals[ch] <= '0;
            end
        end else begin
            s1 <= s0;
            valid_out <= s2;

            if (s0) begin
                for (int ch = 0; ch < OUT_CHANNELS; ch++)
                    quantOut1_reg[ch] <= quantOut1[ch];
            end

            if (s2) begin
                for (int ch = 0; ch < OUT_CHANNELS; ch++) begin
                    if (residual_sum[ch] > SAT_MAX)
                        outputVals[ch] <= $signed(SAT_MAX[W_BIT_WIDTH - 1: 0]);
                    else if (residual_sum[ch] < SAT_MIN)
                        outputVals[ch] <= $signed(SAT_MIN[W_BIT_WIDTH - 1: 0]);
                    else
                        outputVals[ch] <= $signed(residual_sum[ch][W_BIT_WIDTH - 1: 0]);
                end
            end
        end
    end

// some guards for synthesis
`ifndef SYNTHESIS
    initial begin
        if (IN_CHANNELS < 2)
            $error("inout_block currently requires IN_CHANNELS >= 2");
        if (OUT_CHANNELS < 2)
            $error("inout_block currently requires OUT_CHANNELS >= 2");
        if (KERNEL_LEN < 2)
            $error("inout_block currently requires KERNEL_LEN >= 2");
        if (W_BIT_WIDTH != 8)
            $error("The existing ring module has a fixed 32-bit word and requires W_BIT_WIDTH == 8");
        if (B_BIT_WIDTH != (4 * W_BIT_WIDTH))
            $error("MAC_array fixes ACC_LEN at 4*IN_LEN; B_BIT_WIDTH must equal 4*W_BIT_WIDTH");
        if (CHANNELS_PER_RING != 4)
            $error("The existing ring module packs exactly four INT8 channels per 32-bit word");
        if (NUM_RINGS != ((OUT_CHANNELS + CHANNELS_PER_RING - 1) / CHANNELS_PER_RING))
            $error("NUM_RINGS must equal ceil(OUT_CHANNELS / CHANNELS_PER_RING)");
        if (RING_WORD_WIDTH != 32)
            $error("RING_WORD_WIDTH must match the ring module's fixed 32-bit data ports");
    end
`endif

endmodule

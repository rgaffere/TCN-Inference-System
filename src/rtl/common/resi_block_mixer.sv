// same as resi block except it does channel mixing
module resi_block_mixer #(
    parameter int NUM_CHANNELS = 16,
    parameter int W_BIT_WIDTH = 8,
    parameter int B_BIT_WIDTH = 32,
    parameter int KERNEL_LEN = 3,
    parameter int DILATION = 1,
    parameter int NUM_RINGS = 4,
    parameter int DEPTH = 512,
    localparam int ADDR_DEPTH = $clog2(DEPTH),
    localparam int K_IDX_DEPTH = $clog2(KERNEL_LEN),
    localparam int CH_IDX_DEPTH = $clog2(NUM_CHANNELS)
)(
    input logic clk, rst_n, valid_in,
    input var logic signed [W_BIT_WIDTH - 1: 0] inputVals [0: NUM_CHANNELS - 1],
    input var logic signed [W_BIT_WIDTH - 1: 0] weights1 [0: NUM_CHANNELS - 1][0: NUM_CHANNELS - 1][0: KERNEL_LEN - 1],
    input var logic signed [W_BIT_WIDTH - 1: 0] weights2 [0: NUM_CHANNELS - 1][0: NUM_CHANNELS - 1][0: KERNEL_LEN - 1],
    input var logic signed [B_BIT_WIDTH - 1: 0] bias1 [0: NUM_CHANNELS - 1],
    input var logic signed [B_BIT_WIDTH - 1: 0] bias2 [0: NUM_CHANNELS - 1],
    output var logic signed [W_BIT_WIDTH - 1: 0] outputVals [0: NUM_CHANNELS - 1],
    output logic valid_out
);
    // Some vars for control
    logic s0, s1, s2;
    logic mac1go, mac2go, mac1_active, mac2_active, mac1_start, mac2_start;

    logic [ADDR_DEPTH - 1: 0] currOffset1;
    logic [ADDR_DEPTH - 1: 0] currOffset2;

    logic signed [W_BIT_WIDTH - 1:0] macX1 [0:NUM_CHANNELS-1];
    logic signed [W_BIT_WIDTH - 1:0] macX2 [0:NUM_CHANNELS-1];

    logic signed [B_BIT_WIDTH - 1: 0] macOut1 [0: NUM_CHANNELS - 1];
    logic signed [B_BIT_WIDTH - 1: 0] reluOut1 [0: NUM_CHANNELS - 1];
    logic signed [W_BIT_WIDTH - 1: 0] quantOut1 [0: NUM_CHANNELS - 1];
    logic signed [W_BIT_WIDTH - 1: 0] quantOut1_reg [0: NUM_CHANNELS - 1];

    logic signed [B_BIT_WIDTH - 1: 0] macOut2 [0: NUM_CHANNELS - 1];
    logic signed [B_BIT_WIDTH - 1: 0] reluOut2 [0: NUM_CHANNELS - 1];
    logic signed [W_BIT_WIDTH - 1: 0] quantOut2 [0: NUM_CHANNELS - 1];

    // now longer signed since its packed
    logic [(NUM_CHANNELS / NUM_RINGS) * W_BIT_WIDTH - 1:0] ringIn1 [0: NUM_RINGS-1];
    logic [(NUM_CHANNELS / NUM_RINGS) * W_BIT_WIDTH - 1:0] ringOut1 [0: NUM_RINGS-1];

    logic [(NUM_CHANNELS / NUM_RINGS) * W_BIT_WIDTH - 1: 0] ringIn2 [0: NUM_RINGS-1];
    logic [(NUM_CHANNELS / NUM_RINGS) * W_BIT_WIDTH - 1: 0] ringOut2 [0: NUM_RINGS-1];

    logic signed [W_BIT_WIDTH - 1:0] ringVals1 [0:NUM_CHANNELS-1];
    logic signed [W_BIT_WIDTH - 1:0] ringVals2 [0:NUM_CHANNELS-1];

    logic signed [W_BIT_WIDTH - 1:0] selectedIn1;
    logic signed [W_BIT_WIDTH - 1:0] selectedIn2;

    logic signed [W_BIT_WIDTH-1:0] w1_tap [0:NUM_CHANNELS-1];
    logic signed [W_BIT_WIDTH-1:0] w2_tap [0:NUM_CHANNELS-1];

    // Three residual slots aligned to the two MAC stages and final output.
    localparam int RESI_STAGES = 3;
    logic signed [W_BIT_WIDTH - 1: 0] resi_reg [0: RESI_STAGES - 1][0: NUM_CHANNELS - 1];
    // Need this one for overflow guard
    logic signed [W_BIT_WIDTH:0] resi_sum [0: NUM_CHANNELS - 1];

    logic [K_IDX_DEPTH - 1: 0] tap1_idx;
    logic [K_IDX_DEPTH - 1: 0] tap2_idx;

    logic [CH_IDX_DEPTH - 1:0] in_ch1_idx;
    logic [CH_IDX_DEPTH - 1:0] in_ch2_idx;

    assign mac1_start = valid_in && !mac1_active;
    assign mac2_start = s1 && !mac2_active;

    assign mac1go = mac1_start | mac1_active;
    assign mac2go = mac2_start | mac2_active;

    // Init modules
    MAC_array #(
        .IN_LEN(W_BIT_WIDTH),
        .KERNEL_LEN(KERNEL_LEN * NUM_CHANNELS),
        .NUM_CHANNELS(NUM_CHANNELS)
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

    ReLU_array init_relu1 (
        .data_in(macOut1),
        .data_out(reluOut1)
    );

    quant_array init_quant1 (
        .data_in(reluOut1),
        .data_out(quantOut1)
    );

    MAC_array #(
        .IN_LEN(W_BIT_WIDTH),
        .KERNEL_LEN(KERNEL_LEN * NUM_CHANNELS),
        .NUM_CHANNELS(NUM_CHANNELS)
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

    ReLU_array init_relu2 (
        .data_in(macOut2),
        .data_out(reluOut2)
    );

    quant_array init_quant2 (
        .data_in(reluOut2),
        .data_out(quantOut2)
    );

    // Initialize rings modules for both hidden layers
    genvar i, j;
    generate
        for(i = 0; i < NUM_RINGS; i++) begin : gen_ring_banks

            for(j = 0; j < (NUM_CHANNELS / NUM_RINGS); j++) begin : gen_ring_lanes
                localparam int CH = i * (NUM_CHANNELS / NUM_RINGS) + j;
                localparam int LO = j * W_BIT_WIDTH;

                assign ringVals1[CH] = $signed(ringOut1[i][LO +: W_BIT_WIDTH]);
                assign ringVals2[CH] = $signed(ringOut2[i][LO +: W_BIT_WIDTH]);
    
                assign ringIn1[i][LO +: W_BIT_WIDTH] = inputVals[CH];
                assign ringIn2[i][LO +: W_BIT_WIDTH] = quantOut1_reg[CH];
            end

            ring #(
                .DEPTH(DEPTH),
                .DATA_LEN(W_BIT_WIDTH)
            ) ring_bank1 (
                .clk(clk),
                .rst_n(rst_n),
                .write_en(mac1_start),
                .data_in(ringIn1[i]),
                .read_offset(currOffset1),
                .data_out(ringOut1[i])
            );

            ring #(
                .DEPTH(DEPTH),
                .DATA_LEN(W_BIT_WIDTH)
            ) ring_bank2 (
                .clk(clk),
                .rst_n(rst_n),
                .write_en(s1),
                .data_in(ringIn2[i]),
                .read_offset(currOffset2),
                .data_out(ringOut2[i])
            );
        end
    endgenerate

    // MAC 1 control
    always_ff @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            mac1_active <= 1'b0;
            tap1_idx    <= '0;
            in_ch1_idx  <= '0;
        end else begin
            if(mac1_start) begin
                mac1_active <= 1'b1;
                tap1_idx    <= '0;
                in_ch1_idx  <= CH_IDX_DEPTH'(1);
            end else if(mac1_active) begin
                if(in_ch1_idx == NUM_CHANNELS - 1) begin
                    in_ch1_idx <= '0;

                    if(tap1_idx == KERNEL_LEN - 1) begin
                        mac1_active <= 1'b0;
                        tap1_idx    <= '0;
                    end else begin
                        tap1_idx <= tap1_idx + 1'b1;
                    end
                end else begin
                    in_ch1_idx <= in_ch1_idx + 1'b1;
                end
            end
        end
    end

    // MAC 2 control
    always_ff @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            mac2_active <= 1'b0;
            tap2_idx    <= '0;
            in_ch2_idx  <= '0;
        end else begin
            if(mac2_start) begin
                mac2_active <= 1'b1;
                tap2_idx    <= '0;
                in_ch2_idx  <= CH_IDX_DEPTH'(1);
            end else if(mac2_active) begin
                if(in_ch2_idx == NUM_CHANNELS - 1) begin
                    in_ch2_idx <= '0;

                    if(tap2_idx == KERNEL_LEN - 1) begin
                        mac2_active <= 1'b0;
                        tap2_idx    <= '0;
                    end else begin
                        tap2_idx <= tap2_idx + 1'b1;
                    end
                end else begin
                    in_ch2_idx <= in_ch2_idx + 1'b1;
                end
            end
        end
    end

    // Weight control
    always_comb begin
        for (int ch = 0; ch < NUM_CHANNELS; ch++) begin
            w1_tap[ch] = weights1[ch][in_ch1_idx][tap1_idx];
            w2_tap[ch] = weights2[ch][in_ch2_idx][tap2_idx];
        end
    end

    always_comb begin
        if(mac1_start) begin
            // Head has not advanced for this write yet
            currOffset1 = ADDR_DEPTH'(DILATION - 1);
        end else if(tap1_idx == 0) begin
            // Head has advanced; retain t-D
            currOffset1 = ADDR_DEPTH'(DILATION);
        end else if(tap1_idx == 1) begin
            if(in_ch1_idx == NUM_CHANNELS - 1)
                // Prefetch tap 2 on the final tap-1 channel
                currOffset1 = ADDR_DEPTH'(2 * DILATION);
            else
                currOffset1 = ADDR_DEPTH'(DILATION);
        end else begin
            // Retain t-2D throughout tap 2
            currOffset1 = ADDR_DEPTH'(2 * DILATION);
        end

        if(mac2_start) begin
            currOffset2 = ADDR_DEPTH'(DILATION - 1);
        end else if(tap2_idx == 0) begin
            currOffset2 = ADDR_DEPTH'(DILATION);
        end else if(tap2_idx == 1) begin
            if(in_ch2_idx == NUM_CHANNELS - 1)
                currOffset2 = ADDR_DEPTH'(2 * DILATION);
            else
                currOffset2 = ADDR_DEPTH'(DILATION);
        end else begin
            currOffset2 = ADDR_DEPTH'(2 * DILATION);
        end
    end

    // Residual pipeline control
    always_ff @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            for(int i = 0; i < RESI_STAGES; i++) begin
                for(int ch = 0; ch < NUM_CHANNELS; ch++) begin
                    resi_reg[i][ch] <= '0;
                end
            end
        end else begin

            // Capture sample entering MAC1.
            // Also reused as the MAC1 tap-0 hold vector.
            if(mac1_start) begin
                for(int ch = 0; ch < NUM_CHANNELS; ch++) begin
                    resi_reg[0][ch] <= inputVals[ch];
                end
            end

            // Preserve the residual before MAC1 releases and allows
            // a new sample to overwrite resi_reg[0].
            if(mac1_active &&
            tap1_idx == KERNEL_LEN - 1 &&
            in_ch1_idx == NUM_CHANNELS - 1) begin

                for(int ch = 0; ch < NUM_CHANNELS; ch++) begin
                    resi_reg[1][ch] <= resi_reg[0][ch];
                end
            end

            // Move the MAC2 residual one cycle after MAC2 starts.
            // This allows a simultaneous s2 to consume the old
            // resi_reg[2] value before it is replaced.
            if(mac2_active &&
            tap2_idx == 0 &&
            in_ch2_idx == CH_IDX_DEPTH'(1)) begin

                for(int ch = 0; ch < NUM_CHANNELS; ch++) begin
                    resi_reg[2][ch] <= resi_reg[1][ch];
                end
            end
        end
    end

    // residual add, with 9 bits for saturation guard
    always_comb begin
        for (int ch = 0; ch < NUM_CHANNELS; ch++) begin
            resi_sum[ch] = $signed({quantOut2[ch][W_BIT_WIDTH-1], quantOut2[ch]}) + $signed({resi_reg[RESI_STAGES - 1][ch][W_BIT_WIDTH-1], resi_reg[RESI_STAGES - 1][ch]});
        end
    end

    always_comb begin
        if(tap1_idx == 0) begin
            if(mac1_start) begin 
                selectedIn1 = inputVals[in_ch1_idx];
            end else begin
                selectedIn1 = resi_reg[0][in_ch1_idx];
            end
        end else begin
            selectedIn1 = ringVals1[in_ch1_idx];
        end

        if(tap2_idx == 0) begin
            selectedIn2 = quantOut1_reg[in_ch2_idx];
        end else begin
            selectedIn2 = ringVals2[in_ch2_idx];
        end
    end

    always_comb begin
        for (int out_ch = 0; out_ch < NUM_CHANNELS; out_ch++) begin
            macX1[out_ch] = selectedIn1;
            macX2[out_ch] = selectedIn2;
        end
    end

    // State control and quant registering
    always_ff @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            s1 <= 1'b0;
            valid_out <= 1'b0;
            for (int ch = 0; ch < NUM_CHANNELS; ch++) begin
                quantOut1_reg[ch] <= '0;
                outputVals[ch] <= '0;
            end
        end else begin
            s1 <= s0;
            valid_out <= s2;

            if(s0) begin
                quantOut1_reg <= quantOut1;
            end

            if(s2) begin
                for (int ch = 0; ch < NUM_CHANNELS; ch++) begin
                    // overflow guards, TODO: parametrize the borders
                    if (resi_sum[ch] > 9'sd127) begin
                        outputVals[ch] <= 8'sd127;
                    end else if (resi_sum[ch] < -9'sd128) begin
                        outputVals[ch] <= -8'sd128;
                    end else begin
                        outputVals[ch] <= resi_sum[ch][W_BIT_WIDTH-1:0];
                    end
                end
            end
        end
    end


endmodule

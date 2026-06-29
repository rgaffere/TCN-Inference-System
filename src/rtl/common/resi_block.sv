module resi_block #(
    parameter int NUM_CHANNELS = 16,
    parameter int W_BIT_WIDTH = 8,
    parameter int B_BIT_WIDTH = 32,
    parameter int KERNEL_LEN = 3,
    parameter int DILATION = 1,
    parameter int NUM_RINGS = 4,
    parameter int DEPTH = 512,
    localparam int ADDR_WIDTH = $clog2(DEPTH)
)(
    input logic clk, rst_n, valid_in,
    input var logic signed [W_BIT_WIDTH - 1: 0] inputVals [0: NUM_CHANNELS - 1],
    input var logic signed [W_BIT_WIDTH - 1: 0] weights1 [0: NUM_CHANNELS - 1][0: KERNEL_LEN - 1],
    input var logic signed [W_BIT_WIDTH - 1: 0] weights2 [0: NUM_CHANNELS - 1][0: KERNEL_LEN - 1],
    input var logic signed [B_BIT_WIDTH - 1: 0] bias1 [0: NUM_CHANNELS - 1],
    input var logic signed [B_BIT_WIDTH - 1: 0] bias2 [0: NUM_CHANNELS - 1],
    output var logic signed [W_BIT_WIDTH - 1: 0] outputVals [0: NUM_CHANNELS - 1],
    output logic valid_out
);
    // Some vars for control
    logic s0, s1, s2;
    logic mac1go, mac2go, mac1_active, mac2_active, mac1_start, mac2_start;
    logic useReadVals1, useReadVals2;

    logic [ADDR_WIDTH - 1: 0] currOffset1;
    logic [ADDR_WIDTH - 1: 0] currOffset2;

    logic signed [W_BIT_WIDTH - 1: 0] macIn1 [0: NUM_CHANNELS - 1];
    logic signed [B_BIT_WIDTH - 1: 0] macOut1 [0: NUM_CHANNELS - 1];
    logic signed [B_BIT_WIDTH - 1: 0] reluOut1 [0: NUM_CHANNELS - 1];
    logic signed [W_BIT_WIDTH - 1: 0] quantOut1 [0: NUM_CHANNELS - 1];
    logic signed [W_BIT_WIDTH - 1: 0] quantOut1_reg [0: NUM_CHANNELS - 1];

    logic signed [W_BIT_WIDTH - 1:0] macIn2 [0:NUM_CHANNELS-1];
    logic signed [B_BIT_WIDTH - 1: 0] macOut2 [0: NUM_CHANNELS - 1];
    logic signed [B_BIT_WIDTH - 1: 0] reluOut2 [0: NUM_CHANNELS - 1];
    logic signed [W_BIT_WIDTH - 1: 0] quantOut2 [0: NUM_CHANNELS - 1];

    logic signed [W_BIT_WIDTH-1:0] ringIn1  [0:NUM_RINGS-1][0:NUM_RINGS-1];
    logic signed [W_BIT_WIDTH-1:0] ringOut1 [0:NUM_RINGS-1][0:NUM_RINGS-1];

    logic signed [W_BIT_WIDTH-1:0] ringIn2  [0:NUM_RINGS-1][0:NUM_RINGS-1];
    logic signed [W_BIT_WIDTH-1:0] ringOut2 [0:NUM_RINGS-1][0:NUM_RINGS-1];

    logic signed [W_BIT_WIDTH - 1: 0] ringOut1_flat [0: NUM_CHANNELS - 1];
    logic signed [W_BIT_WIDTH - 1: 0] ringOut2_flat [0: NUM_CHANNELS - 1];

    logic signed [W_BIT_WIDTH-1:0] w1_tap [0:NUM_CHANNELS-1];
    logic signed [W_BIT_WIDTH-1:0] w2_tap [0:NUM_CHANNELS-1];

    logic signed [W_BIT_WIDTH - 1: 0] resi_reg [0: KERNEL_LEN - 1][0: NUM_CHANNELS - 1];
    // Need this one for overflow guard
    logic signed [W_BIT_WIDTH:0] resi_sum [0: NUM_CHANNELS - 1];

    logic [$clog2(KERNEL_LEN) - 1: 0] tap1_idx;
    logic [$clog2(KERNEL_LEN) - 1: 0] tap2_idx;

    assign mac1_start = valid_in && !mac1_active;
    assign mac2_start = s1 && !mac2_active;

    assign mac1go = mac1_start | mac1_active;
    assign mac2go = mac2_start | mac2_active;

    assign useReadVals1 = mac1_active;
    assign useReadVals2 = mac2_active;

    // Init modules
    MAC_array init_mac1 (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(mac1go),
        .x(macIn1),
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

    MAC_array init_mac2 (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(mac2go),
        .x(macIn2),
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

            for(j = 0; j < NUM_RINGS; j++) begin : gen_ring_lanes
                localparam int CH = i * NUM_RINGS + j;

                assign ringIn1[i][j] = inputVals[CH];
                assign ringIn2[i][j] = quantOut1_reg[CH];
                assign ringOut1_flat[CH] = ringOut1[i][j];
                assign ringOut2_flat[CH] = ringOut2[i][j];
            end

            ring ring_bank1 (
                .clk(clk),
                .rst_n(rst_n),
                .write_en(mac1_start),
                .data_in(ringIn1[i]),
                .read_offset(currOffset1),
                .data_out(ringOut1[i])
            );

            ring ring_bank2 (
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
            tap1_idx <= '0;
        end else begin
            if(mac1_start) begin
                mac1_active <= 1'b1;
                tap1_idx <= 1;
            end else if(mac1_active) begin
                if(tap1_idx == KERNEL_LEN - 1) begin
                    mac1_active <= 1'b0;
                    tap1_idx <= '0;
                end else begin
                    tap1_idx <= tap1_idx + 1'b1;
                end
            end
        end
    end

    // MAC 2 control
    always_ff @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            mac2_active <= 1'b0;
            tap2_idx <= '0;
        end else begin
            if(mac2_start) begin
                mac2_active <= 1'b1;
                tap2_idx <= 1;
            end else if(mac2_active) begin
                if(tap2_idx == KERNEL_LEN - 1) begin
                    mac2_active <= 1'b0;
                    tap2_idx <= '0;
                end else begin
                    tap2_idx <= tap2_idx + 1'b1;
                end
            end
        end
    end

    // MAC 1 input control
    always_comb begin
        for (int ch = 0; ch < NUM_CHANNELS; ch++) begin
            macIn1[ch] = useReadVals1 ? ringOut1_flat[ch] : inputVals[ch];
        end
    end

    // MAC 2 input control
    always_comb begin
        for (int ch = 0; ch < NUM_CHANNELS; ch++) begin
            macIn2[ch] = useReadVals2 ? ringOut2_flat[ch] : quantOut1_reg[ch];
        end
    end

    // Weight control
    always_comb begin
        for (int ch = 0; ch < NUM_CHANNELS; ch++) begin
            w1_tap[ch] = weights1[ch][tap1_idx];
            w2_tap[ch] = weights2[ch][tap2_idx];
        end
    end

    // Offset controller, we want to look ahead to save a cycle
    // Gotta use - 1 on the first offest since we havent written yet
    always_comb begin
        case (tap1_idx)
            0: currOffset1 = ADDR_WIDTH'(DILATION - 1);
            1: currOffset1 = ADDR_WIDTH'(2 * DILATION);
            default: currOffset1 = '0;
        endcase

        case (tap2_idx)
            0: currOffset2 = ADDR_WIDTH'(DILATION - 1);
            1: currOffset2 = ADDR_WIDTH'(2 * DILATION);
            default: currOffset2 = '0;
        endcase
    end

    // residual register control
    always_ff @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            // im hardcoding this with the assumption that KERNAL_LEN is always 3 for now
            // TODO parameterize this
            for (int ch = 0; ch < NUM_CHANNELS; ch++) begin
                resi_reg[0][ch] <= '0;
                resi_reg[1][ch] <= '0;
                resi_reg[2][ch] <= '0;
            end
        end else if(mac1_start) begin
            for (int ch = 0; ch < NUM_CHANNELS; ch++) begin
                resi_reg[2][ch] <= resi_reg[1][ch];
                resi_reg[1][ch] <= resi_reg[0][ch];
                resi_reg[0][ch] <= inputVals[ch];
            end
        end
    end

    // residual add
    always_comb begin
        for (int ch = 0; ch < NUM_CHANNELS; ch++) begin
            resi_sum[ch] = $signed(quantOut2[ch]) + $signed(resi_reg[KERNEL_LEN - 1][ch]);
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

`timescale 1ns/1ps

module resi_block_tb;

    localparam int NUM_CHANNELS = 16;
    localparam int W_BIT_WIDTH  = 8;
    localparam int B_BIT_WIDTH  = 32;
    localparam int KERNEL_LEN   = 3;
    localparam int DILATION     = 1;
    localparam int NUM_RINGS    = 4;
    localparam int DEPTH        = 64;
    localparam int SHIFT        = 8;
    localparam int CLK_PERIOD   = 10;
    localparam int MAX_DRAIN    = 128;

    localparam int PRIME_TOKENS = (2 * (KERNEL_LEN - 1) * DILATION) + KERNEL_LEN;
    localparam int FLUSH_TOKENS = KERNEL_LEN - 1;
    localparam int MAX_ERR_PRINTS = 120;

    typedef logic signed [W_BIT_WIDTH-1:0] chvec_t [0:NUM_CHANNELS-1];
    typedef logic signed [W_BIT_WIDTH-1:0] wmat_t  [0:NUM_CHANNELS-1][0:KERNEL_LEN-1];
    typedef logic signed [B_BIT_WIDTH-1:0] bvec_t  [0:NUM_CHANNELS-1];

    logic clk = 1'b0;
    logic rst_n;
    logic valid_in;

    chvec_t inputVals;
    wmat_t  weights1, weights2;
    bvec_t  bias1, bias2;
    chvec_t outputVals;
    logic   valid_out;

    resi_block #(
        .NUM_CHANNELS(NUM_CHANNELS),
        .W_BIT_WIDTH(W_BIT_WIDTH),
        .B_BIT_WIDTH(B_BIT_WIDTH),
        .KERNEL_LEN(KERNEL_LEN),
        .DILATION(DILATION),
        .NUM_RINGS(NUM_RINGS),
        .DEPTH(DEPTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .inputVals(inputVals),
        .weights1(weights1),
        .weights2(weights2),
        .bias1(bias1),
        .bias2(bias2),
        .outputVals(outputVals),
        .valid_out(valid_out)
    );

    wire dut_accept = dut.mac1_start;

    // assertions
    property valid_out_low_during_reset;
        @(posedge clk)
        !rst_n |-> !valid_out;
    endproperty

    assert property(valid_out_low_during_reset)
        else $fatal(1, "valid_out is high during the reset");

    property valid_out_known;
        @(posedge clk)
        disable iff(!rst_n)
        !$isunknown(valid_out);
    endproperty

        assert property(valid_out_known)
            else $fatal(1, "valid_out is unknown");

    genvar sva_ch;
    generate
        for (sva_ch = 0; sva_ch < NUM_CHANNELS; sva_ch++) begin : sva_output_known
            property output_known_when_valid;
                @(posedge clk)
                disable iff (!rst_n)
                valid_out |-> !$isunknown(outputVals[sva_ch]);
            endproperty

            assert property (output_known_when_valid)
                else $fatal(1, "outputVals[%0d] is unknown when valid_out is high", sva_ch);
        end
    endgenerate

    typedef struct {
        int token_id;
        longint accept_cycle;
        bit check_en;
    } pending_t;

    longint cycle_num;
    longint latency_min, latency_max;

    int token_count;
    int total_accepted;
    int checked_tokens;
    int unchecked_tokens;
    int check_pass;
    int check_fail;
    int missing_outputs;
    int spurious_valid_out;
    int dropped_busy_pulses;
    int drain_timeouts;
    int err_prints;

    string phase_name;
    bit next_check_en;

    chvec_t tok_in  [int];
    chvec_t tok_q1  [int];
    chvec_t tok_q2  [int];
    chvec_t tok_exp [int];

    pending_t pending_q[$];

    always #(CLK_PERIOD/2) clk = ~clk;

    initial begin
        if ($test$plusargs("VCD")) begin
            $dumpfile("resi_block_tb.vcd");
            $dumpvars(0, resi_block_tb);
        end
    end

    initial begin
        if (NUM_CHANNELS != NUM_RINGS * NUM_RINGS) begin
            $fatal(1, "NUM_CHANNELS must equal NUM_RINGS*NUM_RINGS");
        end

        if (KERNEL_LEN < 2) begin
            $fatal(1, "KERNEL_LEN must be at least 2");
        end
    end

    function automatic logic signed [W_BIT_WIDTH-1:0] s8(input int signed v);
        int signed x;
        begin
            x = v;
            while (x > 127)  x -= 256;
            while (x < -128) x += 256;
            return x[W_BIT_WIDTH-1:0];
        end
    endfunction

    function automatic logic signed [W_BIT_WIDTH-1:0] sat8(input int signed v);
        if (v > 127)       return 8'sd127;
        else if (v < -128) return -8'sd128;
        else               return v[W_BIT_WIDTH-1:0];
    endfunction

    function automatic logic signed [W_BIT_WIDTH-1:0] relu_quant(input int signed v);
        int signed shifted;
        begin
            if (v < 0) v = 0;
            shifted = v >>> SHIFT;
            return sat8(shifted);
        end
    endfunction

    function automatic chvec_t zero_vec();
        chvec_t z;
        for (int ch = 0; ch < NUM_CHANNELS; ch++) z[ch] = '0;
        return z;
    endfunction

    function automatic chvec_t const_vec(input int signed val);
        chvec_t v;
        for (int ch = 0; ch < NUM_CHANNELS; ch++) v[ch] = s8(val);
        return v;
    endfunction

    function automatic chvec_t ramp_vec(input int id);
        chvec_t v;
        int signed x;

        for (int ch = 0; ch < NUM_CHANNELS; ch++) begin
            x = ((id * 9) + (ch * 5)) % 96 - 48;
            v[ch] = s8(x);
        end

        return v;
    endfunction

    function automatic chvec_t stress_vec(input int id, input int mode);
        chvec_t v;
        int signed x;

        for (int ch = 0; ch < NUM_CHANNELS; ch++) begin
            case (mode)
                0: x = ((id * 9) + (ch * 5)) % 96 - 48;
                1: begin
                    case (ch % 4)
                        0: x = 127;
                        1: x = -128;
                        2: x = 64 - id;
                        default: x = -32 + id;
                    endcase
                end
                2: x = $urandom_range(0, 255) - 128;
                3: x = (ch < 8) ? 127 : -128;
                default: x = id * 7 + ch * 13;
            endcase

            v[ch] = s8(x);
        end

        return v;
    endfunction

    task automatic clear_vec(output chvec_t v);
        for (int ch = 0; ch < NUM_CHANNELS; ch++) v[ch] = '0;
    endtask

    task automatic clear_weights;
        for (int ch = 0; ch < NUM_CHANNELS; ch++) begin
            bias1[ch] = '0;
            bias2[ch] = '0;

            for (int k = 0; k < KERNEL_LEN; k++) begin
                weights1[ch][k] = '0;
                weights2[ch][k] = '0;
            end
        end
    endtask

    task automatic reset_model;
        token_count = 0;
        next_check_en = 1'b0;
        tok_in.delete();
        tok_q1.delete();
        tok_q2.delete();
        tok_exp.delete();
        pending_q.delete();
    endtask

    task automatic build_expected(input int id, input chvec_t v);
        int signed acc;
        int hist_id;

        tok_in[id] = v;

        for (int ch = 0; ch < NUM_CHANNELS; ch++) begin
            acc = bias1[ch];

            for (int k = 0; k < KERNEL_LEN; k++) begin
                hist_id = id - (k * DILATION);

                if (hist_id >= 0) begin
                    acc += int'($signed(tok_in[hist_id][ch])) *
                           int'($signed(weights1[ch][k]));
                end
            end

            tok_q1[id][ch] = relu_quant(acc);
        end

        for (int ch = 0; ch < NUM_CHANNELS; ch++) begin
            acc = bias2[ch];

            for (int k = 0; k < KERNEL_LEN; k++) begin
                hist_id = id - (k * DILATION);

                if (hist_id >= 0) begin
                    acc += int'($signed(tok_q1[hist_id][ch])) *
                           int'($signed(weights2[ch][k]));
                end
            end

            tok_q2[id][ch] = relu_quant(acc);
            tok_exp[id][ch] = sat8(int'($signed(tok_q2[id][ch])) +
                                   int'($signed(tok_in[id][ch])));
        end
    endtask

    task automatic set_mixed_weights;
        for (int ch = 0; ch < NUM_CHANNELS; ch++) begin
            weights1[ch][0] = s8(20 + (ch % 7));
            weights1[ch][1] = s8(-12 + (ch % 5));
            weights1[ch][2] = s8(7 - (ch % 6));

            weights2[ch][0] = s8(16 - (ch % 4));
            weights2[ch][1] = s8(-10 + (ch % 3));
            weights2[ch][2] = s8(9 + (ch % 5));

            bias1[ch] = (ch % 2) ? -32'sd384 : 32'sd768 + ch;
            bias2[ch] = (ch % 3) ?  32'sd512 : -32'sd256;
        end
    endtask

    task automatic set_random_weights(input int unsigned seed);
        int signed x;

        void'($urandom(seed));

        for (int ch = 0; ch < NUM_CHANNELS; ch++) begin
            for (int k = 0; k < KERNEL_LEN; k++) begin
                x = $urandom_range(0, 63) - 32;
                weights1[ch][k] = s8(x);

                x = $urandom_range(0, 63) - 32;
                weights2[ch][k] = s8(x);
            end

            bias1[ch] = $urandom_range(0, 4095) - 2048;
            bias2[ch] = $urandom_range(0, 4095) - 2048;
        end
    endtask

    task automatic set_saturation_weights;
        for (int ch = 0; ch < NUM_CHANNELS; ch++) begin
            weights1[ch][0] = 8'sd127;
            weights1[ch][1] = 8'sd64;
            weights1[ch][2] = 8'sd32;

            weights2[ch][0] = 8'sd127;
            weights2[ch][1] = 8'sd127;
            weights2[ch][2] = 8'sd127;

            bias1[ch] = 32'sd32767;
            bias2[ch] = 32'sd32767;
        end
    endtask

    task automatic set_tap_isolation(input int tap);
        for (int ch = 0; ch < NUM_CHANNELS; ch++) begin
            bias1[ch] = '0;
            bias2[ch] = '0;

            for (int k = 0; k < KERNEL_LEN; k++) begin
                weights1[ch][k] = (k == tap) ? 8'sd64 : 8'sd0;
                weights2[ch][k] = (k == 0)   ? 8'sd64 : 8'sd0;
            end
        end
    endtask

    task automatic set_quant2_bias(input int signed q2);
        clear_weights();

        for (int ch = 0; ch < NUM_CHANNELS; ch++) begin
            bias2[ch] = q2 <<< SHIFT;
        end
    endtask

    task automatic do_reset(input int cycles);
        @(negedge clk);
        rst_n = 1'b0;
        valid_in = 1'b0;
        clear_vec(inputVals);
        reset_model();

        repeat (cycles) @(posedge clk);

        @(negedge clk);
        rst_n = 1'b1;

        @(posedge clk);
        #2;
    endtask

    task automatic issue_token(input chvec_t v, input bit check_en);
        @(negedge clk);
        inputVals = v;
        next_check_en = check_en;
        valid_in = 1'b1;

        do @(posedge clk); while (!dut_accept);

        @(negedge clk);
        valid_in = 1'b0;
        next_check_en = 1'b0;
    endtask

    task automatic drain_outputs(input int max_cycles);
        int n;

        n = 0;

        while ((pending_q.size() != 0) && (n < max_cycles)) begin
            @(posedge clk);
            #2;
            n++;
        end

        if (pending_q.size() != 0) begin
            drain_timeouts++;
            missing_outputs += pending_q.size();
            $error("[%s] drain timeout with %0d outputs still pending", phase_name, pending_q.size());
            pending_q.delete();
        end

        repeat (2) @(posedge clk);
    endtask

    task automatic issue_stream_with_zero_prime(
        input int prime_count,
        input int real_count,
        input int base_id,
        input int mode
    );
        for (int i = 0; i < prime_count; i++) begin
            issue_token(zero_vec(), 1'b0);
        end

        for (int i = 0; i < real_count; i++) begin
            issue_token(stress_vec(base_id + i, mode), 1'b1);
        end

        for (int i = 0; i < FLUSH_TOKENS; i++) begin
            issue_token(zero_vec(), 1'b0);
        end

        drain_outputs(MAX_DRAIN);
    endtask

    task automatic issue_const_stream_with_zero_prime(
        input int prime_count,
        input int real_count,
        input int signed val
    );
        for (int i = 0; i < prime_count; i++) begin
            issue_token(zero_vec(), 1'b0);
        end

        for (int i = 0; i < real_count; i++) begin
            issue_token(const_vec(val), 1'b1);
        end

        for (int i = 0; i < FLUSH_TOKENS; i++) begin
            issue_token(zero_vec(), 1'b0);
        end

        drain_outputs(MAX_DRAIN);
    endtask

    task automatic issue_dense_stream_with_busy_pulse;
        for (int i = 0; i < PRIME_TOKENS; i++) begin
            issue_token(zero_vec(), 1'b0);
        end

        issue_token(stress_vec(6000, 0), 1'b1);

        @(negedge clk);
        inputVals = stress_vec(6001, 3);
        valid_in = 1'b1;

        @(posedge clk);
        #1;

        if (dut_accept) begin
            $error("[%s] busy pulse was accepted unexpectedly", phase_name);
        end else begin
            dropped_busy_pulses++;
        end

        @(negedge clk);
        valid_in = 1'b0;

        for (int i = 1; i < 32; i++) begin
            issue_token(stress_vec(6000 + i, 0), 1'b1);
        end

        for (int i = 0; i < FLUSH_TOKENS; i++) begin
            issue_token(zero_vec(), 1'b0);
        end

        drain_outputs(MAX_DRAIN);
    endtask

    task automatic start_phase(input string name);
        phase_name = name;
        err_prints = 0;
        $display("%s", name);
    endtask

    always @(posedge clk) begin
        if (rst_n && dut_accept) begin
            int id;

            id = token_count;
            build_expected(id, inputVals);

            pending_q.push_back('{token_id:id, accept_cycle:cycle_num, check_en:next_check_en});

            token_count++;
            total_accepted++;
        end

        #1;

        if (rst_n && valid_out) begin
            if (pending_q.size() == 0) begin
                spurious_valid_out++;
                $error("[%s][cycle %0d] spurious valid_out", phase_name, cycle_num);
            end else begin
                pending_t chk;
                longint lat;

                chk = pending_q.pop_front();
                lat = cycle_num - chk.accept_cycle;

                if ((latency_min < 0) || (lat < latency_min)) latency_min = lat;
                if ((latency_max < 0) || (lat > latency_max)) latency_max = lat;

                if (chk.check_en) begin
                    checked_tokens++;

                    for (int ch = 0; ch < NUM_CHANNELS; ch++) begin
                        if (outputVals[ch] !== tok_exp[chk.token_id][ch]) begin
                            check_fail++;

                            if (err_prints < MAX_ERR_PRINTS) begin
                                err_prints++;
                                $error("[%s][cycle %0d][tok %0d][ch %0d] dut=%0d exp=%0d latency=%0d in=%0d q1=%0d q2=%0d",
                                       phase_name,
                                       cycle_num,
                                       chk.token_id,
                                       ch,
                                       outputVals[ch],
                                       tok_exp[chk.token_id][ch],
                                       lat,
                                       tok_in[chk.token_id][ch],
                                       tok_q1[chk.token_id][ch],
                                       tok_q2[chk.token_id][ch]);
                            end
                        end else begin
                            check_pass++;
                        end
                    end
                end else begin
                    unchecked_tokens++;
                end
            end
        end

        cycle_num++;
    end

    initial begin
        cycle_num = 0;
        latency_min = -1;
        latency_max = -1;

        token_count = 0;
        total_accepted = 0;
        checked_tokens = 0;
        unchecked_tokens = 0;
        check_pass = 0;
        check_fail = 0;
        missing_outputs = 0;
        spurious_valid_out = 0;
        dropped_busy_pulses = 0;
        drain_timeouts = 0;
        err_prints = 0;
        next_check_en = 1'b0;
        phase_name = "init";

        rst_n = 1'b0;
        valid_in = 1'b0;
        clear_vec(inputVals);
        clear_weights();

        $display("\n=== resi_block cadence-locked bit-exact TB ===");
        $display("NUM_CHANNELS=%0d KERNEL_LEN=%0d DILATION=%0d DEPTH=%0d", NUM_CHANNELS, KERNEL_LEN, DILATION, DEPTH);
        $display("PRIME_TOKENS=%0d FLUSH_TOKENS=%0d\n", PRIME_TOKENS, FLUSH_TOKENS);

        start_phase("PHASE 0: reset");
        do_reset(5);

        if (valid_out !== 1'b0) begin
            $error("valid_out not low after reset");
        end

        for (int ch = 0; ch < NUM_CHANNELS; ch++) begin
            if (outputVals[ch] !== '0) begin
                $error("outputVals[%0d] not zero after reset", ch);
            end
        end

        start_phase("PHASE 1: dense zero-weight residual alignment");
        clear_weights();
        do_reset(5);
        issue_stream_with_zero_prime(PRIME_TOKENS, 48, 1000, 0);

        start_phase("PHASE 2: constant arithmetic sanity, no history ambiguity");
        for (int tap = 0; tap < KERNEL_LEN; tap++) begin
            set_tap_isolation(tap);
            do_reset(5);
            issue_const_stream_with_zero_prime(PRIME_TOKENS, 32, 64);
        end

        start_phase("PHASE 3: tap isolation with zero primed history");
        for (int tap = 0; tap < KERNEL_LEN; tap++) begin
            set_tap_isolation(tap);
            do_reset(5);
            issue_stream_with_zero_prime(PRIME_TOKENS, 48, 2000 + tap*100, 0);
        end

        start_phase("PHASE 4: mixed deterministic dense stream");
        set_mixed_weights();
        do_reset(5);
        issue_stream_with_zero_prime(PRIME_TOKENS, 64, 3000, 2);

        start_phase("PHASE 5: saturation boundaries");
        set_quant2_bias(127);
        do_reset(5);
        issue_stream_with_zero_prime(PRIME_TOKENS, 32, 4000, 3);

        set_quant2_bias(0);
        do_reset(5);
        issue_stream_with_zero_prime(PRIME_TOKENS, 32, 4100, 3);

        set_saturation_weights();
        do_reset(5);
        issue_stream_with_zero_prime(PRIME_TOKENS, 32, 4200, 3);

        start_phase("PHASE 6: randomized dense regression");
        for (int i = 0; i < 8; i++) begin
            set_random_weights(32'hCAFE0000 + i);
            do_reset(5);
            issue_stream_with_zero_prime(PRIME_TOKENS, 48, 5000 + i*100, i % 4);
        end

        // note really relevant since im gonna gate the inputs upstream
        // start_phase("PHASE 7: busy valid pulse ignored, then dense recovery");
        // set_mixed_weights();
        // do_reset(5);
        // issue_dense_stream_with_busy_pulse();

        start_phase("PHASE 8: mid-stream reset recovery");
        set_random_weights(32'h0BADF00D);
        do_reset(5);

        for (int i = 0; i < 8; i++) begin
            issue_token(stress_vec(7000 + i, 2), 1'b0);
        end

        repeat (3) @(posedge clk);
        do_reset(5);
        issue_stream_with_zero_prime(PRIME_TOKENS, 48, 7100, 2);

        drain_outputs(MAX_DRAIN);

        $display("\n=== SUMMARY ===");
        $display("accepted tokens       : %0d", total_accepted);
        $display("checked tokens        : %0d", checked_tokens);
        $display("unchecked outputs     : %0d", unchecked_tokens);
        $display("datapath checks       : pass=%0d fail=%0d", check_pass, check_fail);
        $display("valid_out             : missing=%0d spurious=%0d", missing_outputs, spurious_valid_out);
        //$display("busy pulses ignored   : %0d", dropped_busy_pulses);
        $display("drain timeouts        : %0d", drain_timeouts);

        if (latency_min >= 0) begin
            $display("measured latency      : min=%0d max=%0d cycles", latency_min, latency_max);
        end

        if (check_fail || missing_outputs || spurious_valid_out || drain_timeouts) begin
            $fatal(1, "FAIL");
        end

        $display("PASS");
        $finish;
    end

    initial begin
        #(CLK_PERIOD * 200000);
        $fatal(1, "watchdog timeout");
    end

endmodule

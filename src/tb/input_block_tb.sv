`timescale 1ns/1ps

module input_block_tb;

    localparam int IN_CHANNELS      = 6;
    localparam int OUT_CHANNELS     = 16;
    localparam int W_BIT_WIDTH      = 8;
    localparam int B_BIT_WIDTH      = 32;
    localparam int KERNEL_LEN       = 3;
    localparam int DILATION         = 1;
    localparam int DEPTH            = 512;
    localparam int NUM_RINGS        = 4;
    localparam int QUANT_SHIFT      = 8;
    localparam int CLK_PERIOD       = 10;
    localparam int EXPECTED_LATENCY =
        (KERNEL_LEN * IN_CHANNELS) +
        (KERNEL_LEN * OUT_CHANNELS) + 3;

    // Eight independent randomized weight sets. The final phase is longer
    // than DEPTH so the ring-buffer head wraps during the regression.
    parameter int NUM_RANDOM_PHASES = 8;
    parameter int SAMPLES_PER_PHASE = 64;
    parameter int WRAP_PHASE_SAMPLES = DEPTH + 96;
    parameter int unsigned RANDOM_SEED = 32'h494E_5054;

    logic clk;
    logic rst_n;
    logic valid_in;

    logic signed [W_BIT_WIDTH-1:0] inputVals [0:IN_CHANNELS-1];
    logic signed [W_BIT_WIDTH-1:0]
        weights1 [0:OUT_CHANNELS-1][0:IN_CHANNELS-1][0:KERNEL_LEN-1];
    logic signed [W_BIT_WIDTH-1:0]
        weights2 [0:OUT_CHANNELS-1][0:OUT_CHANNELS-1][0:KERNEL_LEN-1];
    logic signed [W_BIT_WIDTH-1:0]
        residual_weights [0:OUT_CHANNELS-1][0:IN_CHANNELS-1];
    logic signed [B_BIT_WIDTH-1:0] bias1 [0:OUT_CHANNELS-1];
    logic signed [B_BIT_WIDTH-1:0] bias2 [0:OUT_CHANNELS-1];
    logic signed [B_BIT_WIDTH-1:0] residual_bias [0:OUT_CHANNELS-1];
    logic signed [W_BIT_WIDTH-1:0] outputVals [0:OUT_CHANNELS-1];
    logic valid_out;

    int signed stimulus [0:IN_CHANNELS-1];
    int signed expected [0:OUT_CHANNELS-1];

    // Circular histories used only by the independent reference model.
    int signed input_history [0:DEPTH-1][0:IN_CHANNELS-1];
    int signed conv1_history [0:DEPTH-1][0:OUT_CHANNELS-1];
    int reference_sample_count;

    int unsigned rng_state;
    int error_count;
    int completed_samples;
    longint unsigned cycle_count;

    input_block #(
        .IN_CHANNELS(IN_CHANNELS),
        .OUT_CHANNELS(OUT_CHANNELS),
        .W_BIT_WIDTH(W_BIT_WIDTH),
        .B_BIT_WIDTH(B_BIT_WIDTH),
        .KERNEL_LEN(KERNEL_LEN),
        .DILATION(DILATION),
        .DEPTH(DEPTH),
        .NUM_RINGS(NUM_RINGS),
        .QUANT_SHIFT(QUANT_SHIFT)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .inputVals(inputVals),
        .weights1(weights1),
        .weights2(weights2),
        .residual_weights(residual_weights),
        .bias1(bias1),
        .bias2(bias2),
        .residual_bias(residual_bias),
        .outputVals(outputVals),
        .valid_out(valid_out)
    );

    initial clk = 1'b0;
    always #(CLK_PERIOD/2) clk = ~clk;

    always @(posedge clk)
        cycle_count <= cycle_count + 1'b1;

`ifdef DUMP_WAVES
    initial begin
        $dumpfile("tb_input_block.vcd");
        $dumpvars(0, tb_input_block);
    end
`endif

    function automatic int unsigned next_random();
        begin
            rng_state = rng_state ^ (rng_state << 13);
            rng_state = rng_state ^ (rng_state >> 17);
            rng_state = rng_state ^ (rng_state << 5);
            next_random = rng_state;
        end
    endfunction

    function automatic int signed random_between(
        input int signed minimum,
        input int signed maximum
    );
        int unsigned span;
        begin
            span = maximum - minimum + 1;
            random_between = minimum + int'(next_random() % span);
        end
    endfunction

    function automatic int signed quantize_reference(input longint signed value);
        longint signed shifted;
        begin
            shifted = value >>> QUANT_SHIFT;
            if (shifted > 127)
                quantize_reference = 127;
            else if (shifted < -128)
                quantize_reference = -128;
            else
                quantize_reference = int'(shifted);
        end
    endfunction

    function automatic int signed saturate_int8(input longint signed value);
        begin
            if (value > 127)
                saturate_int8 = 127;
            else if (value < -128)
                saturate_int8 = -128;
            else
                saturate_int8 = int'(value);
        end
    endfunction

    function automatic int history_index(input int absolute_index);
        int reduced;
        begin
            reduced = absolute_index % DEPTH;
            if (reduced < 0)
                reduced += DEPTH;
            history_index = reduced;
        end
    endfunction

    task automatic clear_reference_history;
        begin
            reference_sample_count = 0;
            for (int address = 0; address < DEPTH; address++) begin
                for (int ic = 0; ic < IN_CHANNELS; ic++)
                    input_history[address][ic] = 0;
                for (int oc = 0; oc < OUT_CHANNELS; oc++)
                    conv1_history[address][oc] = 0;
            end
        end
    endtask

    task automatic clear_external_signals;
        begin
            for (int ic = 0; ic < IN_CHANNELS; ic++) begin
                stimulus[ic] = 0;
                inputVals[ic] = '0;
            end

            for (int oc = 0; oc < OUT_CHANNELS; oc++) begin
                bias1[oc] = '0;
                bias2[oc] = '0;
                residual_bias[oc] = '0;

                for (int ic = 0; ic < IN_CHANNELS; ic++) begin
                    residual_weights[oc][ic] = '0;
                    for (int tap = 0; tap < KERNEL_LEN; tap++)
                        weights1[oc][ic][tap] = '0;
                end

                for (int ic = 0; ic < OUT_CHANNELS; ic++)
                    for (int tap = 0; tap < KERNEL_LEN; tap++)
                        weights2[oc][ic][tap] = '0;
            end
        end
    endtask

    task automatic apply_reset;
        begin
            valid_in = 1'b0;
            rst_n = 1'b0;
            for (int ic = 0; ic < IN_CHANNELS; ic++)
                inputVals[ic] = '0;
            clear_reference_history();

            repeat (4) @(posedge clk);
            @(negedge clk);
            rst_n = 1'b1;
            repeat (2) @(posedge clk);
            #1;

            if (valid_out !== 1'b0) begin
                $error("valid_out was not cleared by reset");
                error_count++;
            end

            for (int oc = 0; oc < OUT_CHANNELS; oc++) begin
                if ($signed(outputVals[oc]) !== 0) begin
                    $error("outputVals[%0d] was not cleared by reset: %0d",
                           oc, $signed(outputVals[oc]));
                    error_count++;
                end
            end
        end
    endtask

    // Independent fixed-point reference for one accepted input sample:
    //   Conv1 6->16, ReLU, quantize
    //   Conv2 16->16, ReLU, quantize
    //   Residual projection 6->16, quantize without ReLU
    //   Saturated Conv2 + projected residual
    task automatic reference_model_step;
        longint signed accumulator;
        longint signed residual_accumulator;
        int signed conv1_now [0:OUT_CHANNELS-1];
        int source_absolute;
        int source_address;
        int current_address;
        int delay;
        begin
            current_address = history_index(reference_sample_count);

            // Store the current raw sample before evaluating tap zero.
            for (int ic = 0; ic < IN_CHANNELS; ic++)
                input_history[current_address][ic] = stimulus[ic];

            for (int oc = 0; oc < OUT_CHANNELS; oc++) begin
                accumulator = $signed(bias1[oc]);

                for (int ic = 0; ic < IN_CHANNELS; ic++) begin
                    for (int tap = 0; tap < KERNEL_LEN; tap++) begin
                        delay = tap * DILATION;
                        source_absolute = reference_sample_count - delay;
                        if (source_absolute >= 0) begin
                            source_address = history_index(source_absolute);
                            accumulator +=
                                input_history[source_address][ic] *
                                $signed(weights1[oc][ic][tap]);
                        end
                    end
                end

                if (accumulator < 0)
                    accumulator = 0;
                conv1_now[oc] = quantize_reference(accumulator);
                conv1_history[current_address][oc] = conv1_now[oc];
            end

            for (int oc = 0; oc < OUT_CHANNELS; oc++) begin
                accumulator = $signed(bias2[oc]);

                for (int ic = 0; ic < OUT_CHANNELS; ic++) begin
                    for (int tap = 0; tap < KERNEL_LEN; tap++) begin
                        delay = tap * DILATION;
                        source_absolute = reference_sample_count - delay;
                        if (source_absolute >= 0) begin
                            source_address = history_index(source_absolute);
                            accumulator +=
                                conv1_history[source_address][ic] *
                                $signed(weights2[oc][ic][tap]);
                        end
                    end
                end

                if (accumulator < 0)
                    accumulator = 0;

                residual_accumulator = $signed(residual_bias[oc]);
                for (int ic = 0; ic < IN_CHANNELS; ic++)
                    residual_accumulator +=
                        stimulus[ic] * $signed(residual_weights[oc][ic]);

                expected[oc] = saturate_int8(
                    quantize_reference(accumulator) +
                    quantize_reference(residual_accumulator)
                );
            end

            reference_sample_count++;
        end
    endtask

    task automatic drive_sample(input int phase_id, input int sample_id);
        longint unsigned accept_cycle;
        longint signed latency;
        int timeout_cycles;
        int signed actual;
        begin
            // Freeze the independently computed expected result before valid_in.
            reference_model_step();

            @(negedge clk);
            for (int ic = 0; ic < IN_CHANNELS; ic++)
                inputVals[ic] = stimulus[ic];
            valid_in = 1'b1;

            @(posedge clk);
            #1;
            accept_cycle = cycle_count;

            @(negedge clk);
            valid_in = 1'b0;

            // Change the external input bus after acceptance to verify that the
            // DUT uses its internal hold register for the serialized channels.
            for (int ic = 0; ic < IN_CHANNELS; ic++)
                inputVals[ic] = random_between(-128, 127);

            timeout_cycles = 0;
            while ((valid_out !== 1'b1) && (timeout_cycles < 200)) begin
                @(posedge clk);
                #1;
                timeout_cycles++;
            end

            if (valid_out !== 1'b1) begin
                $error("Phase %0d sample %0d timed out", phase_id, sample_id);
                error_count++;
                return;
            end

            latency = cycle_count - accept_cycle;
            if (latency != EXPECTED_LATENCY) begin
                $error("Phase %0d sample %0d latency mismatch: expected %0d, got %0d",
                       phase_id, sample_id, EXPECTED_LATENCY, latency);
                error_count++;
            end

            for (int oc = 0; oc < OUT_CHANNELS; oc++) begin
                actual = $signed(outputVals[oc]);
                if (actual != expected[oc]) begin
                    $error("Phase %0d sample %0d output %0d mismatch: expected %0d, got %0d",
                           phase_id, sample_id, oc, expected[oc], actual);
                    error_count++;
                end
            end

            completed_samples++;

            @(posedge clk);
            #1;
            if (valid_out !== 1'b0) begin
                $error("Phase %0d sample %0d valid_out lasted more than one cycle",
                       phase_id, sample_id);
                error_count++;
            end

            // Vary the idle gap while preserving the sequence history.
            repeat ((phase_id + sample_id) % 4) @(posedge clk);
        end
    endtask

    task automatic randomize_weights(input bit stress);
        begin
            for (int oc = 0; oc < OUT_CHANNELS; oc++) begin
                if (stress) begin
                    bias1[oc] = random_between(-131072, 131071);
                    bias2[oc] = random_between(-131072, 131071);
                    residual_bias[oc] = random_between(-131072, 131071);
                end else begin
                    bias1[oc] = random_between(-8192, 8192);
                    bias2[oc] = random_between(-8192, 8192);
                    residual_bias[oc] = random_between(-8192, 8192);
                end

                for (int ic = 0; ic < IN_CHANNELS; ic++) begin
                    if (stress)
                        residual_weights[oc][ic] = random_between(-128, 127);
                    else
                        residual_weights[oc][ic] = random_between(-16, 16);

                    for (int tap = 0; tap < KERNEL_LEN; tap++) begin
                        if (stress)
                            weights1[oc][ic][tap] = random_between(-128, 127);
                        else
                            weights1[oc][ic][tap] = random_between(-16, 16);
                    end
                end

                for (int ic = 0; ic < OUT_CHANNELS; ic++) begin
                    for (int tap = 0; tap < KERNEL_LEN; tap++) begin
                        if (stress)
                            weights2[oc][ic][tap] = random_between(-128, 127);
                        else
                            weights2[oc][ic][tap] = random_between(-16, 16);
                    end
                end
            end
        end
    endtask

    task automatic randomize_sample(
        input bit stress,
        input int sample_id
    );
        begin
            for (int ic = 0; ic < IN_CHANNELS; ic++) begin
                if (stress)
                    stimulus[ic] = random_between(-128, 127);
                else
                    stimulus[ic] = random_between(-110, 110);
            end

            // Repeated corner patterns guarantee signed extremes are exercised
            // in every independently randomized phase.
            if (sample_id == 0) begin
                for (int ic = 0; ic < IN_CHANNELS; ic++)
                    stimulus[ic] = 0;
            end else if (sample_id == 1) begin
                stimulus[0] = 127;
                stimulus[1] = -128;
                stimulus[2] = 127;
                stimulus[3] = -128;
                stimulus[4] = 1;
                stimulus[5] = -1;
            end
        end
    endtask

    task automatic run_zero_directed_phase;
        begin
            clear_external_signals();
            apply_reset();

            for (int sample_id = 0; sample_id < 8; sample_id++) begin
                for (int ic = 0; ic < IN_CHANNELS; ic++)
                    stimulus[ic] = 0;
                drive_sample(-1, sample_id);
            end
        end
    endtask

    task automatic run_random_phase(
        input int phase_id,
        input int num_samples,
        input bit stress
    );
        begin
            randomize_weights(stress);
            apply_reset();

            $display("  phase=%0d samples=%0d stress=%0d", phase_id, num_samples, stress);
            for (int sample_id = 0; sample_id < num_samples; sample_id++) begin
                randomize_sample(stress, sample_id);
                drive_sample(phase_id, sample_id);
            end
        end
    endtask

    task automatic test_reset_during_operation;
        begin
            randomize_weights(1'b0);
            apply_reset();
            randomize_sample(1'b0, 7);

            @(negedge clk);
            for (int ic = 0; ic < IN_CHANNELS; ic++)
                inputVals[ic] = stimulus[ic];
            valid_in = 1'b1;
            @(posedge clk);
            @(negedge clk);
            valid_in = 1'b0;

            repeat (10) @(posedge clk);
            @(negedge clk);
            rst_n = 1'b0;
            repeat (3) @(posedge clk);
            @(negedge clk);
            rst_n = 1'b1;
            clear_reference_history();
            repeat (3) @(posedge clk);
            #1;

            if (valid_out !== 1'b0) begin
                $error("valid_out asserted after reset aborted an active transaction");
                error_count++;
            end
        end
    endtask

    initial begin
        int seed_override;
        int effective_seed;
        int phase_samples;
        bit stress;

        error_count = 0;
        completed_samples = 0;
        cycle_count = 0;
        rst_n = 1'b0;
        valid_in = 1'b0;

        effective_seed = RANDOM_SEED;
        if ($value$plusargs("SEED=%d", seed_override))
            effective_seed = seed_override;
        rng_state = effective_seed;
        if (rng_state == 0)
            rng_state = 32'h1;

        clear_external_signals();

        $display("input_block standalone randomized test: seed=%0d phases=%0d",
                 effective_seed, NUM_RANDOM_PHASES);

        run_zero_directed_phase();
        test_reset_during_operation();

        for (int phase_id = 0; phase_id < NUM_RANDOM_PHASES; phase_id++) begin
            stress = (phase_id >= (NUM_RANDOM_PHASES - 2));
            if (phase_id == (NUM_RANDOM_PHASES - 1))
                phase_samples = WRAP_PHASE_SAMPLES;
            else
                phase_samples = SAMPLES_PER_PHASE;

            run_random_phase(phase_id, phase_samples, stress);
        end

        if (error_count == 0) begin
            $display("PASS: input_block passed %0d standalone sequential samples",
                     completed_samples);
        end else begin
            $fatal(1, "FAIL: input_block found %0d errors across %0d completed samples",
                   error_count, completed_samples);
        end

        $finish;
    end

endmodule

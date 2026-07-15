`timescale 1ns/1ps

module output_block_tb;

    localparam int IN_CHANNELS  = 16;
    localparam int OUT_CHANNELS = 6;
    localparam int W_BIT_WIDTH  = 8;
    localparam int B_BIT_WIDTH  = 32;
    localparam int QUANT_SHIFT  = 8;
    localparam int CLK_PERIOD   = 10;
    localparam int NUM_GENERAL_TESTS = 6;
    localparam int EXPECTED_LATENCY = IN_CHANNELS + 1;

    localparam longint signed QMAX = (1 <<< (W_BIT_WIDTH - 1)) - 1;
    localparam longint signed QMIN = -(1 <<< (W_BIT_WIDTH - 1));

    logic clk;
    logic rst_n;
    logic valid_in;

    logic signed [W_BIT_WIDTH-1:0]
        inputVals [0:IN_CHANNELS-1];

    logic signed [W_BIT_WIDTH-1:0]
        weights [0:OUT_CHANNELS-1][0:IN_CHANNELS-1];

    logic signed [B_BIT_WIDTH-1:0]
        bias [0:OUT_CHANNELS-1];

    logic signed [W_BIT_WIDTH-1:0]
        outputVals [0:OUT_CHANNELS-1];

    logic valid_out;

    logic signed [W_BIT_WIDTH-1:0]
        stimulus [0:NUM_GENERAL_TESTS-1][0:IN_CHANNELS-1];

    longint signed expected [0:OUT_CHANNELS-1];

    integer error_count;
    integer test_count;
    longint unsigned cycle_count;

    output_block #(
        .IN_CHANNELS(IN_CHANNELS),
        .OUT_CHANNELS(OUT_CHANNELS),
        .W_BIT_WIDTH(W_BIT_WIDTH),
        .B_BIT_WIDTH(B_BIT_WIDTH),
        .QUANT_SHIFT(QUANT_SHIFT)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .inputVals(inputVals),
        .weights(weights),
        .bias(bias),
        .outputVals(outputVals),
        .valid_out(valid_out)
    );

    initial clk = 1'b0;
    always #(CLK_PERIOD/2) clk = ~clk;

    always @(posedge clk)
        cycle_count = cycle_count + 1;

`ifdef DUMP_WAVES
    initial begin
        $dumpfile("tb_output_block.vcd");
        $dumpvars(0, tb_output_block);
    end
`endif

    function automatic longint signed quantize_reference(
        input longint signed value
    );
        longint signed shifted;
        begin
            shifted = value >>> QUANT_SHIFT;

            if (shifted > QMAX)
                quantize_reference = QMAX;
            else if (shifted < QMIN)
                quantize_reference = QMIN;
            else
                quantize_reference = shifted;
        end
    endfunction

    task automatic calculate_expected;
        longint signed acc;
        longint signed input_value;
        longint signed weight_value;
        begin
            for (int oc = 0; oc < OUT_CHANNELS; oc++) begin
                acc = longint'($signed(bias[oc]));

                for (int ic = 0; ic < IN_CHANNELS; ic++) begin
                    input_value = longint'($signed(inputVals[ic]));
                    weight_value = longint'($signed(weights[oc][ic]));
                    acc += input_value * weight_value;
                end

                expected[oc] = quantize_reference(acc);
            end
        end
    endtask

    task automatic apply_reset;
        begin
            valid_in = 1'b0;
            rst_n = 1'b0;

            for (int ic = 0; ic < IN_CHANNELS; ic++)
                inputVals[ic] = '0;

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

    task automatic run_vector(input int vector_id);
        longint unsigned accept_cycle;
        longint unsigned result_cycle;
        longint signed actual;
        longint signed latency;
        integer timeout_cycles;
        begin
            @(negedge clk);
            for (int ic = 0; ic < IN_CHANNELS; ic++)
                inputVals[ic] = stimulus[vector_id][ic];

            calculate_expected();
            valid_in = 1'b1;

            @(posedge clk);
            #1;
            accept_cycle = cycle_count;

            @(negedge clk);
            valid_in = 1'b0;

            timeout_cycles = 0;
            while (valid_out !== 1'b1 && timeout_cycles < 100) begin
                @(posedge clk);
                #1;
                timeout_cycles++;
            end

            if (valid_out !== 1'b1) begin
                $error("Vector %0d timed out waiting for valid_out", vector_id);
                error_count++;
                return;
            end

            result_cycle = cycle_count;
            latency = result_cycle - accept_cycle;

            if (latency != EXPECTED_LATENCY) begin
                $error("Vector %0d latency mismatch: expected %0d cycles, got %0d",
                       vector_id, EXPECTED_LATENCY, latency);
                error_count++;
            end

            for (int oc = 0; oc < OUT_CHANNELS; oc++) begin
                actual = longint'($signed(outputVals[oc]));

                if (actual != expected[oc]) begin
                    $error("Vector %0d output %0d mismatch: expected %0d, got %0d",
                           vector_id, oc, expected[oc], actual);
                    error_count++;
                end
            end

            test_count++;
            $display("PASS output_block vector %0d at cycle %0d", vector_id, result_cycle);

            // valid_out must be a one-cycle pulse.
            @(posedge clk);
            #1;
            if (valid_out !== 1'b0) begin
                $error("valid_out remained asserted for more than one cycle");
                error_count++;
            end
        end
    endtask

    task automatic initialize_general_case;
        begin
            for (int oc = 0; oc < OUT_CHANNELS; oc++) begin
                bias[oc] = B_BIT_WIDTH'((oc - 3) * 173);

                for (int ic = 0; ic < IN_CHANNELS; ic++)
                    weights[oc][ic] = W_BIT_WIDTH'(((oc * 11 + ic * 7 + 3) % 31) - 15);
            end

            for (int t = 0; t < NUM_GENERAL_TESTS; t++) begin
                for (int ic = 0; ic < IN_CHANNELS; ic++)
                    stimulus[t][ic] = W_BIT_WIDTH'(((t * 37 + ic * 43 + 9) % 201) - 100);
            end
        end
    endtask

    task automatic initialize_saturation_case;
        begin
            for (int oc = 0; oc < OUT_CHANNELS; oc++) begin
                bias[oc] = '0;

                for (int ic = 0; ic < IN_CHANNELS; ic++)
                    weights[oc][ic] = '0;
            end

            for (int ic = 0; ic < IN_CHANNELS; ic++)
                stimulus[0][ic] = 127;

            // Lane 0: positive multiply accumulation saturates to +127.
            for (int ic = 0; ic < IN_CHANNELS; ic++)
                weights[0][ic] = 127;

            // Lane 1: negative multiply accumulation saturates to -128.
            for (int ic = 0; ic < IN_CHANNELS; ic++)
                weights[1][ic] = -128;

            // Lanes 2 and 3 test bias-only positive and negative saturation.
            bias[2] = 32'sd51200;   // +200 after >>> 8
            bias[3] = -32'sd51200;  // -200 after >>> 8

            // Lanes 4 and 5 remain non-saturating controls.
            bias[4] = 32'sd1024;
            bias[5] = -32'sd1024;
        end
    endtask

    initial begin
        error_count = 0;
        test_count = 0;
        cycle_count = 0;
        rst_n = 1'b0;
        valid_in = 1'b0;

        initialize_general_case();
        apply_reset();

        for (int t = 0; t < NUM_GENERAL_TESTS; t++)
            run_vector(t);

        // Reset between phases so the saturation case starts from a known state.
        initialize_saturation_case();
        apply_reset();
        run_vector(0);

        if (error_count == 0)
            $display("PASS: tb_output_block completed %0d checks with no errors", test_count);
        else
            $fatal(1, "FAIL: tb_output_block found %0d errors", error_count);

        $finish;
    end

endmodule

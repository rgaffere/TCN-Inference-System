`timescale 1ns/1ps

module tcn_block_tb;

    // ---------------------------------------------------------------------
    // DUT configuration
    // ---------------------------------------------------------------------
    localparam int IMU_CHANNELS    = 6;
    localparam int HIDDEN_CHANNELS = 16;
    localparam int NUM_HIDDEN      = 7;
    localparam int W_WIDTH         = 8;
    localparam int B_WIDTH         = 32;
    localparam int BASE_DILATION   = 2;
    localparam int KERNEL_LEN      = 3;
    localparam int AXIS_WIDTH      = W_WIDTH * IMU_CHANNELS;

    // Change this to match the synthesized clock constraint.
    localparam real CLK_PERIOD_NS = 5.0;

    // Whole-network MAC count per accepted sample:
    // input conv1 + input conv2 + input residual projection
    // + two depthwise convolutions in every hidden block
    // + output projection.
    localparam int MACS_PER_SAMPLE =
        (HIDDEN_CHANNELS * IMU_CHANNELS * KERNEL_LEN) +
        (HIDDEN_CHANNELS * HIDDEN_CHANNELS * KERNEL_LEN) +
        (HIDDEN_CHANNELS * IMU_CHANNELS) +
        (NUM_HIDDEN * 2 * HIDDEN_CHANNELS * KERNEL_LEN) +
        (IMU_CHANNELS * HIDDEN_CHANNELS);

    localparam int DEFAULT_ZERO_SAMPLES   = 8;
    localparam int DEFAULT_STRESS_SAMPLES = 128;
    localparam int DEFAULT_WARMUP_SAMPLES = 512;
    localparam int DEFAULT_POWER_SAMPLES  = 256;
    localparam int DEFAULT_READY_PERCENT  = 70;
    localparam int MAX_SIM_CYCLES         = 2_000_000;

    // Keep randomized weights relatively small while still creating useful
    // switching activity throughout the arithmetic datapath.
    localparam int RANDOM_WEIGHT_LIMIT = 8;
    localparam int RANDOM_BIAS_LIMIT   = 256;

    // ---------------------------------------------------------------------
    // AXI4-Stream interface
    // ---------------------------------------------------------------------
    logic                  clk;
    logic                  rst_n;
    logic                  s_tvalid;
    logic [AXIS_WIDTH-1:0] s_tdata;
    logic                  s_tready;
    logic                  m_tvalid;
    logic [AXIS_WIDTH-1:0] m_tdata;
    logic                  m_tready;

    tcn_block #(
        .IMU_CHANNELS(IMU_CHANNELS),
        .HIDDEN_CHANNELS(HIDDEN_CHANNELS),
        .NUM_HIDDEN(NUM_HIDDEN),
        .W_WIDTH(W_WIDTH),
        .B_WIDTH(B_WIDTH),
        .BASE_DILATION(BASE_DILATION),
        .KERNEL_LEN(KERNEL_LEN)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .s_tvalid(s_tvalid),
        .s_tdata(s_tdata),
        .m_tready(m_tready),
        .s_tready(s_tready),
        .m_tdata(m_tdata),
        .m_tvalid(m_tvalid)
    );

    // ---------------------------------------------------------------------
    // Runtime controls
    // ---------------------------------------------------------------------
    int unsigned random_seed;
    int zero_samples;
    int stress_samples;
    int warmup_samples;
    int power_samples;
    int ready_percent;

    bit randomize_m_tready;
    bit expect_zero_output;
    bit phase_active;
    bit power_capture_active;

    string phase_name;

    // ---------------------------------------------------------------------
    // Metrics and scoreboard state
    // ---------------------------------------------------------------------
    longint unsigned cycle_count;
    longint unsigned global_transaction_id;

    bit              transaction_outstanding;
    bit              core_result_seen;
    longint unsigned current_transaction_id;
    longint unsigned current_accept_cycle;
    longint unsigned current_core_cycle;
    logic [AXIS_WIDTH-1:0] current_input_word;

    int unsigned phase_inputs;
    int unsigned phase_core_results;
    int unsigned phase_outputs;

    longint unsigned phase_first_input_cycle;
    longint unsigned phase_first_output_cycle;
    longint unsigned phase_last_output_cycle;

    longint unsigned phase_core_latency_sum;
    longint unsigned phase_axis_latency_sum;
    longint unsigned phase_core_latency_min;
    longint unsigned phase_core_latency_max;
    longint unsigned phase_axis_latency_min;
    longint unsigned phase_axis_latency_max;

    int error_count;
    integer metrics_fd;
    integer summary_fd;

    logic [AXIS_WIDTH-1:0] previous_m_tdata;
    logic                  previous_m_tvalid;
    logic                  previous_m_tready;

    // ---------------------------------------------------------------------
    // Clock generation
    // ---------------------------------------------------------------------
    initial clk = 1'b0;
    always #(CLK_PERIOD_NS / 2.0) clk = ~clk;

    // ---------------------------------------------------------------------
    // Switching-activity capture
    //
    // The VCD is deliberately disabled during reset, initialization, stress,
    // and receptive-field warm-up. It is enabled only for the sustained-load
    // power phase. Use this same testbench with the synthesized netlist for
    // the most accurate post-synthesis activity annotation.
    // ---------------------------------------------------------------------
    initial begin
        $dumpfile("tcn_block_activity.vcd");
        $dumpvars(0, dut);
        $dumpoff;

        forever begin
            @(posedge power_capture_active);
            $dumpon;
            @(negedge power_capture_active);
            $dumpoff;
            $dumpflush;
        end
    end

    // ---------------------------------------------------------------------
    // Random helpers
    // ---------------------------------------------------------------------
    function automatic logic signed [W_WIDTH-1:0] random_weight();
        int signed value;
        begin
            value = int'($urandom_range(2 * RANDOM_WEIGHT_LIMIT, 0))
                    - RANDOM_WEIGHT_LIMIT;
            random_weight = value[W_WIDTH-1:0];
        end
    endfunction

    function automatic logic signed [B_WIDTH-1:0] random_bias();
        int signed value;
        begin
            value = int'($urandom_range(2 * RANDOM_BIAS_LIMIT, 0))
                    - RANDOM_BIAS_LIMIT;
            random_bias = value[B_WIDTH-1:0];
        end
    endfunction

    function automatic logic [AXIS_WIDTH-1:0] random_sample();
        logic [AXIS_WIDTH-1:0] value;
        begin
            value = '0;
            for (int ch = 0; ch < IMU_CHANNELS; ch++) begin
                value[ch * W_WIDTH +: W_WIDTH] = $urandom();
            end
            random_sample = value;
        end
    endfunction

    // ---------------------------------------------------------------------
    // Weight loading through hierarchical module variables
    //
    // Replace the assignments in load_random_model() with trained model
    // values when running the final network. No DUT weight-loading interface
    // is required for simulation.
    // ---------------------------------------------------------------------
    task automatic clear_model_weights();
        begin
            for (int out_ch = 0; out_ch < HIDDEN_CHANNELS; out_ch++) begin
                for (int in_ch = 0; in_ch < IMU_CHANNELS; in_ch++) begin
                    for (int tap = 0; tap < KERNEL_LEN; tap++) begin
                        dut.i_weights1[out_ch][in_ch][tap] = '0;
                    end
                    dut.i_residual_weights[out_ch][in_ch] = '0;
                end

                for (int in_ch = 0; in_ch < HIDDEN_CHANNELS; in_ch++) begin
                    for (int tap = 0; tap < KERNEL_LEN; tap++) begin
                        dut.i_weights2[out_ch][in_ch][tap] = '0;
                    end
                end

                dut.i_bias1[out_ch]        = '0;
                dut.i_bias2[out_ch]        = '0;
                dut.i_residual_bias[out_ch] = '0;
            end

            for (int block = 0; block < NUM_HIDDEN; block++) begin
                for (int ch = 0; ch < HIDDEN_CHANNELS; ch++) begin
                    for (int tap = 0; tap < KERNEL_LEN; tap++) begin
                        dut.h_weights1[block][ch][tap] = '0;
                        dut.h_weights2[block][ch][tap] = '0;
                    end
                    dut.h_bias1[block][ch] = '0;
                    dut.h_bias2[block][ch] = '0;
                end
            end

            for (int out_ch = 0; out_ch < IMU_CHANNELS; out_ch++) begin
                for (int in_ch = 0; in_ch < HIDDEN_CHANNELS; in_ch++) begin
                    dut.o_weights[out_ch][in_ch] = '0;
                end
                dut.o_bias[out_ch] = '0;
            end
        end
    endtask

    task automatic load_random_model();
        begin
            for (int out_ch = 0; out_ch < HIDDEN_CHANNELS; out_ch++) begin
                for (int in_ch = 0; in_ch < IMU_CHANNELS; in_ch++) begin
                    for (int tap = 0; tap < KERNEL_LEN; tap++) begin
                        dut.i_weights1[out_ch][in_ch][tap] = random_weight();
                    end
                    dut.i_residual_weights[out_ch][in_ch] = random_weight();
                end

                for (int in_ch = 0; in_ch < HIDDEN_CHANNELS; in_ch++) begin
                    for (int tap = 0; tap < KERNEL_LEN; tap++) begin
                        dut.i_weights2[out_ch][in_ch][tap] = random_weight();
                    end
                end

                dut.i_bias1[out_ch]         = random_bias();
                dut.i_bias2[out_ch]         = random_bias();
                dut.i_residual_bias[out_ch] = random_bias();
            end

            for (int block = 0; block < NUM_HIDDEN; block++) begin
                for (int ch = 0; ch < HIDDEN_CHANNELS; ch++) begin
                    for (int tap = 0; tap < KERNEL_LEN; tap++) begin
                        dut.h_weights1[block][ch][tap] = random_weight();
                        dut.h_weights2[block][ch][tap] = random_weight();
                    end
                    dut.h_bias1[block][ch] = random_bias();
                    dut.h_bias2[block][ch] = random_bias();
                end
            end

            for (int out_ch = 0; out_ch < IMU_CHANNELS; out_ch++) begin
                for (int in_ch = 0; in_ch < HIDDEN_CHANNELS; in_ch++) begin
                    dut.o_weights[out_ch][in_ch] = random_weight();
                end
                dut.o_bias[out_ch] = random_bias();
            end
        end
    endtask

    // ---------------------------------------------------------------------
    // Reset and AXI stimulus
    // ---------------------------------------------------------------------
    task automatic reset_dut();
        begin
            phase_active              = 1'b0;
            power_capture_active      = 1'b0;
            randomize_m_tready        = 1'b0;
            expect_zero_output        = 1'b0;
            s_tvalid                  = 1'b0;
            s_tdata                   = '0;
            m_tready                  = 1'b0;

            @(negedge clk);
            rst_n = 1'b0;
            repeat (6) @(posedge clk);
            @(negedge clk);
            rst_n = 1'b1;
            repeat (2) @(posedge clk);
        end
    endtask

    task automatic send_sample(
        input logic [AXIS_WIDTH-1:0] sample,
        input int unsigned maximum_gap
    );
        int unsigned gap_cycles;
        begin
            // Wait until the DUT can accept another transaction. In the
            // current simple wrapper, this occurs only after the previous
            // output has transferred.
            while (s_tready !== 1'b1)
                @(posedge clk);

            gap_cycles = (maximum_gap == 0)
                ? 0 : $urandom_range(maximum_gap, 0);
            repeat (gap_cycles) @(posedge clk);

            @(negedge clk);
            s_tdata  = sample;
            s_tvalid = 1'b1;

            // Hold TVALID and TDATA until the rising-edge handshake.
            do begin
                @(posedge clk);
            end while (!(s_tvalid && s_tready));

            @(negedge clk);
            s_tvalid = 1'b0;
            s_tdata  = '0;
        end
    endtask

    task automatic drive_random_samples(
        input int unsigned sample_count,
        input int unsigned maximum_gap
    );
        logic [AXIS_WIDTH-1:0] sample;
        begin
            for (int sample_id = 0; sample_id < sample_count; sample_id++) begin
                sample = random_sample();
                send_sample(sample, maximum_gap);
            end
        end
    endtask

    // Randomized output backpressure. Drive on the falling edge so TREADY is
    // stable before the next AXI sampling edge.
    always @(negedge clk) begin
        if (!rst_n) begin
            m_tready = 1'b0;
        end else if (randomize_m_tready) begin
            m_tready = ($urandom_range(99, 0) < ready_percent);
        end else begin
            m_tready = 1'b1;
        end
    end

    // ---------------------------------------------------------------------
    // Phase metrics
    // ---------------------------------------------------------------------
    task automatic begin_phase(
        input string name,
        input bit use_random_backpressure,
        input bit require_zero_output
    );
        begin
            phase_name              = name;
            randomize_m_tready      = use_random_backpressure;
            expect_zero_output      = require_zero_output;
            phase_inputs            = 0;
            phase_core_results      = 0;
            phase_outputs           = 0;
            phase_first_input_cycle = 0;
            phase_first_output_cycle = 0;
            phase_last_output_cycle = 0;
            phase_core_latency_sum  = 0;
            phase_axis_latency_sum  = 0;
            phase_core_latency_min  = 64'hffff_ffff_ffff_ffff;
            phase_core_latency_max  = 0;
            phase_axis_latency_min  = 64'hffff_ffff_ffff_ffff;
            phase_axis_latency_max  = 0;
            phase_active            = 1'b1;

            $display("\n[%0t] Starting phase '%s'", $time, phase_name);
        end
    endtask

    task automatic report_phase();
        longint unsigned elapsed_cycles;
        longint unsigned output_window_cycles;
        real average_core_latency;
        real average_axis_latency;
        real samples_per_cycle;
        real sample_rate_msps;
        real effective_gmac_per_second;
        begin
            phase_active = 1'b0;

            if ((phase_outputs == 0) || (phase_core_results == 0)) begin
                $display("Phase '%s' did not produce complete transactions", phase_name);
                error_count++;
            end else begin
                elapsed_cycles = phase_last_output_cycle - phase_first_input_cycle + 1;
                output_window_cycles = phase_last_output_cycle - phase_first_output_cycle;
                average_core_latency = real'(phase_core_latency_sum) /
                                       real'(phase_core_results);
            average_axis_latency = real'(phase_axis_latency_sum) /
                                   real'(phase_outputs);
                if ((phase_outputs > 1) && (output_window_cycles > 0))
                    samples_per_cycle = real'(phase_outputs - 1) /
                                        real'(output_window_cycles);
                else
                    samples_per_cycle = real'(phase_outputs) / real'(elapsed_cycles);
                sample_rate_msps = samples_per_cycle * (1000.0 / CLK_PERIOD_NS);
                effective_gmac_per_second =
                    sample_rate_msps * real'(MACS_PER_SAMPLE) / 1000.0;

                $display("Phase:                         %s", phase_name);
            $display("  Inputs accepted:              %0d", phase_inputs);
            $display("  Outputs transferred:          %0d", phase_outputs);
            $display("  Measurement cycles:           %0d", elapsed_cycles);
            $display("  Core latency avg/min/max:     %0.2f / %0d / %0d cycles",
                     average_core_latency,
                     phase_core_latency_min,
                     phase_core_latency_max);
            $display("  AXIS latency avg/min/max:     %0.2f / %0d / %0d cycles",
                     average_axis_latency,
                     phase_axis_latency_min,
                     phase_axis_latency_max);
            $display("  Sustained throughput:         %0.8f samples/cycle",
                     samples_per_cycle);
            $display("  Sustained sample rate:        %0.4f MSamples/s at %0.2f ns",
                     sample_rate_msps, CLK_PERIOD_NS);
            $display("  Effective whole-TCN rate:     %0.4f GMAC/s (%0d MAC/sample)",
                     effective_gmac_per_second, MACS_PER_SAMPLE);

            $fdisplay(summary_fd,
                "%s,%0d,%0d,%0d,%0.4f,%0d,%0d,%0.4f,%0d,%0d,%0.10f,%0.6f,%0.6f",
                phase_name,
                phase_inputs,
                phase_outputs,
                elapsed_cycles,
                average_core_latency,
                phase_core_latency_min,
                phase_core_latency_max,
                average_axis_latency,
                phase_axis_latency_min,
                phase_axis_latency_max,
                samples_per_cycle,
                sample_rate_msps,
                effective_gmac_per_second);
            end
        end
    endtask

    task automatic run_phase(
        input string name,
        input int unsigned sample_count,
        input int unsigned maximum_input_gap,
        input bit use_random_backpressure,
        input bit require_zero_output
    );
        begin
            begin_phase(name, use_random_backpressure, require_zero_output);

            fork
                drive_random_samples(sample_count, maximum_input_gap);
                begin
                    wait (phase_outputs == sample_count);
                end
            join

            report_phase();
        end
    endtask

    // ---------------------------------------------------------------------
    // AXI protocol checker, transaction pairing, and latency measurement
    // ---------------------------------------------------------------------
    always @(posedge clk) begin
        longint unsigned core_latency;
        longint unsigned axis_latency;

        cycle_count = cycle_count + 1;

        if (!rst_n) begin
            transaction_outstanding = 1'b0;
            core_result_seen         = 1'b0;
            previous_m_tdata         = '0;
            previous_m_tvalid        = 1'b0;
            previous_m_tready        = 1'b0;
        end else begin
            // AXI rule: once TVALID is asserted without TREADY, TVALID and
            // TDATA must remain stable until the transfer occurs.
            if (previous_m_tvalid && !previous_m_tready) begin
                if (m_tvalid !== 1'b1) begin
                    $error("[%0t] m_tvalid dropped during backpressure", $time);
                    error_count++;
                end
                if (m_tdata !== previous_m_tdata) begin
                    $error("[%0t] m_tdata changed during backpressure", $time);
                    error_count++;
                end
            end

            if (m_tvalid && $isunknown(m_tdata)) begin
                $error("[%0t] m_tdata contains X/Z while m_tvalid is asserted: %h",
                       $time, m_tdata);
                error_count++;
            end

            // Input handshake is evaluated using values present before this
            // rising edge, matching AXI synchronous-transfer semantics.
            if (s_tvalid && s_tready) begin
                if (transaction_outstanding) begin
                    $error("[%0t] Multiple transactions are in flight in the simple wrapper",
                           $time);
                    error_count++;
                end

                transaction_outstanding = 1'b1;
                core_result_seen         = 1'b0;
                current_transaction_id   = global_transaction_id;
                global_transaction_id    = global_transaction_id + 1;
                current_accept_cycle     = cycle_count;
                current_core_cycle       = 0;
                current_input_word       = s_tdata;

                if (phase_active) begin
                    phase_inputs++;
                    if (phase_inputs == 1)
                        phase_first_input_cycle = cycle_count;
                end
            end

            // Output AXI handshake.
            if (m_tvalid && m_tready) begin
                if (!transaction_outstanding) begin
                    $error("[%0t] Output transferred without a corresponding input", $time);
                    error_count++;
                end else begin
                    axis_latency = cycle_count - current_accept_cycle;

                    if (!core_result_seen) begin
                        $error("[%0t] AXI output transferred before o_valid_out was observed",
                               $time);
                        error_count++;
                    end

                    if (expect_zero_output && (m_tdata !== '0)) begin
                        $error("[%0t] Zero-weight model produced nonzero output: %h",
                               $time, m_tdata);
                        error_count++;
                    end

                    if (phase_active) begin
                        phase_outputs++;
                        if (phase_outputs == 1)
                            phase_first_output_cycle = cycle_count;
                        phase_last_output_cycle = cycle_count;
                        phase_axis_latency_sum += axis_latency;
                        if (axis_latency < phase_axis_latency_min)
                            phase_axis_latency_min = axis_latency;
                        if (axis_latency > phase_axis_latency_max)
                            phase_axis_latency_max = axis_latency;
                    end

                    $fdisplay(metrics_fd,
                        "%0d,%s,%0d,%0d,%0d,%0d,%h,%h",
                        current_transaction_id,
                        phase_name,
                        current_accept_cycle,
                        current_core_cycle,
                        cycle_count,
                        axis_latency,
                        current_input_word,
                        m_tdata);

                    transaction_outstanding = 1'b0;
                    core_result_seen         = 1'b0;
                end
            end

            previous_m_tdata  = m_tdata;
            previous_m_tvalid = m_tvalid;
            previous_m_tready = m_tready;
        end
    end

    // o_valid_out changes immediately after a clock edge because it is a
    // registered child-module output. Capturing its rising edge separately
    // avoids confusing result availability with the later AXI transfer edge.
    always @(posedge dut.o_valid_out) begin
        longint unsigned core_latency;

        if (rst_n) begin
            if (!transaction_outstanding) begin
                $error("[%0t] o_valid_out asserted without an outstanding input", $time);
                error_count++;
            end else if (core_result_seen) begin
                $error("[%0t] o_valid_out asserted more than once for one input", $time);
                error_count++;
            end else begin
                core_result_seen     = 1'b1;
                current_core_cycle   = cycle_count;
                core_latency         = cycle_count - current_accept_cycle;

                if (phase_active) begin
                    phase_core_results++;
                    phase_core_latency_sum += core_latency;
                    if (core_latency < phase_core_latency_min)
                        phase_core_latency_min = core_latency;
                    if (core_latency > phase_core_latency_max)
                        phase_core_latency_max = core_latency;
                end
            end
        end
    end

    // ---------------------------------------------------------------------
    // Main test sequence
    // ---------------------------------------------------------------------
    initial begin
        random_seed   = 32'h54c4_2026;
        zero_samples  = DEFAULT_ZERO_SAMPLES;
        stress_samples = DEFAULT_STRESS_SAMPLES;
        warmup_samples = DEFAULT_WARMUP_SAMPLES;
        power_samples  = DEFAULT_POWER_SAMPLES;
        ready_percent  = DEFAULT_READY_PERCENT;

        void'($value$plusargs("SEED=%d", random_seed));
        void'($value$plusargs("ZERO_SAMPLES=%d", zero_samples));
        void'($value$plusargs("STRESS_SAMPLES=%d", stress_samples));
        void'($value$plusargs("WARMUP_SAMPLES=%d", warmup_samples));
        void'($value$plusargs("POWER_SAMPLES=%d", power_samples));
        void'($value$plusargs("READY_PERCENT=%d", ready_percent));
        void'($urandom(random_seed));

        rst_n                    = 1'b0;
        s_tvalid                 = 1'b0;
        s_tdata                  = '0;
        m_tready                 = 1'b0;
        randomize_m_tready       = 1'b0;
        expect_zero_output       = 1'b0;
        phase_active             = 1'b0;
        power_capture_active     = 1'b0;
        cycle_count              = 0;
        global_transaction_id    = 0;
        transaction_outstanding  = 1'b0;
        core_result_seen         = 1'b0;
        error_count              = 0;
        phase_name               = "initialization";

        metrics_fd = $fopen("tcn_transaction_metrics.csv", "w");
        summary_fd = $fopen("tcn_performance_summary.csv", "w");

        if (metrics_fd == 0 || summary_fd == 0) begin
            $fatal(1, "Could not open one or more output metric files");
        end

        $fdisplay(metrics_fd,
            "transaction_id,phase,input_cycle,core_valid_cycle,output_cycle,axis_latency_cycles,input_hex,output_hex");
        $fdisplay(summary_fd,
            "phase,inputs,outputs,elapsed_cycles,avg_core_latency,min_core_latency,max_core_latency,avg_axis_latency,min_axis_latency,max_axis_latency,samples_per_cycle,sample_rate_msps,effective_gmac_per_s");

        $display("TCN randomized integration test");
        $display("  Seed:                 %0d (0x%08x)", random_seed, random_seed);
        $display("  Clock period:         %0.2f ns", CLK_PERIOD_NS);
        $display("  MACs per sample:      %0d", MACS_PER_SAMPLE);
        $display("  Stress ready rate:    %0d%%", ready_percent);

        // -------------------------------------------------------------
        // Phase 1: deterministic zero-model self-check with randomized
        // inputs. This is the only numerical oracle that does not require
        // an external golden model.
        // -------------------------------------------------------------
        clear_model_weights();
        reset_dut();
        run_phase("zero_model_check", zero_samples, 2, 1'b1, 1'b1);

        // -------------------------------------------------------------
        // Phase 2: randomized weights, randomized inputs, randomized input
        // gaps, and randomized output backpressure. This stresses control,
        // array connectivity, X propagation, and AXI stability.
        // Replace load_random_model() with trained hierarchical assignments
        // for final model testing.
        // -------------------------------------------------------------
        load_random_model();
        reset_dut();
        run_phase("randomized_protocol_stress",
                  stress_samples,
                  3,
                  1'b1,
                  1'b0);

        // -------------------------------------------------------------
        // Phase 3: reset, then fill the complete 509-sample receptive field
        // at maximum offered load before measuring switching activity.
        // -------------------------------------------------------------
        reset_dut();
        run_phase("power_warmup",
                  warmup_samples,
                  0,
                  1'b0,
                  1'b0);

        // -------------------------------------------------------------
        // Phase 4: full-load performance and power-activity capture.
        // TVALID is offered as soon as the simple wrapper becomes ready and
        // TREADY remains high. Only this window is written to the VCD.
        // -------------------------------------------------------------
        power_capture_active = 1'b1;
        run_phase("sustained_load_power",
                  power_samples,
                  0,
                  1'b0,
                  1'b0);
        power_capture_active = 1'b0;

        repeat (4) @(posedge clk);
        $fclose(metrics_fd);
        $fclose(summary_fd);

        if (transaction_outstanding) begin
            $error("Simulation ended with a transaction still outstanding");
            error_count++;
        end

        if (error_count == 0) begin
            $display("\n============================================================");
            $display("TCN TOP-LEVEL RANDOMIZED TEST PASSED");
            $display("Generated: tcn_transaction_metrics.csv");
            $display("Generated: tcn_performance_summary.csv");
            $display("Generated: tcn_block_activity.vcd");
            $display("============================================================\n");
            $finish;
        end else begin
            $fatal(1, "TCN randomized test failed with %0d errors", error_count);
        end
    end

    // Global deadlock/timeout protection.
    initial begin
        repeat (MAX_SIM_CYCLES) @(posedge clk);
        $fatal(1, "Simulation timeout after %0d cycles", MAX_SIM_CYCLES);
    end

endmodule

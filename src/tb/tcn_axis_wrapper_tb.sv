`timescale 1ns/1ps

module tcn_axis_wrapper_tb;

    parameter int NUM_CHANNELS = 16;
    parameter int W_WIDTH      = 8;
    parameter int B_WIDTH      = 32;
    parameter int KERNEL_LEN   = 3;
    parameter int DILATION     = 1;
    parameter int DEPTH        = 512;

    localparam int AXIS_WIDTH = NUM_CHANNELS * W_WIDTH;
    localparam int NUM_RANDOM_TRANSACTIONS = 2000;
    localparam int MAX_DRAIN_CYCLES = 100000;

    localparam int READY_LOW    = 0;
    localparam int READY_HIGH   = 1;
    localparam int READY_RANDOM = 2;

    typedef logic [AXIS_WIDTH - 1:0] axis_word_t;

    logic clk;
    logic rst_n;

    logic s_tvalid;
    axis_word_t s_tdata;
    logic s_tready;

    logic m_tvalid;
    axis_word_t m_tdata;
    logic m_tready;

    logic signed [W_WIDTH - 1:0] weights1 [0:NUM_CHANNELS - 1][0:KERNEL_LEN - 1];
    logic signed [W_WIDTH - 1:0] weights2 [0:NUM_CHANNELS - 1][0:KERNEL_LEN - 1];
    logic signed [B_WIDTH - 1:0] bias1 [0:NUM_CHANNELS - 1];
    logic signed [B_WIDTH - 1:0] bias2 [0:NUM_CHANNELS - 1];

    logic signed [W_WIDTH - 1:0] ref_inputVals [0:NUM_CHANNELS - 1];
    logic signed [W_WIDTH - 1:0] ref_outputVals [0:NUM_CHANNELS - 1];
    logic ref_valid_in;
    logic ref_valid_out;

    axis_word_t expected_q[$];

    int ready_mode;
    int unsigned seed;
    int error_count;

    longint unsigned epoch_inputs;
    longint unsigned epoch_outputs;
    longint unsigned epoch_ref_outputs;

    longint unsigned total_inputs;
    longint unsigned total_outputs;
    longint unsigned total_ref_outputs;
    longint unsigned input_stall_cycles;
    longint unsigned output_stall_cycles;

    tcn_axis_wrapper #(
        .NUM_CHANNELS(NUM_CHANNELS),
        .W_WIDTH(W_WIDTH),
        .B_WIDTH(B_WIDTH),
        .KERNEL_LEN(KERNEL_LEN),
        .DILATION(DILATION),
        .DEPTH(DEPTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .s_tvalid(s_tvalid),
        .s_tdata(s_tdata),
        .m_tready(m_tready),
        .weights1(weights1),
        .weights2(weights2),
        .bias1(bias1),
        .bias2(bias2),
        .s_tready(s_tready),
        .m_tdata(m_tdata),
        .m_tvalid(m_tvalid)
    );

    // Reference residual block. It receives exactly the transactions accepted
    // by the AXI4-Stream slave side of the wrapper.
    resi_block #(
        .NUM_CHANNELS(NUM_CHANNELS),
        .W_BIT_WIDTH(W_WIDTH),
        .B_BIT_WIDTH(B_WIDTH),
        .KERNEL_LEN(KERNEL_LEN),
        .DILATION(DILATION),
        .DEPTH(DEPTH)
    ) ref_rb (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(ref_valid_in),
        .inputVals(ref_inputVals),
        .weights1(weights1),
        .weights2(weights2),
        .bias1(bias1),
        .bias2(bias2),
        .outputVals(ref_outputVals),
        .valid_out(ref_valid_out)
    );

    always #2.5 clk = ~clk;

    function automatic axis_word_t pack_ref_output();
        axis_word_t packed_word;
        packed_word = '0;

        for(int ch = 0; ch < NUM_CHANNELS; ch++) begin
            packed_word[ch * W_WIDTH +: W_WIDTH] = $unsigned(ref_outputVals[ch]);
        end

        return packed_word;
    endfunction

    function automatic axis_word_t random_axis_word();
        axis_word_t word;
        int signed lane_value;

        word = '0;
        for(int ch = 0; ch < NUM_CHANNELS; ch++) begin
            lane_value = $urandom_range(0, 255) - 128;
            word[ch * W_WIDTH +: W_WIDTH] = lane_value[W_WIDTH - 1:0];
        end

        return word;
    endfunction

    function automatic axis_word_t directed_axis_word(input int base_value);
        axis_word_t word;
        int signed lane_value;

        word = '0;
        for(int ch = 0; ch < NUM_CHANNELS; ch++) begin
            lane_value = base_value + ch;
            word[ch * W_WIDTH +: W_WIDTH] = lane_value[W_WIDTH - 1:0];
        end

        return word;
    endfunction

    task automatic report_error(input string message);
        error_count++;
        $error("[%0t] %s", $time, message);
    endtask

    task automatic compare_words(
        input axis_word_t got,
        input axis_word_t expected
    );
        int signed got_lane;
        int signed expected_lane;

        if(got !== expected) begin
            report_error($sformatf(
                "Output mismatch: got=%0h expected=%0h",
                got,
                expected
            ));

            for(int ch = 0; ch < NUM_CHANNELS; ch++) begin
                if(got[ch * W_WIDTH +: W_WIDTH] !==
                   expected[ch * W_WIDTH +: W_WIDTH]) begin
                    got_lane = $signed(got[ch * W_WIDTH +: W_WIDTH]);
                    expected_lane = $signed(expected[ch * W_WIDTH +: W_WIDTH]);
                    $display(
                        "    channel %0d: got=%0d expected=%0d",
                        ch,
                        got_lane,
                        expected_lane
                    );
                end
            end
        end
    endtask

    task automatic send_word(
        input axis_word_t word,
        input bit keep_valid_after_handshake
    );
        bit accepted;

        @(negedge clk);
        s_tdata  <= word;
        s_tvalid <= 1'b1;

        accepted = 1'b0;
        while(!accepted) begin
            @(posedge clk);
            accepted = s_tvalid && s_tready;
        end

        if(!keep_valid_after_handshake) begin
            @(negedge clk);
            s_tvalid <= 1'b0;
            s_tdata  <= '0;
        end
    endtask

    task automatic wait_for_m_tvalid(input int max_cycles);
        int cycles;

        cycles = 0;
        while(m_tvalid !== 1'b1 && cycles < max_cycles) begin
            @(negedge clk);
            cycles++;
        end

        if(m_tvalid !== 1'b1) begin
            report_error($sformatf(
                "Timed out waiting for m_tvalid after %0d cycles",
                max_cycles
            ));
        end
    endtask

    task automatic wait_for_drain();
        int cycles;
        bit drained;

        cycles = 0;
        drained = 1'b0;

        while(!drained && cycles < MAX_DRAIN_CYCLES) begin
            @(negedge clk);
            drained = (epoch_inputs == epoch_outputs) &&
                      (expected_q.size() == 0) &&
                      (m_tvalid == 1'b0);
            cycles++;
        end

        if(!drained) begin
            report_error($sformatf(
                "Drain timeout: inputs=%0d outputs=%0d ref_outputs=%0d queue=%0d m_tvalid=%0b",
                epoch_inputs,
                epoch_outputs,
                epoch_ref_outputs,
                expected_q.size(),
                m_tvalid
            ));
        end
    endtask

    task automatic apply_reset(input int num_cycles);
        @(negedge clk);
        rst_n     <= 1'b0;
        s_tvalid  <= 1'b0;
        s_tdata   <= '0;

        repeat(num_cycles) @(negedge clk);

        if(m_tvalid !== 1'b0) begin
            report_error("m_tvalid was not cleared by reset");
        end

        if(s_tready !== 1'b1) begin
            report_error("s_tready was not asserted by reset");
        end

        rst_n <= 1'b1;
        @(negedge clk);
    endtask

    // Feed the reference residual block on accepted slave-side AXI transfers.
    always_ff @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            ref_valid_in <= 1'b0;
            for(int ch = 0; ch < NUM_CHANNELS; ch++) begin
                ref_inputVals[ch] <= '0;
            end
        end else begin
            ref_valid_in <= 1'b0;

            if(s_tvalid && s_tready) begin
                for(int ch = 0; ch < NUM_CHANNELS; ch++) begin
                    ref_inputVals[ch] <= $signed(
                        s_tdata[ch * W_WIDTH +: W_WIDTH]
                    );
                end
                ref_valid_in <= 1'b1;
            end
        end
    end

    // Independently drive downstream backpressure.
    always @(negedge clk) begin
        if(!rst_n) begin
            m_tready <= 1'b0;
        end else begin
            case(ready_mode)
                READY_LOW: begin
                    m_tready <= 1'b0;
                end

                READY_HIGH: begin
                    m_tready <= 1'b1;
                end

                default: begin
                    m_tready <= ($urandom_range(0, 99) < 60);
                end
            endcase
        end
    end

    // Scoreboard: queue direct resi_block results and compare them to AXIS
    // output transfers from the wrapper.
    always @(posedge clk or negedge rst_n) begin
        axis_word_t expected_word;

        if(!rst_n) begin
            expected_q.delete();
            epoch_inputs      = 0;
            epoch_outputs     = 0;
            epoch_ref_outputs = 0;
        end else begin
            if(s_tvalid && s_tready) begin
                epoch_inputs++;
                total_inputs++;
            end

            if(ref_valid_out) begin
                expected_q.push_back(pack_ref_output());
                epoch_ref_outputs++;
                total_ref_outputs++;
            end

            if(s_tvalid && !s_tready) begin
                input_stall_cycles++;
            end

            if(m_tvalid && !m_tready) begin
                output_stall_cycles++;
            end

            if(m_tvalid && m_tready) begin
                epoch_outputs++;
                total_outputs++;

                if(expected_q.size() == 0) begin
                    report_error("Unexpected AXI output transfer with empty scoreboard queue");
                end else begin
                    expected_word = expected_q.pop_front();
                    compare_words(m_tdata, expected_word);
                end
            end

            if(epoch_outputs > epoch_inputs) begin
                report_error($sformatf(
                    "Output count exceeded input count: inputs=%0d outputs=%0d",
                    epoch_inputs,
                    epoch_outputs
                ));
            end
        end
    end

    property p_output_stable_while_stalled;
        @(posedge clk) disable iff(!rst_n)
        m_tvalid && !m_tready
        |=> m_tvalid && $stable(m_tdata);
    endproperty

    property p_master_holds_input_while_stalled;
        @(posedge clk) disable iff(!rst_n)
        s_tvalid && !s_tready
        |=> s_tvalid && $stable(s_tdata);
    endproperty

    property p_ready_drops_after_input_transfer;
        @(posedge clk) disable iff(!rst_n)
        s_tvalid && s_tready
        |=> !s_tready;
    endproperty

    property p_no_new_input_while_output_pending;
        @(posedge clk) disable iff(!rst_n)
        m_tvalid
        |-> !s_tready;
    endproperty

    assert property(p_output_stable_while_stalled)
        else report_error("m_tvalid/m_tdata changed while output was stalled");

    assert property(p_master_holds_input_while_stalled)
        else report_error("Testbench master changed s_tvalid/s_tdata while input was stalled");

    assert property(p_ready_drops_after_input_transfer)
        else report_error("s_tready did not deassert after an input transfer");

    assert property(p_no_new_input_while_output_pending)
        else report_error("s_tready asserted while an AXI output was still pending");

    initial begin
        axis_word_t word;
        axis_word_t stalled_output;
        bit keep_valid;
        int gap_cycles;

        clk                  = 1'b0;
        rst_n                = 1'b0;
        s_tvalid             = 1'b0;
        s_tdata              = '0;
        m_tready             = 1'b0;
        ready_mode           = READY_LOW;
        seed                 = 32'h51A5_2026;
        error_count          = 0;
        epoch_inputs         = 0;
        epoch_outputs        = 0;
        epoch_ref_outputs    = 0;
        total_inputs         = 0;
        total_outputs        = 0;
        total_ref_outputs    = 0;
        input_stall_cycles   = 0;
        output_stall_cycles  = 0;

        void'($urandom(seed));

        for(int ch = 0; ch < NUM_CHANNELS; ch++) begin
            bias1[ch] = $urandom_range(0, 1024) - 512;
            bias2[ch] = $urandom_range(0, 1024) - 512;

            for(int tap = 0; tap < KERNEL_LEN; tap++) begin
                weights1[ch][tap] = $urandom_range(0, 14) - 7;
                weights2[ch][tap] = $urandom_range(0, 14) - 7;
            end
        end

        $display("\n============================================================");
        $display("TCN AXI4-Stream wrapper randomized verification");
        $display("NUM_CHANNELS=%0d AXIS_WIDTH=%0d", NUM_CHANNELS, AXIS_WIDTH);
        $display("============================================================\n");

        apply_reset(5);

        $display("[TEST 1] Directed basic transfers");
        ready_mode = READY_HIGH;
        for(int tx = 0; tx < 8; tx++) begin
            send_word(directed_axis_word(-32 + tx), 1'b0);
        end
        wait_for_drain();

        $display("[TEST 2] Output backpressure and payload stability");
        ready_mode = READY_LOW;
        word = directed_axis_word(16);
        send_word(word, 1'b0);
        wait_for_m_tvalid(1000);
        stalled_output = m_tdata;

        repeat(25) begin
            @(negedge clk);
            if(m_tvalid !== 1'b1) begin
                report_error("m_tvalid dropped during forced output backpressure");
            end
            if(m_tdata !== stalled_output) begin
                report_error("m_tdata changed during forced output backpressure");
            end
        end

        ready_mode = READY_HIGH;
        wait_for_drain();

        $display("[TEST 3] Continuous TVALID with randomized TREADY");
        ready_mode = READY_RANDOM;
        for(int tx = 0; tx < 128; tx++) begin
            send_word(random_axis_word(), tx != 127);
        end
        wait_for_drain();

        $display("[TEST 4] Reset during active computation");
        ready_mode = READY_HIGH;
        send_word(random_axis_word(), 1'b0);
        repeat(2) @(posedge clk);
        apply_reset(4);

        if(expected_q.size() != 0) begin
            report_error("Scoreboard queue was not cleared by reset during computation");
        end

        $display("[TEST 5] Reset while output is stalled");
        ready_mode = READY_LOW;
        send_word(random_axis_word(), 1'b0);
        wait_for_m_tvalid(1000);
        repeat(5) @(posedge clk);
        apply_reset(4);

        if(m_tvalid !== 1'b0 || s_tready !== 1'b1) begin
            report_error("Wrapper did not return to idle state after stalled-output reset");
        end

        $display("[TEST 6] %0d randomized AXI transactions", NUM_RANDOM_TRANSACTIONS);
        ready_mode = READY_RANDOM;

        for(int tx = 0; tx < NUM_RANDOM_TRANSACTIONS; tx++) begin
            keep_valid = (tx != NUM_RANDOM_TRANSACTIONS - 1) &&
                         ($urandom_range(0, 99) < 55);

            send_word(random_axis_word(), keep_valid);

            if(!keep_valid) begin
                gap_cycles = $urandom_range(0, 5);
                repeat(gap_cycles) @(negedge clk);
            end
        end

        wait_for_drain();

        @(negedge clk);
        s_tvalid <= 1'b0;
        s_tdata  <= '0;
        ready_mode = READY_HIGH;
        repeat(5) @(posedge clk);

        if(epoch_inputs != epoch_outputs) begin
            report_error($sformatf(
                "Final transaction count mismatch: inputs=%0d outputs=%0d",
                epoch_inputs,
                epoch_outputs
            ));
        end

        if(epoch_ref_outputs != epoch_outputs) begin
            report_error($sformatf(
                "Final reference/output count mismatch: ref=%0d outputs=%0d",
                epoch_ref_outputs,
                epoch_outputs
            ));
        end

        if(expected_q.size() != 0) begin
            report_error($sformatf(
                "Expected-output queue not empty at end of test: size=%0d",
                expected_q.size()
            ));
        end

        $display("\n============================================================");
        $display("AXI4-Stream verification summary");
        $display("Total accepted inputs : %0d", total_inputs);
        $display("Total AXI outputs     : %0d", total_outputs);
        $display("Reference outputs     : %0d", total_ref_outputs);
        $display("Input stall cycles    : %0d", input_stall_cycles);
        $display("Output stall cycles   : %0d", output_stall_cycles);
        $display("Errors                : %0d", error_count);
        $display("============================================================");

        if(error_count == 0) begin
            $display("PASS: tcn_axis_wrapper passed randomized AXI4-Stream verification.\n");
        end else begin
            $fatal(1, "FAIL: tcn_axis_wrapper verification found %0d errors", error_count);
        end

        $finish;
    end

endmodule

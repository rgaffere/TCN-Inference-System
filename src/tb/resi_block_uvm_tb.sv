`timescale 1ns/1ps

package resi_uvm_pkg;
    import uvm_pkg::*;
    `include "uvm_macros.svh"

    localparam int NUM_CHANNELS = 16;
    localparam int W_BIT_WIDTH  = 8;
    localparam int B_BIT_WIDTH  = 32;
    localparam int KERNEL_LEN   = 3;
    localparam int DILATION     = 1;
    localparam int NUM_RINGS    = 4;
    localparam int DEPTH        = 64;
    localparam int SHIFT        = 8;
    localparam int CLK_PERIOD   = 10;

    localparam int PRIME_TOKENS = (2 * (KERNEL_LEN - 1) * DILATION) + KERNEL_LEN;
    localparam int FLUSH_TOKENS = KERNEL_LEN - 1;
    localparam int NUM_CHECKED_TXNS = 64;
    localparam int TOTAL_TXNS = PRIME_TOKENS + NUM_CHECKED_TXNS + FLUSH_TOKENS;

    typedef logic signed [W_BIT_WIDTH-1:0] chvec_t [0:NUM_CHANNELS-1];
    typedef logic signed [W_BIT_WIDTH-1:0] wmat_t  [0:NUM_CHANNELS-1][0:KERNEL_LEN-1];
    typedef logic signed [B_BIT_WIDTH-1:0] bvec_t  [0:NUM_CHANNELS-1];

    `uvm_analysis_imp_decl(_exp)
    `uvm_analysis_imp_decl(_act)

    class resi_txn extends uvm_sequence_item;
        rand chvec_t inputVals;
        chvec_t outputVals;
        bit check_en;
        int token_id;

        constraint small_values_c {
            foreach (inputVals[i]) inputVals[i] inside {[-64:64]};
        }

        `uvm_object_utils(resi_txn)

        function new(string name = "resi_txn");
            super.new(name);
            check_en = 1'b0;
            token_id = -1;
        endfunction

        function void make_zero();
            foreach (inputVals[ch]) inputVals[ch] = '0;
        endfunction
    endclass

    class resi_smoke_seq extends uvm_sequence #(resi_txn);
        `uvm_object_utils(resi_smoke_seq)

        function new(string name = "resi_smoke_seq");
            super.new(name);
        endfunction

        task body();
            resi_txn tx;
            int id;

            id = 0;

            repeat (PRIME_TOKENS) begin
                tx = resi_txn::type_id::create($sformatf("prime_%0d", id));
                start_item(tx);
                tx.make_zero();
                tx.check_en = 1'b0;
                tx.token_id = id;
                finish_item(tx);
                id++;
            end

            repeat (NUM_CHECKED_TXNS) begin
                tx = resi_txn::type_id::create($sformatf("real_%0d", id));
                start_item(tx);
                if (!tx.randomize()) begin
                    `uvm_fatal("RAND", "resi_txn randomization failed")
                end
                tx.check_en = 1'b1;
                tx.token_id = id;
                finish_item(tx);
                id++;
            end

            repeat (FLUSH_TOKENS) begin
                tx = resi_txn::type_id::create($sformatf("flush_%0d", id));
                start_item(tx);
                tx.make_zero();
                tx.check_en = 1'b0;
                tx.token_id = id;
                finish_item(tx);
                id++;
            end
        endtask
    endclass

    class resi_sequencer extends uvm_sequencer #(resi_txn);
        `uvm_component_utils(resi_sequencer)

        function new(string name, uvm_component parent);
            super.new(name, parent);
        endfunction
    endclass
endpackage

interface resi_if(input logic clk);
    import resi_uvm_pkg::*;

    logic rst_n;
    logic valid_in;
    logic valid_out;
    logic accept;

    chvec_t inputVals;
    wmat_t  weights1;
    wmat_t  weights2;
    bvec_t  bias1;
    bvec_t  bias2;
    chvec_t outputVals;
endinterface

package resi_uvm_components_pkg;
    import uvm_pkg::*;
    `include "uvm_macros.svh"
    import resi_uvm_pkg::*;

    class resi_driver extends uvm_driver #(resi_txn);
        `uvm_component_utils(resi_driver)

        virtual resi_if vif;
        uvm_analysis_port #(resi_txn) exp_ap;

        function new(string name, uvm_component parent);
            super.new(name, parent);
            exp_ap = new("exp_ap", this);
        endfunction

        function void build_phase(uvm_phase phase);
            super.build_phase(phase);

            if (!uvm_config_db#(virtual resi_if)::get(this, "", "vif", vif)) begin
                `uvm_fatal("NOVIF", "virtual interface not found")
            end
        endfunction

        task clear_dut_inputs();
            vif.valid_in <= 1'b0;

            foreach (vif.inputVals[ch]) begin
                vif.inputVals[ch] <= '0;
                vif.bias1[ch]     <= '0;
                vif.bias2[ch]     <= '0;
            end

            foreach (vif.weights1[ch]) begin
                foreach (vif.weights1[ch, k]) begin
                    vif.weights1[ch][k] <= '0;
                    vif.weights2[ch][k] <= '0;
                end
            end
        endtask

        task reset_dut();
            clear_dut_inputs();
            vif.rst_n <= 1'b0;

            repeat (5) @(posedge vif.clk);

            @(negedge vif.clk);
            vif.rst_n <= 1'b1;

            @(posedge vif.clk);
            #2;
        endtask

        task run_phase(uvm_phase phase);
            resi_txn tx;
            resi_txn exp;

            reset_dut();

            forever begin
                seq_item_port.get_next_item(tx);

                @(negedge vif.clk);
                foreach (vif.inputVals[ch]) begin
                    vif.inputVals[ch] <= tx.inputVals[ch];
                end
                vif.valid_in <= 1'b1;

                do @(posedge vif.clk); while (!vif.accept);

                exp = resi_txn::type_id::create("exp");
                exp.check_en = tx.check_en;
                exp.token_id = tx.token_id;
                foreach (exp.inputVals[ch]) begin
                    exp.inputVals[ch] = tx.inputVals[ch];
                end
                exp_ap.write(exp);

                @(negedge vif.clk);
                vif.valid_in <= 1'b0;

                seq_item_port.item_done();
            end
        endtask
    endclass

    class resi_monitor extends uvm_monitor;
        `uvm_component_utils(resi_monitor)

        virtual resi_if vif;
        uvm_analysis_port #(resi_txn) act_ap;

        function new(string name, uvm_component parent);
            super.new(name, parent);
            act_ap = new("act_ap", this);
        endfunction

        function void build_phase(uvm_phase phase);
            super.build_phase(phase);

            if (!uvm_config_db#(virtual resi_if)::get(this, "", "vif", vif)) begin
                `uvm_fatal("NOVIF", "virtual interface not found")
            end
        endfunction

        task run_phase(uvm_phase phase);
            resi_txn act;

            forever begin
                @(posedge vif.clk);
                #1;

                if (vif.rst_n && vif.valid_out) begin
                    act = resi_txn::type_id::create("act");

                    foreach (act.outputVals[ch]) begin
                        act.outputVals[ch] = vif.outputVals[ch];
                    end

                    act_ap.write(act);
                end
            end
        endtask
    endclass

    class resi_scoreboard extends uvm_scoreboard;
        `uvm_component_utils(resi_scoreboard)

        uvm_analysis_imp_exp #(resi_txn, resi_scoreboard) exp_imp;
        uvm_analysis_imp_act #(resi_txn, resi_scoreboard) act_imp;

        resi_txn exp_q[$];
        int checked_tokens;
        int unchecked_tokens;
        int checks;
        int mismatches;
        int err_prints;

        function new(string name, uvm_component parent);
            super.new(name, parent);
            exp_imp = new("exp_imp", this);
            act_imp = new("act_imp", this);
            checked_tokens = 0;
            unchecked_tokens = 0;
            checks = 0;
            mismatches = 0;
            err_prints = 0;
        endfunction

        function void write_exp(resi_txn tx);
            exp_q.push_back(tx);
        endfunction

        function void write_act(resi_txn act);
            resi_txn exp;

            if (exp_q.size() == 0) begin
                mismatches++;
                `uvm_error("SCOREBOARD", "spurious valid_out with no accepted token pending")
                return;
            end

            exp = exp_q.pop_front();

            if (exp.check_en) begin
                checked_tokens++;

                foreach (act.outputVals[ch]) begin
                    checks++;

                    if (act.outputVals[ch] !== exp.inputVals[ch]) begin
                        mismatches++;
                        if (err_prints < 64) begin
                            err_prints++;
                            `uvm_error("MISMATCH", $sformatf(
                                "tok=%0d ch=%0d exp=%0d got=%0d",
                                exp.token_id,
                                ch,
                                exp.inputVals[ch],
                                act.outputVals[ch]
                            ))
                        end
                    end
                end
            end else begin
                unchecked_tokens++;
            end
        endfunction

        function void check_phase(uvm_phase phase);
            super.check_phase(phase);

            if (exp_q.size() != 0) begin
                mismatches++;
                `uvm_error("SCOREBOARD", $sformatf("%0d expected outputs never arrived", exp_q.size()))
            end

            if (checked_tokens != NUM_CHECKED_TXNS) begin
                mismatches++;
                `uvm_error("SCOREBOARD", $sformatf(
                    "checked token count mismatch: exp=%0d got=%0d",
                    NUM_CHECKED_TXNS,
                    checked_tokens
                ))
            end
        endfunction

        function void report_phase(uvm_phase phase);
            super.report_phase(phase);

            if (mismatches == 0) begin
                `uvm_info("RESULT", $sformatf(
                    "PASS: %0d checked tokens, %0d unchecked prime/flush tokens, %0d channel checks",
                    checked_tokens,
                    unchecked_tokens,
                    checks
                ), UVM_LOW)
            end else begin
                `uvm_error("RESULT", $sformatf(
                    "FAIL: %0d mismatches / %0d checks, checked_tokens=%0d unchecked_tokens=%0d pending=%0d",
                    mismatches,
                    checks,
                    checked_tokens,
                    unchecked_tokens,
                    exp_q.size()
                ))
            end
        endfunction
    endclass

    class resi_agent extends uvm_agent;
        `uvm_component_utils(resi_agent)

        resi_sequencer sqr;
        resi_driver    drv;
        resi_monitor   mon;

        function new(string name, uvm_component parent);
            super.new(name, parent);
        endfunction

        function void build_phase(uvm_phase phase);
            super.build_phase(phase);
            sqr = resi_sequencer::type_id::create("sqr", this);
            drv = resi_driver::type_id::create("drv", this);
            mon = resi_monitor::type_id::create("mon", this);
        endfunction

        function void connect_phase(uvm_phase phase);
            super.connect_phase(phase);
            drv.seq_item_port.connect(sqr.seq_item_export);
        endfunction
    endclass

    class resi_env extends uvm_env;
        `uvm_component_utils(resi_env)

        resi_agent      agent;
        resi_scoreboard sb;

        function new(string name, uvm_component parent);
            super.new(name, parent);
        endfunction

        function void build_phase(uvm_phase phase);
            super.build_phase(phase);
            agent = resi_agent::type_id::create("agent", this);
            sb    = resi_scoreboard::type_id::create("sb", this);
        endfunction

        function void connect_phase(uvm_phase phase);
            super.connect_phase(phase);
            agent.drv.exp_ap.connect(sb.exp_imp);
            agent.mon.act_ap.connect(sb.act_imp);
        endfunction
    endclass

    class resi_smoke_test extends uvm_test;
        `uvm_component_utils(resi_smoke_test)

        resi_env env;
        virtual resi_if vif;

        function new(string name, uvm_component parent);
            super.new(name, parent);
        endfunction

        function void build_phase(uvm_phase phase);
            super.build_phase(phase);
            env = resi_env::type_id::create("env", this);

            if (!uvm_config_db#(virtual resi_if)::get(this, "", "vif", vif)) begin
                `uvm_fatal("NOVIF", "virtual interface not found")
            end
        endfunction

        task run_phase(uvm_phase phase);
            resi_smoke_seq seq;

            phase.raise_objection(this);

            seq = resi_smoke_seq::type_id::create("seq");
            seq.start(env.agent.sqr);

            repeat (128) @(posedge vif.clk);

            phase.drop_objection(this);
        endtask
    endclass
endpackage

module resi_block_uvm_tb;
    import uvm_pkg::*;
    import resi_uvm_pkg::*;
    import resi_uvm_components_pkg::*;

    logic clk = 1'b0;

    always #(CLK_PERIOD/2) clk = ~clk;

    resi_if vif(clk);

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
        .rst_n(vif.rst_n),
        .valid_in(vif.valid_in),
        .inputVals(vif.inputVals),
        .weights1(vif.weights1),
        .weights2(vif.weights2),
        .bias1(vif.bias1),
        .bias2(vif.bias2),
        .outputVals(vif.outputVals),
        .valid_out(vif.valid_out)
    );

    assign vif.accept = dut.mac1_start;

    initial begin
        if ($test$plusargs("VCD")) begin
            $dumpfile("resi_block_uvm_tb.vcd");
            $dumpvars(0, resi_block_uvm_tb);
        end
    end

    initial begin
        uvm_config_db#(virtual resi_if)::set(null, "*", "vif", vif);
        run_test("resi_smoke_test");
    end
endmodule

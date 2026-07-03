`timescale 1ns/1ps

package resi_uvm_pkg;
    import uvm_pkg::*;
    `include "uvm_macros.svh"

    localparam int NUM_CHANNELS = 16;
    localparam int W_BIT_WIDTH = 8;
    localparam int B_BIT_WIDTH = 32;
    localparam int KERNEL_LEN = 3;
    localparam int DILATION = 1;
    localparam int NUM_RINGS = 4;
    localparam int DEPTH = 64;
    localparam int CLK_PERIOD = 10;

    typedef logic signed [W_BIT_WIDTH-1:0] chvec_t [0:NUM_CHANNELS-1];
    typedef logic signed [W_BIT_WIDTH-1:0] wmat_t [0:NUM_CHANNELS-1][0:KERNEL_LEN-1];
    typedef logic signed [B_BIT_WIDTH-1:0] bvec_t [0:NUM_CHANNELS-1];

    `uvm_analysis_imp_decl(_exp)
    `uvm_analysis_imp_decl(_act)

    class resi_txn extends uvm_sequence_item;
        rand chvec_t inputVals;
        chvec_t outputVals;

        constraint small_values_c {
            foreach (inputVals[i]) inputVals[i] inside {[-64:64]};
        }

        `uvm_object_utils(resi_txn)

        function new(string name = "resi_txn");
            super.new(name);
        endfunction
    endclass

    class resi_smoke_seq extends uvm_sequence #(resi_txn);
        `uvm_object_utils(resi_smoke_seq)

        function new(string name = "resi_smoke_seq");
            super.new(name);
        endfunction

        task body();
            resi_txn tx;

            repeat (64) begin
                tx = resi_txn::type_id::create("tx");
                start_item(tx);
                if (!tx.randomize()) begin
                    `uvm_fatal("RAND", "resi_txn randomization failed")
                end
                finish_item(tx);
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

        task reset_dut();
            vif.valid_in <= 1'b0;
            vif.rst_n <= 1'b0;

            foreach (vif.inputVals[ch]) begin
                vif.inputVals[ch] <= '0;
                vif.bias1[ch] <= '0;
                vif.bias2[ch] <= '0;

                foreach (vif.weights1[ch][k]) begin
                    vif.weights1[ch][k] <= '0;
                    vif.weights2[ch][k] <= '0;
                end
            end

            repeat (5) @(posedge vif.clk);
            vif.rst_n <= 1'b1;
            repeat (2) @(posedge vif.clk);
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
        int checks;
        int mismatches;

        function new(string name, uvm_component parent);
            super.new(name, parent);
            exp_imp = new("exp_imp", this);
            act_imp = new("act_imp", this);
        endfunction

        function void write_exp(resi_txn tx);
            exp_q.push_back(tx);
        endfunction

        function void write_act(resi_txn act);
            resi_txn exp;

            if (exp_q.size() == 0) begin
                mismatches++;
                `uvm_error("SCOREBOARD", "valid_out observed with no expected transaction")
                return;
            end

            exp = exp_q.pop_front();

            foreach (act.outputVals[ch]) begin
                checks++;

                if (act.outputVals[ch] !== exp.inputVals[ch]) begin
                    mismatches++;
                    `uvm_error("MISMATCH", $sformatf(
                        "ch=%0d exp=%0d got=%0d",
                        ch,
                        exp.inputVals[ch],
                        act.outputVals[ch]
                    ))
                end
            end
        endfunction

        function void check_phase(uvm_phase phase);
            super.check_phase(phase);

            if (exp_q.size() != 0) begin
                `uvm_error("SCOREBOARD", $sformatf("%0d expected outputs never arrived", exp_q.size()))
            end
        endfunction

        function void report_phase(uvm_phase phase);
            super.report_phase(phase);

            if (mismatches == 0) begin
                `uvm_info("RESULT", $sformatf("PASS: %0d channel checks completed", checks), UVM_LOW)
            end else begin
                `uvm_error("RESULT", $sformatf("FAIL: %0d mismatches / %0d checks", mismatches, checks))
            end
        endfunction
    endclass

    class resi_agent extends uvm_agent;
        `uvm_component_utils(resi_agent)

        resi_sequencer sqr;
        resi_driver drv;
        resi_monitor mon;

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

        resi_agent agent;
        resi_scoreboard sb;

        function new(string name, uvm_component parent);
            super.new(name, parent);
        endfunction

        function void build_phase(uvm_phase phase);
            super.build_phase(phase);
            agent = resi_agent::type_id::create("agent", this);
            sb = resi_scoreboard::type_id::create("sb", this);
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

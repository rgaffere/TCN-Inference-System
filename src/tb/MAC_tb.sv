`timescale 1ns/1ps
/*
Author: Ryan G
Description: Testbench for the two mac units I created earlier
Notes:

This is the testplan:

1. Zero multiplication
    - 0 * N + acc

2. Positive × Positive
    - verifies normal signed multiplication

3. Positive × Negative
    - verifies signed arithmetic handling

4. Negative × Negative
    - verifies double-negative behavior

5. Accumulation with positive accumulator
    - verifies MAC accumulation path

6. Accumulation with negative accumulator
    - verifies signed accumulation behavior

7. Boundary-value tests
    - maximum and minimum INT8 values

The combinational MAC is checked immediately after input stimulus.
The pipelined MAC is checked after its expected pipeline latency.
*/

module MAC_tb;
    localparam int IN_LEN = 8;
    localparam int OUT_LEN = 4 * IN_LEN;
    localparam time CLK_PERIOD = 10ns;

    // These are the driving signals
    logic clk, rst_n;

    logic signed [IN_LEN - 1: 0] a, b;
    logic signed [OUT_LEN - 1: 0] acc_in, acc_out_combi, acc_out_pipeline;

    logic valid_in, valid_out;

    // Testbench metrics
    int tests_run, tests_failed;

    // Instantaiate both of the MACS
    MAC_combi dut_combi (
        .x(a),
        .weight(b),
        .acc_in(acc_in),
        .acc_out(acc_out_combi)
    );

    MAC_pipeline dut_pipeline (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .x(a),
        .w(b),
        .acc_in(acc_in),
        .valid_out(valid_out),
        .acc_out(acc_out_pipeline)
    );

    // Inital set up
    initial clk = 1'b0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // Now for some tasks
    task automatic check_combi(        
        input logic signed [IN_LEN - 1: 0] test_a, test_b,
        input logic signed [OUT_LEN - 1: 0] test_acc_in
    );
        logic signed [OUT_LEN - 1: 0] expected;

        a = test_a;
        b = test_b;
        acc_in = test_acc_in;

        expected = (test_a * test_b) + test_acc_in;
        tests_run++;
        // let the logic settle 
        #1;

        if(expected !== acc_out_combi) begin
            tests_failed++;
            $error("COMBI FAIL: a=%0d b=%0d acc=%0d expected=%0d got=%0d",
                   test_a, test_b, test_acc_in, expected, acc_out_combi);
        end
    endtask


    task automatic check_pipe(
        input logic signed [IN_LEN - 1: 0] test_a, test_b,
        input logic signed [OUT_LEN - 1: 0] test_acc_in
    );
        logic signed [OUT_LEN - 1: 0] expected;
        
        @(posedge clk);
        a <= test_a;
        b <= test_b;
        acc_in <= test_acc_in;

        expected = (test_a * test_b) + test_acc_in;
        valid_in <= 1'b1;
        tests_run++;

        @(posedge clk);
        valid_in <= 1'b0;
        @(posedge clk);
        #1;

        if(!valid_out || expected !== acc_out_pipeline) begin
            tests_failed++;
            $error("PIPE FAIL: a=%0d b=%0d acc=%0d expected=%0d got=%0d valid=%0b",
                   test_a, test_b, test_acc_in, expected, acc_out_pipeline, valid_out);
        end
    endtask

    // Testing time
    initial begin
        rst_n = 1'b0;
        valid_in = 1'b0;
        a = '0;
        b = '0;
        acc_in = '0;
        
        tests_run = 0;
        tests_failed = 0;

        // Hold reset for two cycles at the start
        repeat (2) @(posedge clk);
        rst_n = 1'b1;


        // Combinational MAC tests
        check_combi(0,    57,   0);
        check_combi(5,    6,    0);
        check_combi(5,   -6,    0);
        check_combi(-5,  -6,    0);
        check_combi(5,    6,   10);
        check_combi(5,   -6,  -10);
        check_combi(127, 127,   0);
        check_combi(-128, 127,  0);
        check_combi(-128, -128, 0);

        // Pipelined MAC tests
        check_pipe(0,    57,   0);
        check_pipe(5,    6,    0);
        check_pipe(5,   -6,    0);
        check_pipe(-5,  -6,    0);
        check_pipe(5,    6,   10);
        check_pipe(5,   -6,  -10);
        check_pipe(127, 127,   0);
        check_pipe(-128, 127,  0);
        check_pipe(-128, -128, 0);

        if (tests_failed == 0)
            $display("ALL TESTS PASSED: %0d tests run", tests_run);
        else
            $error("TESTS FAILED: %0d / %0d failed", tests_failed, tests_run);

        $finish;
    end
endmodule
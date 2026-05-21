/*
 * conv_MAC_tb.sv
 * Testbench for conv_MAC pipelined accumulator
 *
 * Coverage:
 *   - Functional correctness (positive, negative, mixed signs, zero bias)
 *   - Boundary values (INT8 max/min, all-zeros, bias-only)
 *   - Spurious valid_out check (must be LOW during all accumulation cycles)
 *   - valid_out de-assertion check (must go LOW the cycle after output)
 *   - Back-to-back convolution (pipeline continuity, no idle gap)
 *   - Timeout watchdog (catches hung DUT)
 */

`default_nettype none

module conv_MAC_tb;

    // ----------------------------------------------------------------
    // Parameters (mirror DUT)
    // ----------------------------------------------------------------
    parameter int  IN_LEN        = 8;
    parameter int  KERNEL_LEN    = 3;
    localparam int ACC_LEN       = 4 * IN_LEN;
    localparam int PROD_LEN      = 2 * IN_LEN;

    parameter time CLK_PERIOD    = 10ns;
    parameter int  TIMEOUT_CYCLES = 50;

    // ----------------------------------------------------------------
    // Signals
    // ----------------------------------------------------------------
    logic clk    = 1'b0;
    logic rst_n;
    logic valid_in;
    logic signed [IN_LEN-1:0]  x, w;
    logic signed [ACC_LEN-1:0] bias;
    logic valid_out;
    logic signed [ACC_LEN-1:0] acc_out;

    // Scoreboard
    int tests_run    = 0;
    int tests_failed = 0;

    int start_cycle;
    int end_cycle;
    int cycle_count;

    always @(posedge clk) begin
        if(!rst_n)
            cycle_count <= 0;
        else
            cycle_count <= cycle_count + 1;
    end

    // ----------------------------------------------------------------
    // DUT
    // ----------------------------------------------------------------
    conv_MAC #(
        .IN_LEN(IN_LEN),
        .KERNEL_LEN(KERNEL_LEN)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .valid_in(valid_in),
        .x(x), .w(w),
        .bias(bias),
        .valid_out(valid_out),
        .acc_out(acc_out)
    );

    always #(CLK_PERIOD / 2) clk = ~clk;

    // ----------------------------------------------------------------
    // Tasks
    // ----------------------------------------------------------------

    task automatic reset_dut();
        rst_n    = 1'b0;
        valid_in = 1'b0;
        x        = '0;
        w        = '0;
        bias     = '0;
        repeat(2) @(posedge clk);
        rst_n = 1'b1;
        @(posedge clk); #1;
    endtask

    // Score a single check point
    task automatic check(
        input string test_name,
        input logic  pass,
        input string msg = ""
    );
        tests_run++;
        if (!pass) begin
            tests_failed++;
            if (msg != "")
                $display("FAIL [%s]: %s", test_name, msg);
            else
                $display("FAIL [%s]", test_name);
        end
    endtask

    // Verify output value and valid strobe
    task automatic check_result(
        input string                     test_name,
        input logic signed [ACC_LEN-1:0] expected
    );
        check(test_name, valid_out === 1'b1,
              $sformatf("valid_out=%b (expected 1)", valid_out));
        check(test_name, acc_out === expected,
              $sformatf("acc_out=%0d, expected=%0d", acc_out, expected));
        if (valid_out === 1'b1 && acc_out === expected)
            $display("PASS [%s]", test_name);
    endtask

    // Wait for valid_out with a cycle timeout watchdog
    task automatic wait_for_output(input string test_name);
        fork
            begin : wait_valid
                wait (valid_out === 1'b1);
                #1;
                disable wait_timeout;
            end : wait_valid
            begin : wait_timeout
                repeat (TIMEOUT_CYCLES) @(posedge clk);
                $display("FAIL [%s]: timeout — valid_out never asserted", test_name);
                tests_failed++;
                tests_run++;
                disable wait_valid;
            end : wait_timeout
        join
    endtask

    // ----------------------------------------------------------------
    // Single convolution test
    //   - Drives KERNEL_LEN taps with valid_in held high
    //   - Checks valid_out is LOW on every accumulation cycle
    //   - Waits for valid_out, checks result
    //   - Checks valid_out de-asserts the following cycle
    // ----------------------------------------------------------------
    task automatic run_mac_test(
        input string                     test_name,
        input logic signed [ACC_LEN-1:0] test_bias,
        input logic signed [IN_LEN-1:0]  xs[KERNEL_LEN],
        input logic signed [IN_LEN-1:0]  ws[KERNEL_LEN]
    );
        // Golden reference — compute in wide signed arithmetic
        logic signed [ACC_LEN-1:0] expected;
        logic signed [PROD_LEN-1:0] prod_ref;
        expected = test_bias;
        for (int i = 0; i < KERNEL_LEN; i++) begin
            prod_ref  = xs[i] * ws[i];     // 16-bit signed product
            expected += prod_ref;           // sign-extends to 32-bit on assign
        end

        bias = test_bias;
        @(posedge clk); #1;                // let DUT preload acc_reg with bias
        start_cycle = cycle_count;

        // Drive taps; check no spurious valid_out during accumulation
        for (int i = 0; i < KERNEL_LEN; i++) begin
            valid_in <= 1'b1;
            x        <= xs[i];
            w        <= ws[i];
            @(posedge clk); #1;
            check(test_name, valid_out === 1'b0,
                  $sformatf("valid_out asserted during tap %0d (should be 0)", i));
        end

        // Deassert and wait for result
        valid_in <= 1'b0;
        x        <= '0;
        w        <= '0;

        wait_for_output(test_name);
        end_cycle = cycle_count;
        $display("INFO [%s]: cycles per convolution = %0d",
            test_name, end_cycle - start_cycle);
        check_result(test_name, expected);

        // valid_out must de-assert the cycle after output
        @(posedge clk); #1;
        check(test_name, valid_out === 1'b0,
              "valid_out did not de-assert after output cycle");
    endtask

    // ----------------------------------------------------------------
    // Back-to-back convolution test
    //   - Two convolutions with no gap between them (valid_in never drops)
    //   - Conv0 result appears during conv1's first tap (pipelined overlap)
    //   - Conv1 result appears one cycle after the last tap
    // ----------------------------------------------------------------
    task automatic run_backtoback_test(
        input string                     test_name,
        input logic signed [ACC_LEN-1:0] test_bias,
        input logic signed [IN_LEN-1:0]  xs0[KERNEL_LEN],
        input logic signed [IN_LEN-1:0]  ws0[KERNEL_LEN],
        input logic signed [IN_LEN-1:0]  xs1[KERNEL_LEN],
        input logic signed [IN_LEN-1:0]  ws1[KERNEL_LEN]
    );
        logic signed [ACC_LEN-1:0] expected0, expected1;
        logic signed [PROD_LEN-1:0] prod_ref;

        expected0 = test_bias;
        expected1 = test_bias;
        for (int i = 0; i < KERNEL_LEN; i++) begin
            prod_ref   = xs0[i] * ws0[i]; expected0 += prod_ref;
            prod_ref   = xs1[i] * ws1[i]; expected1 += prod_ref;
        end

        bias = test_bias;
        @(posedge clk); #1;

        // Feed conv0 — no output expected yet
        for (int i = 0; i < KERNEL_LEN; i++) begin
            valid_in <= 1'b1;
            x        <= xs0[i];
            w        <= ws0[i];
            @(posedge clk); #1;
            check({test_name, " [conv0 accum]"}, valid_out === 1'b0,
                  $sformatf("spurious valid_out during conv0 tap %0d", i));
        end

        // Feed conv1 back-to-back — conv0 result overlaps with conv1's first tap
        for (int i = 0; i < KERNEL_LEN; i++) begin
            valid_in <= 1'b1;
            x        <= xs1[i];
            w        <= ws1[i];
            @(posedge clk); #1;

            if (i == 0) begin
                // Pipeline overlap: conv0 result fires while conv1 tap 0 is clocked in
                check_result({test_name, " [conv0]"}, expected0);
            end else begin
                check({test_name, " [conv1 accum]"}, valid_out === 1'b0,
                      $sformatf("spurious valid_out during conv1 tap %0d", i));
            end
        end

        // Deassert and collect conv1 result
        valid_in <= 1'b0;
        x        <= '0;
        w        <= '0;

        wait_for_output({test_name, " [conv1]"});
        check_result({test_name, " [conv1]"}, expected1);

        @(posedge clk); #1;
        check({test_name, " [conv1]"}, valid_out === 1'b0,
              "valid_out did not de-assert after conv1 output");
    endtask

    // ----------------------------------------------------------------
    // Stimulus
    // ----------------------------------------------------------------
    initial begin
        reset_dut();

        // --- Functional correctness ---
        run_mac_test("all positive",    10, '{2, 4, 6},       '{3, 5, 7});
        run_mac_test("negative weight",  5, '{3, 4, 2},       '{-2, 3, -5});
        run_mac_test("negative input",  -8, '{-3, 2, -1},     '{4, 5, -6});
        run_mac_test("zero bias",        0, '{1, 2, 3},       '{1, 2, 3});

        // --- Boundary values ---
        run_mac_test("max positive",     0, '{127, 127, 127}, '{127, 127, 127}); // 3*127^2=48387
        run_mac_test("max neg input",    0, '{-128,-128,-128},'{127, 127, 127}); // 3*(-128*127)
        run_mac_test("all zeros",        0, '{0, 0, 0},       '{0, 0, 0});
        run_mac_test("bias only",       42, '{0, 0, 0},       '{0, 0, 0});
        run_mac_test("neg bias",       -42, '{1, 2, 3},       '{1, 2, 3});

        // --- Pipeline continuity ---
        run_backtoback_test("back-to-back", 5,
            '{1, 2, 3},  '{3, 2, 1},
            '{-1,-2,-3}, '{3, 2, 1});

        // --- Results ---
        $display("\n=============================");
        $display("Tests run:    %0d", tests_run);
        $display("Tests failed: %0d", tests_failed);
        if (tests_failed == 0)
            $display("ALL TESTS PASSED");
        else
            $display("SOME TESTS FAILED");
        $display("=============================");

        $finish;
    end

endmodule

`default_nettype wire
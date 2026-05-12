`timescale 1ns/1ps

module dilation_offsets_tb;
    parameter int DEPTH = 512;
    parameter int KERNEL_LEN = 3;
    parameter int NUM_D = 3;

    localparam int ADDR_WIDTH = $clog2(DEPTH);

    int tests_run, tests_failed;

    // [dilation_idx][tap]
    logic [ADDR_WIDTH-1:0] offsets [0:NUM_D-1][0:KERNEL_LEN-1];

    dilation_offsets #(.D(1)) dut_d1 (.read_offsets(offsets[0]));
    dilation_offsets #(.D(2)) dut_d2 (.read_offsets(offsets[1]));
    dilation_offsets #(.D(4)) dut_d4 (.read_offsets(offsets[2]));

    task automatic check_offsets(
        input int dut_idx,
        input int expected [0:KERNEL_LEN-1],
        input string test_name
    );
        bit local_fail;
        local_fail = 0;

        #1;
        tests_run++;

        for (int i = 0; i < KERNEL_LEN; i++) begin
            if (offsets[dut_idx][i] !== expected[i][ADDR_WIDTH-1:0]) begin
                local_fail = 1;
                $error("%s FAIL[%0d]: expected=%0d actual=%0d",
                    test_name, i, expected[i], offsets[dut_idx][i]);
            end
        end

        if (!local_fail) $display("%s PASS", test_name);
        else tests_failed++;
    endtask

    initial begin
        int d_vals [0:NUM_D-1] = '{1, 2, 4};
        int expected [0:NUM_D-1][0:KERNEL_LEN-1];
        string test_name;

        tests_run = 0;
        tests_failed = 0;

        // Build expected table from D values
        for (int d = 0; d < NUM_D; d++)
            for (int k = 0; k < KERNEL_LEN; k++)
                expected[d][k] = k * d_vals[d];

        // Run all checks in one loop
        for (int d = 0; d < NUM_D; d++) begin
            test_name = $sformatf("DILATION_OFFSETS_D%0d", d_vals[d]);
            check_offsets(d, expected[d], test_name);
        end

        $display("-----------------------------");
        $display("Dilation Offset TB Complete");
        $display("Tests run    = %0d", tests_run);
        $display("Tests failed = %0d", tests_failed);
        $display("-----------------------------");

        if (tests_failed == 0) $display("ALL TESTS PASSED");
        else                   $fatal(1, "TESTBENCH FAILED");

        $finish;
    end

endmodule
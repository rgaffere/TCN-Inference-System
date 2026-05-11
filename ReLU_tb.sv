module ReLU_tb;
    parameter int DATA_LEN = 8;

    int tests_run, tests_failed;

    logic signed [DATA_LEN - 1: 0] data_in;
    logic signed [DATA_LEN - 1: 0] data_out;

    // Initialize module
    ReLU dut_relu (
        .data_in(data_in),
        .data_out(data_out)
    );

    // A generic testing task
    task automatic check_relu(
        input logic signed [DATA_LEN - 1: 0] test_data_in,
        // Im putting expected as an input now because I feel like doing it inside the task is a circular proof
        input logic signed [DATA_LEN - 1: 0] expected
    );
        data_in = test_data_in;

        // ReLU is combinational so wait for signal to settle
        #1;
        tests_run++;
        
        if(expected !== data_out) begin
            tests_failed++;
            $error("ReLU FAIL: input=%0d output=%0d expected=%0d",
                test_data_in, data_out, expected);
        end
    endtask

    // Were gonna test positive, negative, zero, as well as boundary tests
    initial begin
        data_in = '0;
        tests_run = 0;
        tests_failed = 0;

        check_relu(8'sd1, 8'sd1);
        check_relu(-8'sd1, 8'sd0);
        check_relu(8'sd0, 8'sd0);
        check_relu(8'sd127, 8'sd127);
        check_relu(-8'sd128, 8'sd0);

        if (tests_failed == 0)
            $display("ALL TESTS PASSED: %0d tests run", tests_run);
        else
            $error("TESTS FAILED: %0d / %0d failed", tests_failed, tests_run);

        $finish;
    end
endmodule
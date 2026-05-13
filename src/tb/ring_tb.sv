module ring_tb;
    parameter int DATA_LEN = 8;
    parameter int DEPTH = 512;
    parameter time CLK_PERIOD = 10ns;
    localparam int ADDR_WIDTH = $clog2(DEPTH);

    int tests_run, tests_failed;

    logic clk, rst_n, write_en;

    logic signed [DATA_LEN - 1: 0] data_in;
    logic signed [DATA_LEN - 1: 0] data_out;

    logic [ADDR_WIDTH - 1: 0] read_offset;

    // Clock generation
    initial clk = 1'b0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // Initialize the DUT
    ring dut_ring(
        .clk(clk),
        .rst_n(rst_n),
        .write_en(write_en),
        .data_in(data_in),
        .read_offset(read_offset),
        .data_out(data_out)
    );

    // Testing tasks definitions
    task automatic check_read(
        input logic signed [DATA_LEN - 1: 0] test_data_out,
        input logic [ADDR_WIDTH - 1: 0] test_read_offset
    );
        @(negedge clk);
        read_offset = test_read_offset;

        // 1 cycle to read
        @(posedge clk);
        #1;
        tests_run++;

        if (data_out !== test_data_out) begin
            tests_failed++;
            $error("READ FAIL: offset=%0d expected=%0d actual=%0d",
                test_read_offset, test_data_out, data_out);
        end
    endtask

    task automatic write_sample(
        input logic signed [DATA_LEN - 1: 0] test_data_in
    );
        @(negedge clk);
        data_in = test_data_in;
        write_en = 1'b1;

        // Wait 1 cycle
        @(negedge clk);
        write_en = 1'b0;
    endtask

    // Testing
    initial begin
        // Initialize signals
        rst_n = 1'b0;
        write_en = 1'b0;
        data_in = '0;
        read_offset = '0;

        tests_run = 0;
        tests_failed = 0;

        // Reset DUT
        repeat (2) @(posedge clk);
        rst_n = 1'b1;
        @(posedge clk);

        // Zero padding before any writes
        check_read(8'sd0, 0);
        check_read(8'sd0, 1);
        check_read(8'sd0, 4);

        // Write one sample and verify newest sample
        write_sample(8'sd10);

        check_read(8'sd10, 0);
        check_read(8'sd0, 1);
        check_read(8'sd0, 2);

        // Write known samples into the ring
        write_sample(8'sd20);
        write_sample(8'sd30);
        write_sample(8'sd40);
        write_sample(8'sd50);

        // Most recent samples
        check_read(8'sd50, 0);
        check_read(8'sd40, 1);
        check_read(8'sd30, 2);

        // Dilated-style read
        check_read(8'sd50, 0);
        check_read(8'sd30, 2);
        check_read(8'sd10, 4);

        // Zero padding when read offset is older than written history
        check_read(8'sd0, 5);
        check_read(8'sd0, 12);

        // Write another sample and verify head moved
        write_sample(-8'sd12);

        check_read(-8'sd12, 0);
        check_read(8'sd50, 1);
        check_read(8'sd40, 2);

        // Fill past DEPTH to test ring wrap around
        for (int i = 0; i < DEPTH; i++) begin
            write_sample(logic'(i[DATA_LEN - 1: 0]));
        end

        // After wrap around, newest value should be DEPTH - 1 truncated to DATA_LEN
        check_read(logic'((DEPTH - 1)[DATA_LEN - 1: 0]), 0);
        check_read(logic'((DEPTH - 2)[DATA_LEN - 1: 0]), 1);
        check_read(logic'((DEPTH - 3)[DATA_LEN - 1: 0]), 2);

        // Old pre-wrap data should be overwritten
        check_read(logic'((DEPTH - 8)[DATA_LEN - 1: 0]), 7);

        // Summary
        if (tests_failed == 0) begin
            $display("ALL TESTS PASSED: %0d tests run", tests_run);
        end else begin
            $error("TESTS FAILED: %0d / %0d failed", tests_failed, tests_run);
        end

        $finish;
    end
endmodule
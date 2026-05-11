module ring_tb;
    parameter int DATA_LEN = 8;
    parameter int DEPTH = 512;
    parameter int KERNEL_LEN = 3;
    parameter time CLK_PERIOD = 10ns;
    localparam int ADDR_WIDTH = $clog2(DEPTH);

    int tests_run, tests_failed;

    logic clk, rst_n, write_en;

    logic signed [DATA_LEN - 1: 0] data_in;
    logic signed [DATA_LEN - 1: 0] data_out [0: KERNEL_LEN - 1];

    logic [ADDR_WIDTH - 1: 0] read_offsets [0: KERNEL_LEN - 1];

    // Clock generation
    initial clk = 1'b0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // Initialize the DUT
    ring dut_ring(
        .clk(clk),
        .rst_n(rst_n),
        .write_en(write_en),
        .data_in(data_in),
        .read_offsets(read_offsets),
        .data_out(data_out)
    );

    // Testing tasks definitions
    task automatic check_read(
        input logic signed [DATA_LEN - 1: 0] test_data_out [0: KERNEL_LEN - 1],
        input logic [ADDR_WIDTH - 1: 0] test_read_offsets [0: KERNEL_LEN - 1]
    );
        @(negedge clk);
        for (int i = 0; i < KERNEL_LEN; i++) begin
            read_offsets[i] = test_read_offsets[i];
        end

        // 1 cycle to read
        @(posedge clk);
        #1;
        tests_run++;

        for (int i = 0; i < KERNEL_LEN; i++) begin
            if (data_out[i] !== test_data_out[i]) begin
                tests_failed++;
                $error("READ FAIL[%0d]: offset=%0d expected=%0d actual=%0d",
                    i, test_read_offsets[i], test_data_out[i], data_out[i]);
            end
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
        logic signed [DATA_LEN - 1: 0] expected [0: KERNEL_LEN - 1];
        logic [ADDR_WIDTH - 1: 0] offsets [0: KERNEL_LEN - 1];

        // Initialize signals
        rst_n = 1'b0;
        write_en = 1'b0;
        data_in = '0;

        for (int i = 0; i < KERNEL_LEN; i++) begin
            read_offsets[i] = '0;
        end

        tests_run = 0;
        tests_failed = 0;

        // Reset DUT
        repeat (2) @(posedge clk);
        rst_n = 1'b1;
        @(posedge clk);

        // Write known samples into the ring
        write_sample(8'sd10);
        write_sample(8'sd20);
        write_sample(8'sd30);
        write_sample(8'sd40);
        write_sample(8'sd50);

        // Most recent sample
        offsets[0] = 0;
        offsets[1] = 1;
        offsets[2] = 2;

        expected[0] = 8'sd50;
        expected[1] = 8'sd40;
        expected[2] = 8'sd30;

        check_read(expected, offsets);

        // Dilated-style read
        offsets[0] = 0;
        offsets[1] = 2;
        offsets[2] = 4;

        expected[0] = 8'sd50;
        expected[1] = 8'sd30;
        expected[2] = 8'sd10;

        check_read(expected, offsets);

        // Write another sample and verify head moved
        write_sample(-8'sd12);

        offsets[0] = 0;
        offsets[1] = 1;
        offsets[2] = 2;

        expected[0] = -8'sd12;
        expected[1] = 8'sd50;
        expected[2] = 8'sd40;

        check_read(expected, offsets);

        // Summary
        if (tests_failed == 0) begin
            $display("ALL TESTS PASSED: %0d tests run", tests_run);
        end else begin
            $error("TESTS FAILED: %0d / %0d failed", tests_failed, tests_run);
        end

        $finish;
    end
endmodule
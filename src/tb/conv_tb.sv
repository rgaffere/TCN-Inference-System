`timescale 1ns/1ps

module conv_tb;

    parameter int DATA_LEN   = 8;
    parameter int DEPTH      = 512;
    parameter int KERNEL_LEN = 3;
    parameter int ACC_LEN    = 32;
    parameter time CLK_PERIOD = 10ns;

    logic clk, rst_n, valid_in;
    logic valid_out;
    logic saw_valid_out;

    logic signed [ACC_LEN-1:0] bias;
    logic signed [DATA_LEN-1:0] weights [0:KERNEL_LEN-1];

    int cycles;
    int tests_run, tests_failed;
    int conv_start_cycle;
    int conv_done_cycle;
    int cycles_per_conv;

    int valid_count;
    int first_valid_cycle;
    int second_valid_cycle;
    int valid_spacing;

    conv #(
        .DATA_LEN(DATA_LEN),
        .DEPTH(DEPTH),
        .KERNEL_LEN(KERNEL_LEN)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .bias(bias),
        .weights(weights),
        .valid_out(valid_out)
    );

    always #(CLK_PERIOD / 2) clk = ~clk;

    task check(input string name, input logic condition);
        tests_run++;

        if(condition) begin
            $display("[PASS] %s", name);
        end else begin
            tests_failed++;
            $display("[FAIL] %s", name);
        end
    endtask

    task reset_dut();
        begin
            clk = 1'b0;
            rst_n = 1'b0;
            valid_in = 1'b0;
            bias = '0;

            for (int i = 0; i < KERNEL_LEN; i++) begin
                weights[i] = '0;
            end

            saw_valid_out = 1'b0;
            cycles = 0;
            conv_start_cycle = 0;
            conv_done_cycle = 0;
            cycles_per_conv = 0;
            valid_count = 0;
            first_valid_cycle = 0;
            second_valid_cycle = 0;
            valid_spacing = 0;

            repeat(3) @(posedge clk);
            rst_n = 1'b1;
            @(posedge clk);
        end
    endtask

    task preload_ring();
        begin
            force dut.s3 = 1'b1;

            force dut.write_in = 8'sd10;
            @(posedge clk);

            force dut.write_in = 8'sd20;
            @(posedge clk);

            force dut.write_in = 8'sd30;
            @(posedge clk);

            force dut.write_in = 8'sd40;
            @(posedge clk);

            release dut.s3;
            release dut.write_in;

            @(posedge clk);
        end
    endtask

    task check_ring_read(
        input int offset,
        input logic signed [DATA_LEN - 1:0] expected
    );
        begin
            force dut.read_offset = offset[($clog2(DEPTH))-1:0];

            @(posedge clk);
            #1;

            check($sformatf("ring read offset %0d returns %0d", offset, expected), dut.x == expected);

            release dut.read_offset;
            @(posedge clk);
        end
    endtask

    task run_conv_valid();
        begin
            saw_valid_out = 1'b0;
            conv_start_cycle = cycles;

            valid_in = 1'b1;
            repeat(KERNEL_LEN + 2) @(posedge clk);
            valid_in = 1'b0;
        end
    endtask

    task run_back_to_back_convs();
        begin
            valid_count = 0;
            first_valid_cycle = 0;
            second_valid_cycle = 0;

            valid_in = 1'b1;

            repeat((KERNEL_LEN + 2) * 2) @(posedge clk);

            valid_in = 1'b0;

            repeat(10) @(posedge clk);
        end
    endtask

    task print_state();
        begin
            $display("cycle=%0d valid_in=%0b s0=%0b s1=%0b s2=%0b s3=%0b tap_idx=%0d valid_out=%0b saw_valid_out=%0b x=%0d w=%0d mac_out=%0d relu_out=%0d write_in=%0d head=%0d warmup=%0d",
                cycles,
                valid_in,
                dut.s0,
                dut.s1,
                dut.s2,
                dut.s3,
                dut.tap_idx,
                valid_out,
                saw_valid_out,
                dut.x,
                dut.w,
                dut.mac_out,
                dut.relu_out,
                dut.write_in,
                dut.init_ring.head,
                dut.init_ring.warmup_count
            );
        end
    endtask

    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            saw_valid_out <= 1'b0;
            conv_done_cycle <= 0;
            valid_count <= 0;
            first_valid_cycle <= 0;
            second_valid_cycle <= 0;
        end else begin
            if(valid_out && !saw_valid_out) begin
                saw_valid_out <= 1'b1;
                conv_done_cycle <= cycles;
            end

            if(valid_out) begin
                valid_count <= valid_count + 1;

                if(valid_count == 0) begin
                    first_valid_cycle <= cycles;
                end else if(valid_count == 1) begin
                    second_valid_cycle <= cycles;
                end
            end
        end
    end

    always @(posedge clk) begin
        if(rst_n) begin
            cycles++;
            print_state();
        end
    end

    initial begin
        $dumpfile("conv_tb.vcd");
        $dumpvars(0, conv_tb);

        tests_run = 0;
        tests_failed = 0;

        reset_dut();

        check("reset clears s0", dut.s0 == 1'b0);
        check("reset clears s1", dut.s1 == 1'b0);
        check("reset clears s3", dut.s3 == 1'b0);
        check("reset clears tap_idx", dut.tap_idx == '0);

        weights[0] = 8'sd1;
        weights[1] = 8'sd2;
        weights[2] = 8'sd3;
        bias       = 32'sd0;

        preload_ring();

        check("ring head after preload", dut.init_ring.head == 4);
        check("ring warmup after preload", dut.init_ring.warmup_count == 4);

        check_ring_read(0, 8'sd40);
        check_ring_read(1, 8'sd30);
        check_ring_read(2, 8'sd20);
        check_ring_read(3, 8'sd10);

        saw_valid_out = 1'b0;

        $display("\nStarting single convolution valid window...\n");

        run_conv_valid();

        repeat(12) @(posedge clk);

        cycles_per_conv = conv_done_cycle - conv_start_cycle;

        check("valid_out eventually asserted", saw_valid_out);
        check("tap_idx stays in valid range", dut.tap_idx < KERNEL_LEN);

        check("mac_out matches current aligned convolution", dut.mac_out == 32'sd190);
        check("relu_out matches positive mac_out", dut.relu_out == 32'sd190);

        check("quant shifted result matches current quant behavior", dut.write_in == 8'sd0);

        check("ring head advanced after conv write", dut.init_ring.head == 5);
        check("ring warmup advanced after conv write", dut.init_ring.warmup_count == 5);

        `ifndef SYNTHESIS
            check("ring wrote quantized conv result", dut.init_ring.mem[4] == 8'sd0);
        `endif

        $display("\nSingle convolution cycle count:");
        $display("conv_start_cycle = %0d", conv_start_cycle);
        $display("conv_done_cycle  = %0d", conv_done_cycle);
        $display("cycles_per_conv  = %0d", cycles_per_conv);

        $display("\nStarting back-to-back convolution throughput test...\n");

        run_back_to_back_convs();

        valid_spacing = second_valid_cycle - first_valid_cycle;

        check("back-to-back test produced at least two valid outputs", valid_count >= 2);
        check("first valid cycle was captured", first_valid_cycle > 0);
        check("second valid cycle was captured", second_valid_cycle > 0);

        $display("\nBack-to-back throughput:");
        $display("valid_count        = %0d", valid_count);
        $display("first_valid_cycle  = %0d", first_valid_cycle);
        $display("second_valid_cycle = %0d", second_valid_cycle);
        $display("valid_spacing      = %0d cycles", valid_spacing);

        check("back-to-back valid spacing is 3 cycles", valid_spacing == 3);

        $display("\nTests run:    %0d", tests_run);
        $display("Tests failed: %0d", tests_failed);

        if(tests_failed == 0) begin
            $display("\nALL TESTS PASSED");
        end else begin
            $display("\nSOME TESTS FAILED");
        end

        $finish;
    end

endmodule
/*
Author: Ryan G
Description: Ring buffer for O(1) address generation with 1-cycle synchronous read latency.
Notes:
This module models a 1RW1R ring-buffered activation memory.

Each SRAM word stores 4 INT8 channels. One read returns 4 channel activations.
Four ring instances cover 16 channels.

We use all 32 bits so we can have one ring per 4 channels, giving 100% sram usage.
*/

module ring #(
    parameter int DEPTH = 512,
    parameter int DATA_LEN = 8,
    localparam int NUM_BYTES = 4,
    localparam int WORD_LEN = 32,
    localparam int ADDR_WIDTH = $clog2(DEPTH)
)(
`ifdef USE_POWER_PINS
    inout vccd1,
    inout vssd1,
`endif 

    input logic clk, rst_n, write_en,

    input logic signed [DATA_LEN - 1: 0] data_in [0: NUM_BYTES - 1],
    input logic [ADDR_WIDTH - 1: 0] read_offset,

    output logic signed [DATA_LEN - 1: 0] data_out [0: NUM_BYTES - 1]
);
    logic [ADDR_WIDTH - 1: 0] head;
    logic [ADDR_WIDTH: 0] warmup_count;
    
    logic valid_read, valid_read_q;
    logic [ADDR_WIDTH - 1: 0] read_addr;


    logic [WORD_LEN - 1: 0] sram_din0;
    logic [WORD_LEN - 1: 0] sram_dout0;
    logic [WORD_LEN - 1: 0] sram_dout1;

    // read logic
    assign read_addr = head - read_offset - 1'b1;
    assign valid_read = (read_offset < warmup_count);

    // pack all four channel data into one word
    assign sram_din0 = {data_in[3], data_in[2], data_in[1], data_in[0]};

    // write logic
    always_ff @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            head <= '0;
            warmup_count <= '0;
            valid_read_q <= 1'b0;
        end else begin
            valid_read_q <= valid_read;
            if (write_en) begin
                
                head <= head + 1'b1;
                
                if(warmup_count < DEPTH) begin
                    warmup_count <= warmup_count + 1'b1;
                end
            end
        end
    end

    `ifdef SYNTHESIS
        assign data_out[0] = valid_read_q ? $signed(sram_dout1[DATA_LEN - 1: 0]) : '0;
        assign data_out[1] = valid_read_q ? $signed(sram_dout1[2 * DATA_LEN - 1: DATA_LEN]) : '0;
        assign data_out[2] = valid_read_q ? $signed(sram_dout1[3 * DATA_LEN - 1: 2 * DATA_LEN]) : '0;
        assign data_out[3] = valid_read_q ? $signed(sram_dout1[WORD_LEN - 1: 3 * DATA_LEN]) : '0;
        
        sky130_sram_2kbyte_1rw1r_32x512_8 u_ring_sram (
        `ifdef USE_POWER_PINS
            .vccd1(vccd1),
            .vssd1(vssd1),
        `endif

            // Port 0: write/read-write port
            .clk0(clk),
            .csb0(~write_en),
            .web0(~write_en),
            .wmask0(4'b1111),
            .addr0(head),
            .din0(sram_din0),
            .dout0(sram_dout0),

            // Port 1: read-only port
            .clk1(clk),
            .csb1(~valid_read),
            .addr1(read_addr),
            .dout1(sram_dout1)
        );
    `else

        logic [WORD_LEN - 1: 0] mem [0: DEPTH-1];
        logic [WORD_LEN - 1: 0] sim_dout_q;

        assign data_out[0] = valid_read_q ? $signed(sim_dout_q[DATA_LEN - 1: 0]) : '0;
        assign data_out[1] = valid_read_q ? $signed(sim_dout_q[2 * DATA_LEN - 1: DATA_LEN]) : '0;
        assign data_out[2] = valid_read_q ? $signed(sim_dout_q[3 * DATA_LEN - 1: 2 * DATA_LEN]) : '0;
        assign data_out[3] = valid_read_q ? $signed(sim_dout_q[WORD_LEN - 1: 3 * DATA_LEN]) : '0;

        always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                sim_dout_q <= '0;
            end else begin
                if (write_en) begin
                    mem[head] <= sram_din0;
                end
                if (valid_read) begin
                    sim_dout_q <= mem[read_addr];
                end else begin
                    sim_dout_q <= '0;
                end
            end
        end
    `endif
endmodule
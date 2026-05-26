/*
Author: Ryan G
Description: Ring buffer for O(1) address generation with 1-cycle synchronous read latency.
Notes:
This module models a 1W1R ring-buffered activation memory.

Since MACs are channel-wise parallel, we only need one read.
*/

module ring #(
    parameter int DEPTH = 512,
    parameter int DATA_LEN = 8,
    localparam int ADDR_WIDTH = $clog2(DEPTH)
)(
`ifdef USE_POWER_PINS
    inout vccd1,
    inout vssd1,
`endif 

    input logic clk, rst_n, write_en,

    input logic signed [DATA_LEN - 1: 0] data_in,
    input logic [ADDR_WIDTH - 1: 0] read_offset,

    output logic signed [DATA_LEN - 1: 0] data_out
);
    logic [ADDR_WIDTH - 1: 0] head;
    logic [ADDR_WIDTH: 0] warmup_count;
    
    logic valid_read, valid_read_q;
    logic [ADDR_WIDTH - 1: 0] read_addr;


    logic [31:0] sram_din0;
    logic [31:0] sram_dout0;
    logic [31:0] sram_dout1;

    // read logic
    assign read_addr = head - read_offset - 1'b1;
    assign valid_read = (read_offset < warmup_count);

    // Use byte lane 0 for one INT8 activation.
    assign sram_din0 = {24'b0, data_in};

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
        // Gate output one cycle later to match synchronous SRAM read.
        assign data_out = valid_read_q ? sram_dout1[DATA_LEN-1:0] : '0;
        
        // SRAM macro
        sky130_sram_2kbyte_1rw1r_32x512_8 u_ring_sram (
            `ifdef USE_POWER_PINS
            .vccd1(vccd1),
            .vssd1(vssd1),
            `endif

            // Only using byte lane 0 for writes
            .clk0(clk),
            .csb0(~write_en),
            .web0(~write_en),
            .wmask0(4'b0001),
            .addr0(head),
            .din0(sram_din0),
            .dout0(sram_dout0),

            .clk1(clk),
            .csb1(1'b0),
            .addr1(read_addr),
            .dout1(sram_dout1)
        );

    `else

        logic signed [DATA_LEN-1:0] mem [0:DEPTH-1];
        logic signed [DATA_LEN-1:0] sim_dout_q;

        assign data_out = sim_dout_q;

        always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                sim_dout_q <= '0;
            end else begin
                if (write_en)
                    mem[head] <= data_in;

                sim_dout_q <= valid_read_q ? mem[read_addr] : '0;
            end
        end

    `endif
endmodule
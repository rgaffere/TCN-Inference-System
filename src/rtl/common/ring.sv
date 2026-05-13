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
    input logic clk, rst_n, write_en,

    input logic signed [DATA_LEN - 1: 0] data_in,
    input logic [ADDR_WIDTH - 1: 0] read_offset,

    output logic signed [DATA_LEN - 1: 0] data_out
);
    logic [ADDR_WIDTH - 1: 0] head;
    logic [ADDR_WIDTH: 0] warmup_count;
    
    logic valid_read;
    logic [ADDR_WIDTH - 1: 0] read_addr;

    logic signed [DATA_LEN-1:0] mem [0:DEPTH-1];

    
    // read logic
    assign read_addr = head - read_offset - 1'b1;
    assign valid_read = (read_offset < warmup_count);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            data_out <= '0;
        end else begin
            data_out <= valid_read ? mem[read_addr] : '0;
        end
    end


    // write logic
    always_ff @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            head <= '0;
            warmup_count <= '0;

        end else if (write_en) begin
            mem[head] <= data_in;
            head <= head + 1'b1;
            
            if(warmup_count < DEPTH) begin
                warmup_count <= warmup_count + 1'b1;
            end
        end
    end
    

endmodule
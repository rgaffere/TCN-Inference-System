/*
Author: Ryan G
Description: Ring buffer for O(1) address generation with 1-cycle synchronous read latency.
Notes:
This module models a 1W3R ring-buffered activation memory.

We're using replicated SRAM banks. Throughput is higher than single-read SRAM access.
Also, cost/complexity is better than using a true 3-read SRAM macro.
*/

module ring #(
    parameter int DEPTH = 512,
    parameter int DATA_LEN = 8,
    parameter int KERNEL_LEN = 3,
    localparam int ADDR_WIDTH = $clog2(DEPTH)
)(
    input logic clk, rst_n, write_en,

    input logic signed [DATA_LEN - 1: 0] data_in,
    input var logic [ADDR_WIDTH - 1: 0] read_offsets [0: KERNEL_LEN - 1],

    output logic signed [DATA_LEN - 1: 0] data_out [0: KERNEL_LEN - 1]
);
    logic [ADDR_WIDTH - 1: 0] head;
    logic [ADDR_WIDTH: 0] warmup_count;
    
    logic [KERNEL_LEN - 1: 0] valid_reads;
    logic [ADDR_WIDTH - 1: 0] read_addr [0: KERNEL_LEN - 1];

    // Replicated SRAM-style banks.
    // mem[bank][address]
    logic signed [DATA_LEN-1:0] mem [0:KERNEL_LEN-1][0:DEPTH-1];

    
    // read logic
    genvar i;
    generate
        for (i = 0; i < KERNEL_LEN; i++) begin : gen_read_taps
            assign read_addr[i] = head - read_offsets[i] - 1'b1;
            assign valid_reads[i] = (read_offsets[i] < warmup_count);

            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    data_out[i] <= '0;
                end else begin
                    data_out[i] <= valid_reads[i] ? mem[i][read_addr[i]] : '0;
                end
            end
        end
    endgenerate


    // write logic
    genvar j;
    generate
        for(j = 0; j < KERNEL_LEN; j++) begin : gen_write_taps
            always_ff @(posedge clk) begin
                if(write_en) begin
                    mem[j][head] <= data_in;
                end
            end
        end
    endgenerate


    always_ff @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            head <= '0;
            warmup_count <= '0;

        end else if (write_en) begin
            head <= head + 1'b1;
            
            if(warmup_count < DEPTH) begin
                warmup_count <= warmup_count + 1'b1;
            end
        end
    end
    

endmodule
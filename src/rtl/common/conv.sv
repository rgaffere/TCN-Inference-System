module conv #(
    parameter int DATA_LEN = 8,
    parameter int DEPTH = 512,
    parameter int KERNEL_LEN = 3,
    localparam int ADDR_WIDTH = $clog2(DEPTH)
)(
    input logic clk, rst_n, valid_in,
    output logic valid_out
);
    typedef enum bit [1:0] {0, 1, 2, 3, 4} idx;
    
    logic [ADDR_WIDTH - 1: 0] offsets [0: KERNEL_LEN - 1];
    logic [ADDR_WIDTH - 1: 0] curr_offset;

    logic mac_done;
    logic [DATA_LEN - 1: 0] data_in, mac_out, read_out, acc_in, relu_in, relu_out, acc_temp;
    logic [DATA_LEN - 1: 0] mac_out [0: KERNEL_LEN - 1];

    // Initialize modules
    ring init_ring (
        .clk(clk),
        .rst_n(rst_n)
        .write_en(mac_done),
        .data_in(data_in),
        .read_offset(curr_offset),
        .data_out(read_out)
    );

    ReLU init_relu (
        .data_in(relu_in),
        .data_out(relu_out)
    );

    MAC init_mac (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .x(read_out),
        .w(w),
        .acc_in(b),
        .valid_out(mac_done),
        .acc_out(mac_out)
    );

    dilation_offsets init_offsets(
        .read_offsets(offsets)
    );

    assign acc_in = mac_out[0] + mac_out[1] + mac_out[2];
    assign data_in = relu_out;

    always_ff @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            mac_done <= 1'b0;
            valid_out <= 1'b0;
            idx <= 0;
        end else if (valid_in) begin
            if(idx == 0) begin
                valid_out <= 1'b0;
                curr_offset <= offsets[0];
            end else if (idx == 1) begin
                curr_offset <= offset[1];
            end else if (idx == 2) begin
                curr_offset <= offset[2];
            end else if (idx == 3) begin
                curr_offset <= offset[3];
            end else begin
                mac_done <= 1'b1;
                relu_in <= acc_in;
                valid_out <= mac_done;
            end
        end
    end
endmodule
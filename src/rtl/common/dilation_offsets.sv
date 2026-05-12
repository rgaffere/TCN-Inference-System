/*
Author: Ryan G
Description: Dedicated module that computes the offsets for the dilation taps, cleaner solution upstream
*/

module dilation_offsets #(
    parameter int D = 1,
    parameter int DEPTH = 512,
    parameter int KERNEL_LEN = 3,
    localparam int ADDR_WIDTH = $clog2(DEPTH)
)(
    output logic [ADDR_WIDTH - 1: 0] read_offsets [0: KERNEL_LEN - 1]
);
    // Since dilation factor is constant so is the offsets
    // note that initialization upstream of dilation will determine offsets via parameterization
    genvar i;
    generate
        for(i = 0; i < KERNEL_LEN; i++) begin : gen_offset_taps
            assign read_offsets[i] = i * D;
        end
    endgenerate

endmodule
// Im going to do a real quantization scheme later, i first have to check the weights and such
module quant #(
    parameter int IN_LEN = 32,
    parameter int OUT_LEN = 8,
    parameter int SHIFT = 8
)(
    input  logic signed [IN_LEN - 1:0] data_in,
    output logic signed [OUT_LEN - 1:0] data_out
);

    assign data_out = data_in >>> SHIFT;

endmodule
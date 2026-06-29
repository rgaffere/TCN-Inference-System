// Vectorized quantization scheme, a better version will come in later versions. Right now this one shifts right and clamps
module quant_array #(
    parameter int IN_LEN = 32,
    parameter int OUT_LEN = 8,
    parameter int SHIFT = 8,
    parameter int NUM_CHANNELS = 16
)(
    input logic signed [IN_LEN-1:0] data_in [0:NUM_CHANNELS-1],
    output logic signed [OUT_LEN-1:0] data_out [0:NUM_CHANNELS-1]
);

    genvar i;
    generate
        for (i = 0; i < NUM_CHANNELS; i++) begin : gen_quant_lane
            logic signed [IN_LEN-1:0] shifted;

            assign shifted = data_in[i] >>> SHIFT;

            always_comb begin
                if (shifted > $signed({1'b0, {(OUT_LEN-1){1'b1}}}))
                    data_out[i] = $signed({1'b0, {(OUT_LEN-1){1'b1}}}); // +127
                else if (shifted < $signed({1'b1, {(OUT_LEN-1){1'b0}}}))
                    data_out[i] = $signed({1'b1, {(OUT_LEN-1){1'b0}}}); // -128
                else
                    data_out[i] = shifted[OUT_LEN-1:0];
            end
        end
    endgenerate

endmodule
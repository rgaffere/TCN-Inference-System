module quant_array #(
    parameter int IN_LEN = 32,
    parameter int OUT_LEN = 8,
    parameter int NUM_CHANNELS = 16,
    localparam int SHIFT_LEN = $clog2(IN_LEN)
)(
    input logic [SHIFT_LEN - 1: 0] shift,
    input var logic signed [IN_LEN - 1: 0] data_in [0: NUM_CHANNELS - 1],
    output var logic signed [OUT_LEN - 1: 0] data_out [0: NUM_CHANNELS - 1]
);

    genvar i;
    generate
        for (i = 0; i < NUM_CHANNELS; i++) begin : gen_quant_lane
            logic signed [IN_LEN: 0] extended;
            logic signed [IN_LEN: 0] rounded;
            logic signed [IN_LEN: 0] shifted;
            logic signed [IN_LEN: 0] round_bias;

            always_comb begin
                extended = $signed({data_in[i][IN_LEN - 1], data_in[i]});
                round_bias = '0;
                rounded = extended;

                if (shift != 0) begin
                    round_bias = {{IN_LEN{1'b0}}, 1'b1} << (shift - 1);
                    if (data_in[i][IN_LEN - 1] == 1'b0)
                        rounded = extended + round_bias;
                    else
                        rounded = extended + round_bias - {{IN_LEN{1'b0}}, 1'b1};;
                end

                shifted = rounded >>> shift;
            end

            always_comb begin
                if (shifted > $signed({1'b0, {(OUT_LEN - 1){1'b1}}}))
                    data_out[i] = $signed({1'b0, {(OUT_LEN - 1){1'b1}}}); // +127
                else if (shifted < $signed({1'b1, {(OUT_LEN - 1){1'b0}}}))
                    data_out[i] = $signed({1'b1, {(OUT_LEN - 1){1'b0}}}); // -128
                else
                    data_out[i] = shifted[OUT_LEN-1:0];
            end
        end
    endgenerate

endmodule
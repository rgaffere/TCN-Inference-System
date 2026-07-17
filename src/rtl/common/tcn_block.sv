// this is the whole tcn body
// key for variable naming: h = hidden layer, i = input layer, o = output_layer
module tcn_block #(
    parameter int IMU_CHANNELS = 6,
    parameter int HIDDEN_CHANNELS = 16,
    parameter int NUM_HIDDEN = 7,
    parameter int W_WIDTH = 8,
    parameter int B_WIDTH = 32,
    parameter int BASE_DILATION = 2,
    parameter int KERNEL_LEN = 3,

    localparam int AXIS_WIDTH = W_WIDTH * IMU_CHANNELS
) (
    input logic clk,
    input logic rst_n,

    input logic s_tvalid,
    input logic [AXIS_WIDTH - 1: 0] s_tdata,
    input logic m_tready,

    // axis output vars
    output logic s_tready,
    output logic [AXIS_WIDTH - 1: 0] m_tdata,
    output logic m_tvalid
);

    logic signed [W_WIDTH - 1: 0] i_input_vals [0: IMU_CHANNELS - 1];
    logic signed [W_WIDTH - 1: 0] o_output_vals [0: IMU_CHANNELS - 1];

    // Input-block control and output
    logic i_valid_in;
    logic i_valid_out;

    // Input block uses channel-mixing convolutions
    // Conv1: IMU_CHANNELS -> HIDDEN_CHANNELS
    logic signed [W_WIDTH - 1: 0] i_weights1 [0: HIDDEN_CHANNELS - 1][0: IMU_CHANNELS - 1][0: KERNEL_LEN - 1];
    // Conv2: HIDDEN_CHANNELS -> HIDDEN_CHANNELS
    logic signed [W_WIDTH - 1: 0] i_weights2 [0: HIDDEN_CHANNELS - 1][0: HIDDEN_CHANNELS - 1][0: KERNEL_LEN - 1];
    // Residual projection: IMU_CHANNELS -> HIDDEN_CHANNELS
    logic signed [W_WIDTH - 1: 0] i_residual_weights [0: HIDDEN_CHANNELS - 1][0: IMU_CHANNELS - 1];
    logic signed [B_WIDTH - 1: 0] i_bias1 [0: HIDDEN_CHANNELS - 1];
    logic signed [B_WIDTH - 1: 0] i_bias2 [0: HIDDEN_CHANNELS - 1];
    logic signed [B_WIDTH - 1: 0] i_residual_bias [0: HIDDEN_CHANNELS - 1];

    logic signed [W_WIDTH - 1: 0] i_output_vals [0: HIDDEN_CHANNELS - 1];


    // Hidden depthwise residual-block signals
    logic h_valid_out [0: NUM_HIDDEN - 1];

    // Each hidden block is depthwise
    // one KERNEL_LEN filter per channel, with no input-channel dimension
    logic signed [W_WIDTH - 1: 0] h_weights1 [0: NUM_HIDDEN - 1][0: HIDDEN_CHANNELS - 1][0: KERNEL_LEN - 1];
    logic signed [W_WIDTH - 1: 0]  h_weights2 [0: NUM_HIDDEN - 1][0: HIDDEN_CHANNELS - 1][0: KERNEL_LEN - 1];
    logic signed [B_WIDTH - 1: 0] h_bias1 [0: NUM_HIDDEN - 1][0: HIDDEN_CHANNELS - 1];
    logic signed [B_WIDTH - 1: 0] h_bias2 [0: NUM_HIDDEN - 1][0: HIDDEN_CHANNELS - 1];
    logic signed [W_WIDTH - 1: 0] h_output_vals [0: NUM_HIDDEN - 1][0: HIDDEN_CHANNELS - 1];


    // Output-block control and channel-mixing projection
    logic o_valid_out;

    // HIDDEN_CHANNELS -> IMU_CHANNELS
    logic signed [W_WIDTH - 1: 0] o_weights [0: IMU_CHANNELS - 1][0: HIDDEN_CHANNELS - 1];
    logic signed [B_WIDTH - 1: 0] o_bias [0: IMU_CHANNELS - 1];

    // IGNORE CUZ WERE DOING SEQUENTIAL NOW
    // AXI4-S input/output control
    // for(genvar i = 0; i < IMU_CHANNELS; i++) begin : gen_axis_channels
    //     assign i_input_vals[i] = s_tdata[i * W_WIDTH +: W_WIDTH];
    //     assign m_tdata[i * W_WIDTH +: W_WIDTH] = o_output_vals[i];
    // end

    // instantiations

    // Network: INPUT -> 7 x HIDDEN -> OUTPUT

    input_block #(
        .IN_CHANNELS(IMU_CHANNELS),
        .OUT_CHANNELS(HIDDEN_CHANNELS),
        .W_BIT_WIDTH(W_WIDTH),
        .B_BIT_WIDTH(B_WIDTH),
        .KERNEL_LEN(KERNEL_LEN)
    ) input_layer (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(i_valid_in),
        .inputVals(i_input_vals),
        .weights1(i_weights1),
        .weights2(i_weights2),
        .residual_weights(i_residual_weights),
        .bias1(i_bias1),
        .bias2(i_bias2),
        .residual_bias(i_residual_bias),
        .outputVals(i_output_vals),
        .valid_out(i_valid_out)
    );

    genvar HRB; // HRB = Hidden Residual Block
    generate
        for(HRB = 0; HRB < NUM_HIDDEN; HRB++) begin : gen_hidden_resi_blocks
            resi_block #(
                .NUM_CHANNELS(HIDDEN_CHANNELS),
                .W_BIT_WIDTH(W_WIDTH),
                .B_BIT_WIDTH(B_WIDTH),
                .KERNEL_LEN(KERNEL_LEN),
                .DILATION(BASE_DILATION ** HRB)
            ) hidden_layer (
                .clk(clk),
                .rst_n(rst_n),
                .valid_in(HRB == 0 ? i_valid_out : h_valid_out[HRB - 1]),
                .inputVals(HRB == 0 ? i_output_vals : h_output_vals[HRB - 1]),
                .weights1(h_weights1[HRB]),
                .weights2(h_weights2[HRB]),
                .bias1(h_bias1[HRB]),
                .bias2(h_bias2[HRB]),
                .outputVals(h_output_vals[HRB]),
                .valid_out(h_valid_out[HRB])
            );
        end
    endgenerate

    output_block #(
        .IN_CHANNELS(HIDDEN_CHANNELS),
        .OUT_CHANNELS(IMU_CHANNELS),
        .W_BIT_WIDTH(W_WIDTH),
        .B_BIT_WIDTH(B_WIDTH)
    ) output_layer (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(h_valid_out[NUM_HIDDEN - 1]),
        .inputVals(h_output_vals[NUM_HIDDEN - 1]),
        .weights(o_weights),
        .bias(o_bias),
        .outputVals(o_output_vals),
        .valid_out(o_valid_out)
    );

    // AXI4-S control from here on out
    always_ff @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            for(int CH = 0; CH < IMU_CHANNELS; CH++) begin
                i_input_vals[CH] <= '0;
            end
            m_tdata <= '0;
            i_valid_in <= 1'b0;
        end else begin
            // DOING THE DATA BUS VALUES...
            // assign input vals and start computation
            if(s_tready && s_tvalid) begin
                for(int i = 0; i < IMU_CHANNELS; i++) begin
                    i_input_vals[i] <= $signed(s_tdata[(i + 1) * W_WIDTH - 1 -: W_WIDTH]);
                end
                i_valid_in <= 1'b1;
            end else begin
                i_valid_in <= 1'b0;
            end
            // assign output vals
            if(o_valid_out) begin
                for(int j = 0; j < IMU_CHANNELS; j++) begin
                    m_tdata[(j + 1) * W_WIDTH - 1 -: W_WIDTH] <= $unsigned(o_output_vals[j]);
                end
            end
        end
    end

    // s_tready control
    always_ff @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            s_tready <= 1'b1;
        end else if(m_tvalid && m_tready) begin
            s_tready <= 1'b1;
        end else if(s_tvalid && s_tready) begin
            s_tready <= 1'b0;
        end
    end

    // m_tvalid control
    always_ff @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            m_tvalid <= 1'b0;
        end else if(o_valid_out) begin
            m_tvalid <= 1'b1;
        end else if(m_tvalid && m_tready) begin
            m_tvalid <= 1'b0;
        end
    end

endmodule
module tcn_axis_wrapper #(
    parameter int NUM_CHANNELS = 16,
    parameter int W_WIDTH = 8,
    parameter int B_WIDTH = 32,
    parameter int KERNEL_LEN = 3,
    parameter int DILATION = 1,
    parameter int DEPTH = 512,

    localparam int AXIS_WIDTH = NUM_CHANNELS * W_WIDTH
) (
    input logic clk,
    input logic rst_n,

    // axis input vars
    input logic s_tvalid,
    input logic [AXIS_WIDTH - 1: 0] s_tdata,
    input logic m_tready,
    // these are flashed in on init
    input var logic signed [W_WIDTH - 1: 0] weights1 [0: NUM_CHANNELS - 1][0: KERNEL_LEN - 1],
    input var logic signed [W_WIDTH - 1: 0] weights2 [0: NUM_CHANNELS - 1][0: KERNEL_LEN - 1],
    input var logic signed [B_WIDTH - 1: 0] bias1 [0: NUM_CHANNELS - 1],
    input var logic signed [B_WIDTH - 1: 0] bias2 [0: NUM_CHANNELS - 1],

    // axis output vars
    output logic s_tready,
    output logic [AXIS_WIDTH - 1: 0] m_tdata,
    output logic m_tvalid
);
    // vars for dut
    var logic signed [W_WIDTH - 1: 0] inputVals [0: NUM_CHANNELS - 1];
    var logic signed [W_WIDTH - 1: 0] outputVals [0: NUM_CHANNELS - 1];

    logic valid_in, valid_out;

    // Instantiate DUT
    resi_block #(
        .NUM_CHANNELS(NUM_CHANNELS),
        .W_BIT_WIDTH(W_WIDTH),
        .B_BIT_WIDTH(B_WIDTH),
        .KERNEL_LEN(KERNEL_LEN),
        .DILATION(DILATION),
        .DEPTH(DEPTH)
    ) dut_rb (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .inputVals(inputVals),
        .weights1(weights1),
        .weights2(weights2),
        .bias1(bias1),
        .bias2(bias2),
        .outputVals(outputVals),
        .valid_out(valid_out)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            for(int CH = 0; CH < NUM_CHANNELS; CH++) begin
                inputVals[CH] <= '0;
            end
            m_tdata <= '0;
            valid_in <= 1'b0;
        end else begin
            // DOING THE DATA BUS VALUES...
            // assign input vals and start computation
            if(s_tready && s_tvalid) begin
                for(int i = 0; i < NUM_CHANNELS; i++) begin
                    inputVals[i] <= $signed(s_tdata[(i + 1) * W_WIDTH - 1 -: W_WIDTH]);
                end
                valid_in <= 1'b1;
            end else begin
                valid_in <= 1'b0;
            end
            // assign output vals
            if(valid_out) begin
                for(int j = 0; j < NUM_CHANNELS; j++) begin
                    m_tdata[(j + 1) * W_WIDTH - 1 -: W_WIDTH] <= $unsigned(outputVals[j]);
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
        end else if(valid_out) begin
            m_tvalid <= 1'b1;
        end else if(m_tvalid && m_tready) begin
            m_tvalid <= 1'b0;
        end
    end
endmodule
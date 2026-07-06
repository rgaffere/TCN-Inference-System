module detect_anomaly #(
  parameter int THRESHOLD = 25,
  parameter int DATA_WIDTH = 8,
  // 6 channels from the IMU
  parameter int NUM_CHANNELS = 6,
  localparam int NUM_STAGES = 3
) (
  input logic clk,
  input logic rst_n,
  input logic valid_in,
  input logic signed [DATA_WIDTH - 1: 0] latestIMUInput [0: NUM_CHANNELS - 1],
  input logic signed [DATA_WIDTH - 1: 0] lastNNOutput [0: NUM_CHANNELS - 1],
  output logic anomalyDetected [0: NUM_CHANNELS - 1],
  output logic valid_out
);
  logic [NUM_STAGES - 1: 0] s;
  logic signed [DATA_WIDTH: 0] diff [0: NUM_CHANNELS - 1];
  logic [2 * (DATA_WIDTH + 1) - 1: 0] errors [0: NUM_CHANNELS - 1];

  assign valid_out = s[NUM_STAGES - 1];

  
  for(genvar i = 0; i < NUM_CHANNELS; i++) begin
    always_ff @(posedge clk or negedge rst_n) begin
      if(!rst_n) begin
        errors[i] <= '0;
        anomalyDetected[i] <= 1'b0;
        diff[i] <= '0;
      end else begin
        // im sign extending the diff here
        if(valid_in) begin
          diff[i] <= $signed({latestIMUInput[i][DATA_WIDTH-1], latestIMUInput[i]}) - $signed({lastNNOutput[i][DATA_WIDTH-1], lastNNOutput[i]});
        end
        
        if(s[0]) begin
          errors[i] <= diff[i] * diff[i];
        end
        
        if(s[1]) begin
          if(errors[i] > THRESHOLD) begin
            anomalyDetected[i] <= 1'b1;
          end else begin
            anomalyDetected[i] <= 1'b0;
          end
        end
      end
    end
  end

  //stage control
  always_ff @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
      s <= '0;
    end else begin
      s <= {s[NUM_STAGES - 2: 0], valid_in};
    end
  end
endmodule

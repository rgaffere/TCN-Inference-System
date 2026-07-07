`timescale 1ns/1ps

module detect_anomaly_tb;

  localparam int THRESHOLD    = 25;
  localparam int DATA_WIDTH   = 8;
  localparam int NUM_CHANNELS = 6;
  localparam int NUM_TESTS    = 1000;

  logic clk;
  logic rst_n;
  logic valid_in;

  logic signed [DATA_WIDTH - 1: 0] latestIMUInput [0: NUM_CHANNELS - 1];
  logic signed [DATA_WIDTH - 1: 0] lastNNOutput   [0: NUM_CHANNELS - 1];

  logic anomalyDetected [0: NUM_CHANNELS - 1];
  logic valid_out;

  logic [NUM_CHANNELS - 1: 0] expected_q[$];

  int scoreboard_errors;
  int assertion_errors;
  int transactions_checked;


  detect_anomaly #(
    .THRESHOLD    (THRESHOLD),
    .DATA_WIDTH   (DATA_WIDTH),
    .NUM_CHANNELS (NUM_CHANNELS)
  ) dut (
    .clk             (clk),
    .rst_n           (rst_n),
    .valid_in        (valid_in),
    .latestIMUInput  (latestIMUInput),
    .lastNNOutput    (lastNNOutput),
    .anomalyDetected (anomalyDetected),
    .valid_out       (valid_out)
  );


  always #5 clk = ~clk;


  task automatic drive_random();
    logic [NUM_CHANNELS - 1: 0] expected;
    int signed diff;
    int error;

    valid_in = ($urandom_range(0, 99) < 75);

    for(int i = 0; i < NUM_CHANNELS; i++) begin
      latestIMUInput[i] = $signed($urandom());
      lastNNOutput[i]   = $signed($urandom());

      diff  = $signed(latestIMUInput[i])
            - $signed(lastNNOutput[i]);

      error = diff * diff;

      expected[i] = (error > THRESHOLD);
    end

    if(valid_in)
      expected_q.push_back(expected);
  endtask


  task automatic check_output();
    logic [NUM_CHANNELS - 1: 0] expected;

    if(rst_n && valid_out) begin
      if(expected_q.size() == 0) begin
        scoreboard_errors++;
        $error("valid_out asserted with empty expected queue");
      end else begin
        expected = expected_q.pop_front();

        for(int i = 0; i < NUM_CHANNELS; i++) begin
          if(anomalyDetected[i] !== expected[i]) begin
            scoreboard_errors++;

            $error(
              "Channel %0d mismatch: IMU=%0d NN=%0d expected=%0b actual=%0b",
              i,
              latestIMUInput[i],
              lastNNOutput[i],
              expected[i],
              anomalyDetected[i]
            );
          end
        end

        transactions_checked++;
      end
    end
  endtask


  a_valid_latency:
    assert property (
      @(negedge clk)
      disable iff (!rst_n)
      valid_in |-> ##2 valid_out
    )
    else begin
      assertion_errors++;
      $error("valid_out was not asserted 2 cycles after valid_in");
    end


  a_no_spurious_valid:
    assert property (
      @(negedge clk)
      disable iff (!rst_n)
      !valid_in |-> ##2 !valid_out
    )
    else begin
      assertion_errors++;
      $error("valid_out asserted without valid_in 2 cycles earlier");
    end


  a_valid_known:
    assert property (
      @(negedge clk)
      disable iff (!rst_n)
      !$isunknown(valid_out)
    )
    else begin
      assertion_errors++;
      $error("valid_out contains X or Z");
    end


  generate
    for(genvar i = 0; i < NUM_CHANNELS; i++) begin : gen_output_assertions

      a_anomaly_known:
        assert property (
          @(negedge clk)
          disable iff (!rst_n)
          valid_out |-> !$isunknown(anomalyDetected[i])
        )
        else begin
          assertion_errors++;
          $error("anomalyDetected[%0d] contains X or Z", i);
        end

    end
  endgenerate


  initial begin
    clk                  = 1'b0;
    rst_n                = 1'b0;
    valid_in             = 1'b0;
    scoreboard_errors    = 0;
    assertion_errors     = 0;
    transactions_checked = 0;

    for(int i = 0; i < NUM_CHANNELS; i++) begin
      latestIMUInput[i] = '0;
      lastNNOutput[i]   = '0;
    end

    repeat(3) @(negedge clk);

    rst_n = 1'b1;


    for(int test_num = 0; test_num < NUM_TESTS; test_num++) begin
      check_output();
      drive_random();

      @(negedge clk);
    end


    valid_in = 1'b0;

    repeat(4) begin
      check_output();
      @(negedge clk);
    end

    check_output();

    #1;

    if(expected_q.size() != 0) begin
      scoreboard_errors++;
      $error(
        "%0d expected transactions remain in queue",
        expected_q.size()
      );
    end


    $display("");
    $display("========================================");
    $display("Transactions checked : %0d", transactions_checked);
    $display("Scoreboard errors     : %0d", scoreboard_errors);
    $display("Assertion errors      : %0d", assertion_errors);
    $display("========================================");

    if(scoreboard_errors == 0 && assertion_errors == 0)
      $display("TEST PASSED");
    else
      $display("TEST FAILED");

    $finish;
  end

endmodule
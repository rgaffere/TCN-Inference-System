module sky130_sram_2kbyte_1rw1r_32x512_8 (
`ifdef USE_POWER_PINS
    input  logic        vccd1,
    input  logic        vssd1,
`endif

    input  logic        clk0,
    input  logic        csb0,
    input  logic        web0,
    input  logic [3:0]  wmask0,
    input  logic [8:0]  addr0,
    input  logic [31:0] din0,
    output logic [31:0] dout0,

    input  logic        clk1,
    input  logic        csb1,
    input  logic [8:0]  addr1,
    output logic [31:0] dout1
);
endmodule

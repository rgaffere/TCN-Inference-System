/////////////////////////////////////////////////////////////
// Created by: Synopsys DC Ultra(TM) in wire load mode
// Version   : X-2025.06-SP4
// Date      : Mon May 25 20:21:55 2026
/////////////////////////////////////////////////////////////


module ring ( clk, rst_n, write_en, data_in, read_offset, data_out );
  input [7:0] data_in;
  input [8:0] read_offset;
  output [7:0] data_out;
  input clk, rst_n, write_en;
  wire   valid_read, valid_read_q, n10, n11, n12, n13, n14, n15, n16, n17, n18,
         n19, n20, n21, n22, n23, n24, n25, n26, n27, n28, n29, n30, n31, n8,
         n9, n32, n34, n35, n36, n37, n38, n39, n40, n41, n42, n43, n44, n45,
         n46, n47, n48, n49, n50, n51, n52, n53, n54, n55, n56, n58, n59, n60,
         n61, n62, n63, n64, n65, n66, n67, n68, n69, n70, n71, n72, n73, n74,
         n75, n76, n77, n78, n79, n80, n81, n82, n83, n84, n85, n86, n87, n88,
         n89, n90, n91, n92, n93, n94, n95, n96, n97, n98, n99, n100, n101,
         n102, n103, n104, n105;
  wire   [8:0] head;
  wire   [8:0] read_addr;
  wire   [9:0] warmup_count;
  wire   [31:0] sram_dout1;
  wire   SYNOPSYS_UNCONNECTED__0, SYNOPSYS_UNCONNECTED__1, 
        SYNOPSYS_UNCONNECTED__2, SYNOPSYS_UNCONNECTED__3, 
        SYNOPSYS_UNCONNECTED__4, SYNOPSYS_UNCONNECTED__5, 
        SYNOPSYS_UNCONNECTED__6, SYNOPSYS_UNCONNECTED__7, 
        SYNOPSYS_UNCONNECTED__8, SYNOPSYS_UNCONNECTED__9, 
        SYNOPSYS_UNCONNECTED__10, SYNOPSYS_UNCONNECTED__11, 
        SYNOPSYS_UNCONNECTED__12, SYNOPSYS_UNCONNECTED__13, 
        SYNOPSYS_UNCONNECTED__14, SYNOPSYS_UNCONNECTED__15, 
        SYNOPSYS_UNCONNECTED__16, SYNOPSYS_UNCONNECTED__17, 
        SYNOPSYS_UNCONNECTED__18, SYNOPSYS_UNCONNECTED__19, 
        SYNOPSYS_UNCONNECTED__20, SYNOPSYS_UNCONNECTED__21, 
        SYNOPSYS_UNCONNECTED__22, SYNOPSYS_UNCONNECTED__23;

  sky130_sram_2kbyte_1rw1r_32x512_8 u_ring_sram ( .din0({n31, n31, n31, n31, 
        n31, n31, n31, n31, n31, n31, n31, n31, n31, n31, n31, n31, n31, n31, 
        n31, n31, n31, n31, n31, n31, data_in}), .addr0({n47, n50, n43, n52, 
        n44, n54, n38, n42, n48}), .wmask0({n31, n31, n31, n20}), .dout1({
        SYNOPSYS_UNCONNECTED__0, SYNOPSYS_UNCONNECTED__1, 
        SYNOPSYS_UNCONNECTED__2, SYNOPSYS_UNCONNECTED__3, 
        SYNOPSYS_UNCONNECTED__4, SYNOPSYS_UNCONNECTED__5, 
        SYNOPSYS_UNCONNECTED__6, SYNOPSYS_UNCONNECTED__7, 
        SYNOPSYS_UNCONNECTED__8, SYNOPSYS_UNCONNECTED__9, 
        SYNOPSYS_UNCONNECTED__10, SYNOPSYS_UNCONNECTED__11, 
        SYNOPSYS_UNCONNECTED__12, SYNOPSYS_UNCONNECTED__13, 
        SYNOPSYS_UNCONNECTED__14, SYNOPSYS_UNCONNECTED__15, 
        SYNOPSYS_UNCONNECTED__16, SYNOPSYS_UNCONNECTED__17, 
        SYNOPSYS_UNCONNECTED__18, SYNOPSYS_UNCONNECTED__19, 
        SYNOPSYS_UNCONNECTED__20, SYNOPSYS_UNCONNECTED__21, 
        SYNOPSYS_UNCONNECTED__22, SYNOPSYS_UNCONNECTED__23, sram_dout1[7:0]}), 
        .addr1({n35, read_addr[7], n40, read_addr[5], n39, read_addr[3], n45, 
        read_addr[1], n36}), .csb0(n30), .web0(n30), .clk0(clk), .csb1(n31), 
        .clk1(clk) );
  sky130_fd_sc_hd__dfrtp_1 \warmup_count_reg[9]  ( .D(n28), .CLK(clk), 
        .RESET_B(rst_n), .Q(warmup_count[9]) );
  sky130_fd_sc_hd__dfrtp_1 \warmup_count_reg[8]  ( .D(n19), .CLK(clk), 
        .RESET_B(rst_n), .Q(warmup_count[8]) );
  sky130_fd_sc_hd__dfrtp_1 \warmup_count_reg[7]  ( .D(n21), .CLK(clk), 
        .RESET_B(rst_n), .Q(warmup_count[7]) );
  sky130_fd_sc_hd__dfrtp_1 \warmup_count_reg[6]  ( .D(n22), .CLK(clk), 
        .RESET_B(rst_n), .Q(warmup_count[6]) );
  sky130_fd_sc_hd__dfrtp_1 \warmup_count_reg[5]  ( .D(n23), .CLK(clk), 
        .RESET_B(rst_n), .Q(warmup_count[5]) );
  sky130_fd_sc_hd__dfrtp_1 \warmup_count_reg[4]  ( .D(n24), .CLK(clk), 
        .RESET_B(rst_n), .Q(warmup_count[4]) );
  sky130_fd_sc_hd__dfrtp_1 \warmup_count_reg[3]  ( .D(n25), .CLK(clk), 
        .RESET_B(rst_n), .Q(warmup_count[3]) );
  sky130_fd_sc_hd__dfrtp_1 \warmup_count_reg[2]  ( .D(n26), .CLK(clk), 
        .RESET_B(rst_n), .Q(warmup_count[2]) );
  sky130_fd_sc_hd__dfrtp_1 \warmup_count_reg[1]  ( .D(n27), .CLK(clk), 
        .RESET_B(rst_n), .Q(warmup_count[1]) );
  sky130_fd_sc_hd__dfrtp_1 \warmup_count_reg[0]  ( .D(n29), .CLK(clk), 
        .RESET_B(rst_n), .Q(warmup_count[0]) );
  sky130_fd_sc_hd__dfrtp_1 \head_reg[7]  ( .D(n17), .CLK(clk), .RESET_B(rst_n), 
        .Q(head[7]) );
  sky130_fd_sc_hd__dfrtp_1 \head_reg[6]  ( .D(n16), .CLK(clk), .RESET_B(rst_n), 
        .Q(head[6]) );
  sky130_fd_sc_hd__dfrtp_1 \head_reg[5]  ( .D(n15), .CLK(clk), .RESET_B(rst_n), 
        .Q(head[5]) );
  sky130_fd_sc_hd__dfrtp_1 \head_reg[4]  ( .D(n14), .CLK(clk), .RESET_B(rst_n), 
        .Q(head[4]) );
  sky130_fd_sc_hd__dfrtp_1 \head_reg[3]  ( .D(n13), .CLK(clk), .RESET_B(rst_n), 
        .Q(head[3]) );
  sky130_fd_sc_hd__dfrtp_1 \head_reg[2]  ( .D(n12), .CLK(clk), .RESET_B(rst_n), 
        .Q(head[2]) );
  sky130_fd_sc_hd__dfrtp_1 \head_reg[1]  ( .D(n11), .CLK(clk), .RESET_B(rst_n), 
        .Q(head[1]) );
  sky130_fd_sc_hd__dfrtp_4 valid_read_q_reg ( .D(valid_read), .CLK(clk), 
        .RESET_B(rst_n), .Q(valid_read_q) );
  sky130_fd_sc_hd__dfrtp_1 \head_reg[0]  ( .D(n10), .CLK(clk), .RESET_B(rst_n), 
        .Q(head[0]) );
  sky130_fd_sc_hd__dfrtp_1 \head_reg[8]  ( .D(n18), .CLK(clk), .RESET_B(rst_n), 
        .Q(head[8]) );
  sky130_fd_sc_hd__inv_2 U3 ( .A(head[7]), .Y(n49) );
  sky130_fd_sc_hd__inv_2 U4 ( .A(head[5]), .Y(n51) );
  sky130_fd_sc_hd__inv_2 U5 ( .A(head[3]), .Y(n53) );
  sky130_fd_sc_hd__fa_1 U6 ( .A(read_offset[1]), .B(n65), .CIN(n64), .COUT(n66), .SUM(n62) );
  sky130_fd_sc_hd__inv_2 U7 ( .A(head[2]), .Y(n37) );
  sky130_fd_sc_hd__inv_1 U8 ( .A(head[0]), .Y(n98) );
  sky130_fd_sc_hd__inv_1 U9 ( .A(head[6]), .Y(n89) );
  sky130_fd_sc_hd__inv_1 U10 ( .A(head[4]), .Y(n92) );
  sky130_fd_sc_hd__inv_8 U11 ( .A(n49), .Y(n50) );
  sky130_fd_sc_hd__inv_8 U12 ( .A(n51), .Y(n52) );
  sky130_fd_sc_hd__inv_8 U13 ( .A(n53), .Y(n54) );
  sky130_fd_sc_hd__o21a_1 U14 ( .A1(n70), .A2(n89), .B1(n71), .X(n8) );
  sky130_fd_sc_hd__o21a_1 U15 ( .A1(n68), .A2(n92), .B1(n69), .X(n9) );
  sky130_fd_sc_hd__o22a_1 U16 ( .A1(head[0]), .A2(read_offset[0]), .B1(n98), 
        .B2(n63), .X(n32) );
  sky130_fd_sc_hd__and2_1 U17 ( .A(sram_dout1[6]), .B(valid_read_q), .X(
        data_out[6]) );
  sky130_fd_sc_hd__a21o_4 U18 ( .A1(n50), .A2(n71), .B1(n72), .X(read_addr[7])
         );
  sky130_fd_sc_hd__a21o_4 U19 ( .A1(n52), .A2(n69), .B1(n70), .X(read_addr[5])
         );
  sky130_fd_sc_hd__a21o_4 U20 ( .A1(n54), .A2(n67), .B1(n68), .X(read_addr[3])
         );
  sky130_fd_sc_hd__inv_4 U21 ( .A(n62), .Y(read_addr[1]) );
  sky130_fd_sc_hd__clkinv_1 U22 ( .A(read_addr[8]), .Y(n34) );
  sky130_fd_sc_hd__inv_4 U23 ( .A(n34), .Y(n35) );
  sky130_fd_sc_hd__xor2_1 U24 ( .A(head[8]), .B(n72), .X(read_addr[8]) );
  sky130_fd_sc_hd__inv_4 U25 ( .A(n32), .Y(n36) );
  sky130_fd_sc_hd__inv_4 U26 ( .A(n37), .Y(n38) );
  sky130_fd_sc_hd__inv_4 U27 ( .A(n9), .Y(n39) );
  sky130_fd_sc_hd__inv_4 U28 ( .A(n8), .Y(n40) );
  sky130_fd_sc_hd__clkinv_1 U29 ( .A(head[1]), .Y(n41) );
  sky130_fd_sc_hd__inv_4 U30 ( .A(n41), .Y(n42) );
  sky130_fd_sc_hd__inv_4 U31 ( .A(n89), .Y(n43) );
  sky130_fd_sc_hd__inv_4 U32 ( .A(n92), .Y(n44) );
  sky130_fd_sc_hd__buf_4 U33 ( .A(read_addr[2]), .X(n45) );
  sky130_fd_sc_hd__clkinv_1 U34 ( .A(head[8]), .Y(n46) );
  sky130_fd_sc_hd__inv_4 U35 ( .A(n46), .Y(n47) );
  sky130_fd_sc_hd__inv_4 U36 ( .A(n98), .Y(n48) );
  sky130_fd_sc_hd__o21a_1 U37 ( .A1(n88), .A2(n50), .B1(n87), .X(n17) );
  sky130_fd_sc_hd__o21a_1 U38 ( .A1(n91), .A2(n52), .B1(n90), .X(n15) );
  sky130_fd_sc_hd__o21a_1 U39 ( .A1(n94), .A2(n54), .B1(n93), .X(n13) );
  sky130_fd_sc_hd__nand2_2 U40 ( .A(sram_dout1[7]), .B(valid_read_q), .Y(n55)
         );
  sky130_fd_sc_hd__inv_4 U41 ( .A(n55), .Y(data_out[7]) );
  sky130_fd_sc_hd__inv_2 U42 ( .A(n56), .Y(data_out[2]) );
  sky130_fd_sc_hd__nand2_1 U43 ( .A(sram_dout1[2]), .B(valid_read_q), .Y(n56)
         );
  sky130_fd_sc_hd__nand2_1 U44 ( .A(sram_dout1[4]), .B(valid_read_q), .Y(n61)
         );
  sky130_fd_sc_hd__and2_0 U45 ( .A(sram_dout1[0]), .B(valid_read_q), .X(
        data_out[0]) );
  sky130_fd_sc_hd__conb_1 U46 ( .LO(n31), .HI(n20) );
  sky130_fd_sc_hd__nand2_2 U47 ( .A(sram_dout1[3]), .B(valid_read_q), .Y(n58)
         );
  sky130_fd_sc_hd__inv_4 U48 ( .A(n58), .Y(data_out[3]) );
  sky130_fd_sc_hd__nand2_2 U49 ( .A(sram_dout1[1]), .B(valid_read_q), .Y(n59)
         );
  sky130_fd_sc_hd__inv_4 U50 ( .A(n59), .Y(data_out[1]) );
  sky130_fd_sc_hd__nand2_1 U51 ( .A(sram_dout1[5]), .B(valid_read_q), .Y(n60)
         );
  sky130_fd_sc_hd__inv_2 U52 ( .A(n60), .Y(data_out[5]) );
  sky130_fd_sc_hd__clkinv_1 U53 ( .A(write_en), .Y(n30) );
  sky130_fd_sc_hd__clkinv_1 U54 ( .A(n61), .Y(data_out[4]) );
  sky130_fd_sc_hd__clkinv_1 U55 ( .A(head[1]), .Y(n65) );
  sky130_fd_sc_hd__clkinv_1 U56 ( .A(read_offset[0]), .Y(n63) );
  sky130_fd_sc_hd__nand2_1 U57 ( .A(head[0]), .B(n63), .Y(n64) );
  sky130_fd_sc_hd__clkinv_1 U58 ( .A(n38), .Y(n95) );
  sky130_fd_sc_hd__nand2_1 U59 ( .A(n66), .B(n95), .Y(n67) );
  sky130_fd_sc_hd__o21ai_1 U60 ( .A1(n66), .A2(n95), .B1(n67), .Y(read_addr[2]) );
  sky130_fd_sc_hd__nor2_1 U61 ( .A(head[3]), .B(n67), .Y(n68) );
  sky130_fd_sc_hd__nand2_1 U62 ( .A(n68), .B(n92), .Y(n69) );
  sky130_fd_sc_hd__nor2_1 U63 ( .A(head[5]), .B(n69), .Y(n70) );
  sky130_fd_sc_hd__nand2_1 U64 ( .A(n70), .B(n89), .Y(n71) );
  sky130_fd_sc_hd__nor2_1 U65 ( .A(head[7]), .B(n71), .Y(n72) );
  sky130_fd_sc_hd__clkinv_1 U66 ( .A(warmup_count[0]), .Y(n99) );
  sky130_fd_sc_hd__clkinv_1 U67 ( .A(warmup_count[9]), .Y(n100) );
  sky130_fd_sc_hd__nand2_1 U68 ( .A(write_en), .B(n100), .Y(n73) );
  sky130_fd_sc_hd__nor3_1 U69 ( .A(warmup_count[9]), .B(n99), .C(n30), .Y(n76)
         );
  sky130_fd_sc_hd__a21oi_1 U70 ( .A1(n99), .A2(n73), .B1(n76), .Y(n29) );
  sky130_fd_sc_hd__and3_1 U71 ( .A(warmup_count[0]), .B(warmup_count[3]), .C(
        warmup_count[2]), .X(n74) );
  sky130_fd_sc_hd__nand3_1 U72 ( .A(warmup_count[4]), .B(warmup_count[1]), .C(
        n74), .Y(n81) );
  sky130_fd_sc_hd__nand4_1 U73 ( .A(warmup_count[8]), .B(warmup_count[7]), .C(
        warmup_count[6]), .D(warmup_count[5]), .Y(n75) );
  sky130_fd_sc_hd__o31ai_1 U74 ( .A1(n81), .A2(n30), .A3(n75), .B1(n100), .Y(
        n28) );
  sky130_fd_sc_hd__nand2_1 U75 ( .A(warmup_count[1]), .B(n76), .Y(n77) );
  sky130_fd_sc_hd__o21a_1 U76 ( .A1(warmup_count[1]), .A2(n76), .B1(n77), .X(
        n27) );
  sky130_fd_sc_hd__clkinv_1 U77 ( .A(warmup_count[2]), .Y(n78) );
  sky130_fd_sc_hd__nor2_1 U78 ( .A(n78), .B(n77), .Y(n79) );
  sky130_fd_sc_hd__a21oi_1 U79 ( .A1(n78), .A2(n77), .B1(n79), .Y(n26) );
  sky130_fd_sc_hd__nand2_1 U80 ( .A(warmup_count[3]), .B(n79), .Y(n80) );
  sky130_fd_sc_hd__o21a_1 U81 ( .A1(warmup_count[3]), .A2(n79), .B1(n80), .X(
        n25) );
  sky130_fd_sc_hd__xnor2_1 U82 ( .A(warmup_count[4]), .B(n80), .Y(n24) );
  sky130_fd_sc_hd__nor3_1 U83 ( .A(warmup_count[9]), .B(n81), .C(n30), .Y(n82)
         );
  sky130_fd_sc_hd__nand2_1 U84 ( .A(warmup_count[5]), .B(n82), .Y(n83) );
  sky130_fd_sc_hd__o21a_1 U85 ( .A1(warmup_count[5]), .A2(n82), .B1(n83), .X(
        n23) );
  sky130_fd_sc_hd__clkinv_1 U86 ( .A(warmup_count[6]), .Y(n84) );
  sky130_fd_sc_hd__nor2_1 U87 ( .A(n84), .B(n83), .Y(n85) );
  sky130_fd_sc_hd__a21oi_1 U88 ( .A1(n84), .A2(n83), .B1(n85), .Y(n22) );
  sky130_fd_sc_hd__nand2_1 U89 ( .A(warmup_count[7]), .B(n85), .Y(n86) );
  sky130_fd_sc_hd__o21a_1 U90 ( .A1(warmup_count[7]), .A2(n85), .B1(n86), .X(
        n21) );
  sky130_fd_sc_hd__xnor2_1 U91 ( .A(warmup_count[8]), .B(n86), .Y(n19) );
  sky130_fd_sc_hd__nand3_1 U92 ( .A(write_en), .B(head[0]), .C(head[1]), .Y(
        n96) );
  sky130_fd_sc_hd__nor2_1 U93 ( .A(n96), .B(n95), .Y(n94) );
  sky130_fd_sc_hd__nand2_1 U94 ( .A(n94), .B(n54), .Y(n93) );
  sky130_fd_sc_hd__nor2_1 U95 ( .A(n93), .B(n92), .Y(n91) );
  sky130_fd_sc_hd__nand2_1 U96 ( .A(n91), .B(n52), .Y(n90) );
  sky130_fd_sc_hd__nor2_1 U97 ( .A(n90), .B(n89), .Y(n88) );
  sky130_fd_sc_hd__nand2_1 U98 ( .A(n88), .B(n50), .Y(n87) );
  sky130_fd_sc_hd__xnor2_1 U99 ( .A(head[8]), .B(n87), .Y(n18) );
  sky130_fd_sc_hd__a21oi_1 U100 ( .A1(n90), .A2(n89), .B1(n88), .Y(n16) );
  sky130_fd_sc_hd__a21oi_1 U101 ( .A1(n93), .A2(n92), .B1(n91), .Y(n14) );
  sky130_fd_sc_hd__a21oi_1 U102 ( .A1(n96), .A2(n95), .B1(n94), .Y(n12) );
  sky130_fd_sc_hd__nor2_1 U103 ( .A(n30), .B(n98), .Y(n97) );
  sky130_fd_sc_hd__o21a_1 U104 ( .A1(n97), .A2(head[1]), .B1(n96), .X(n11) );
  sky130_fd_sc_hd__a21oi_1 U105 ( .A1(n30), .A2(n98), .B1(n97), .Y(n10) );
  sky130_fd_sc_hd__clkinv_1 U106 ( .A(warmup_count[1]), .Y(n101) );
  sky130_fd_sc_hd__a21o_1 U107 ( .A1(n101), .A2(read_offset[1]), .B1(n99), .X(
        n105) );
  sky130_fd_sc_hd__nor4_1 U108 ( .A(warmup_count[8]), .B(warmup_count[5]), .C(
        warmup_count[4]), .D(warmup_count[2]), .Y(n104) );
  sky130_fd_sc_hd__o21ai_1 U109 ( .A1(read_offset[1]), .A2(n101), .B1(n100), 
        .Y(n102) );
  sky130_fd_sc_hd__nor4_1 U110 ( .A(warmup_count[7]), .B(warmup_count[6]), .C(
        warmup_count[3]), .D(n102), .Y(n103) );
  sky130_fd_sc_hd__o211ai_1 U111 ( .A1(read_offset[0]), .A2(n105), .B1(n104), 
        .C1(n103), .Y(valid_read) );
endmodule


module conv ( clk, rst_n, valid_in, .weights({\weights[0][7] , \weights[0][6] , 
        \weights[0][5] , \weights[0][4] , \weights[0][3] , \weights[0][2] , 
        \weights[0][1] , \weights[0][0] , \weights[1][7] , \weights[1][6] , 
        \weights[1][5] , \weights[1][4] , \weights[1][3] , \weights[1][2] , 
        \weights[1][1] , \weights[1][0] , \weights[2][7] , \weights[2][6] , 
        \weights[2][5] , \weights[2][4] , \weights[2][3] , \weights[2][2] , 
        \weights[2][1] , \weights[2][0] }), bias, valid_out );
  input [31:0] bias;
  input clk, rst_n, valid_in, \weights[0][7] , \weights[0][6] ,
         \weights[0][5] , \weights[0][4] , \weights[0][3] , \weights[0][2] ,
         \weights[0][1] , \weights[0][0] , \weights[1][7] , \weights[1][6] ,
         \weights[1][5] , \weights[1][4] , \weights[1][3] , \weights[1][2] ,
         \weights[1][1] , \weights[1][0] , \weights[2][7] , \weights[2][6] ,
         \weights[2][5] , \weights[2][4] , \weights[2][3] , \weights[2][2] ,
         \weights[2][1] , \weights[2][0] ;
  output valid_out;
  wire   s1, s2, \read_offset[0] , s0, \init_mac/N156 , \init_mac/N155 ,
         \init_mac/N154 , \init_mac/N153 , \init_mac/N152 , \init_mac/N151 ,
         \init_mac/N150 , \init_mac/N149 , \init_mac/N148 , \init_mac/N147 ,
         \init_mac/N146 , \init_mac/N145 , \init_mac/N144 , \init_mac/N143 ,
         \init_mac/N142 , \init_mac/N141 , \init_mac/N140 , \init_mac/N139 ,
         \init_mac/N138 , \init_mac/N137 , \init_mac/N136 , \init_mac/N135 ,
         \init_mac/N134 , \init_mac/N133 , \init_mac/N132 , \init_mac/N131 ,
         \init_mac/N130 , \init_mac/N129 , \init_mac/N128 , \init_mac/N127 ,
         \init_mac/N126 , \init_mac/N125 , \init_mac/N124 , \init_mac/do_acc ,
         \init_mac/prod_ext_31 , n43, n44, n45, n46, n47, n48, n49, n50, n51,
         n52, n53, n54, n55, n56, n57, n58, n59, n60, n61, n62, n63, n64, n65,
         n66, n67, n68, n69, n79, n80, n81, n82, n83, n84, n85, n86, n87, n88,
         n89, n90, n91, n92, n93, n94, n95, n96, n97, n98, n99, n100, n101,
         n102, n103, n104, n105, n106, n107, n108, n109, n110, n111, n112,
         n113, n114, n115, n116, n117, n118, n119, n120, n121, n122, n123,
         n124, n125, n126, n127, n128, n129, n130, n131, n132, n133, n134,
         n135, n136, n137, n138, n139, n140, n141, n142, n143, n144, n145,
         n146, n147, n148, n149, n150, n151, n152, n153, n154, n155, n156,
         n157, n158, n159, n160, n161, n162, n163, n164, n165, n166, n167,
         n168, n169, n170, n171, n172, n173, n174, n175, n176, n177, n178,
         n179, n180, n181, n182, n183, n184, n185, n186, n187, n188, n189,
         n190, n191, n192, n193, n194, n195, n196, n197, n198, n199, n200,
         n201, n202, n203, n204, n205, n206, n207, n208, n209, n210, n211,
         n212, n213, n214, n215, n216, n217, n218, n219, n220, n221, n222,
         n223, n224, n225, n226, n227, n228, n229, n230, n231, n232, n233,
         n234, n235, n236, n237, n238, n239, n240, n241, n242, n243, n244,
         n245, n246, n247, n248, n249, n250, n251, n252, n253, n254, n255,
         n256, n257, n258, n259, n260, n261, n262, n263, n264, n265, n266,
         n267, n268, n269, n270, n271, n272, n273, n274, n275, n276, n277,
         n278, n279, n280, n281, n282, n283, n284, n285, n286, n287, n288,
         n289, n290, n291, n292, n293, n294, n295, n296, n297, n298, n299,
         n300, n301, n302, n303, n304, n305, n306, n307, n308, n309, n310,
         n311, n312, n313, n314, n315, n316, n317, n318, n319, n320, n321,
         n322, n323, n324, n325, n326, n327, n328, n329, n330, n331, n332,
         n333, n334, n335, n336, n337, n338, n339, n340, n341, n342, n343,
         n344, n345, n346, n347, n348, n349, n350, n351, n352, n353, n354,
         n355, n356, n357, n358, n359, n360, n361, n362, n363, n364, n365,
         n366, n367, n368, n369, n370, n371, n372, n373, n374, n375, n376,
         n377, n378, n379, n380, n381, n382, n383, n384, n385, n386, n387,
         n388, n389, n390, n391, n392, n393, n394, n395, n396, n397, n398,
         n399, n400, n401, n402, n403, n404, n405, n406, n407, n408, n409,
         n410, n411, n412, n413, n414, n415, n416, n417, n418, n419, n420,
         n421, n422, n423, n424, n425, n426, n427, n428, n429, n430, n431,
         n432, n433, n434, n435, n436, n437, n438, n439, n440, n441, n442,
         n443, n444, n445, n446, n447, n448, n449, n450, n451, n452, n453,
         n454, n455, n456, n457, n458, n459, n460, n461, n462, n463, n464,
         n465, n466, n467, n468, n469, n470, n471, n472, n473, n474, n475,
         n476, n477, n478, n479, n480, n481, n482, n483, n484, n485, n486,
         n487, n488, n489, n490, n491, n492, n493, n494, n495, n496, n497,
         n498, n499, n500, n501, n502, n503, n504, n505, n506, n507, n508,
         n509, n510, n511, n512, n513, n514, n515, n516, n517, n518, n519,
         n520, n521, n522, n523, n524, n525, n526, n527, n528, n529, n530,
         n531, n532, n533, n534, n535, n536, n537, n538, n539, n540, n541,
         n542, n543, n544, n545, n546, n547, n548, n549, n550, n551, n552,
         n553, n554, n555, n556, n557, n558, n559, n560, n561, n562, n563,
         n564, n565, n566, n567, n568, n569, n570, n571, n572, n573, n574,
         n575, n576, n577, n578, n579, n580, n581, n582, n583, n584, n585,
         n586, n587, n588, n589, n590, n591, n592, n593, n594, n595, n596,
         n597, n598, n599, n600, n601, n602, n603, n604, n605, n606, n607,
         n608, n609, n610, n611, n612, n613, n614, n615, n616, n617, n618,
         n619, n620, n621, n622, n623, n624, n625, n626, n627, n628, n629,
         n630, n631, n632, n633, n634, n635, n636, n637, n638, n639, n640,
         n641, n642, n643, n644, n645, n646, n647, n648, n649, n650, n651,
         n652, n653, n654, n655, n656, n657, n658, n659, n660, n661, n662,
         n663, n664, n665, n666, n667, n668, n669, n670, n671, n672, n673,
         n674, n675, n676, n677, n678, n679, n680, n681, n682, n683, n684,
         n685, n686, n687, n688, n689, n690, n691, n692, n693, n694, n695,
         n696, n697, n698, n699, net3023, net3024, net3025, net3026, net3027,
         net3028, net3029;
  wire   [7:0] x;
  wire   [31:0] mac_out;
  wire   [7:0] write_in;
  wire   [1:0] tap_idx;
  wire   [1:0] \init_mac/tap_idx ;
  wire   [31:0] \init_mac/acc_reg ;
  wire   [14:0] \init_mac/prod_ext ;

  ring init_ring ( .clk(clk), .rst_n(rst_n), .write_en(valid_out), .data_in(
        write_in), .read_offset({net3023, net3024, net3025, net3026, net3027, 
        net3028, net3029, tap_idx[1], \read_offset[0] }), .data_out(x) );
  sky130_fd_sc_hd__dfrtp_1 \tap_idx_reg[0]  ( .D(n699), .CLK(clk), .RESET_B(
        rst_n), .Q(tap_idx[0]) );
  sky130_fd_sc_hd__dfrtp_1 s0_reg ( .D(valid_in), .CLK(clk), .RESET_B(rst_n), 
        .Q(s0) );
  sky130_fd_sc_hd__dfrtp_1 \init_mac/do_acc_reg  ( .D(s1), .CLK(clk), 
        .RESET_B(rst_n), .Q(\init_mac/do_acc ) );
  sky130_fd_sc_hd__dfrtp_1 \init_mac/acc_reg_reg[0]  ( .D(\init_mac/N125 ), 
        .CLK(clk), .RESET_B(rst_n), .Q(\init_mac/acc_reg [0]) );
  sky130_fd_sc_hd__dfrtp_1 \init_mac/acc_reg_reg[1]  ( .D(\init_mac/N126 ), 
        .CLK(clk), .RESET_B(rst_n), .Q(\init_mac/acc_reg [1]) );
  sky130_fd_sc_hd__dfrtp_1 \init_mac/acc_reg_reg[2]  ( .D(\init_mac/N127 ), 
        .CLK(clk), .RESET_B(rst_n), .Q(\init_mac/acc_reg [2]) );
  sky130_fd_sc_hd__dfrtp_1 \init_mac/acc_reg_reg[3]  ( .D(\init_mac/N128 ), 
        .CLK(clk), .RESET_B(rst_n), .Q(\init_mac/acc_reg [3]) );
  sky130_fd_sc_hd__dfrtp_1 \init_mac/acc_reg_reg[4]  ( .D(\init_mac/N129 ), 
        .CLK(clk), .RESET_B(rst_n), .Q(\init_mac/acc_reg [4]) );
  sky130_fd_sc_hd__dfrtp_1 \init_mac/acc_reg_reg[5]  ( .D(\init_mac/N130 ), 
        .CLK(clk), .RESET_B(rst_n), .Q(\init_mac/acc_reg [5]) );
  sky130_fd_sc_hd__dfrtp_1 \init_mac/acc_reg_reg[6]  ( .D(\init_mac/N131 ), 
        .CLK(clk), .RESET_B(rst_n), .Q(\init_mac/acc_reg [6]) );
  sky130_fd_sc_hd__dfrtp_1 \init_mac/acc_reg_reg[7]  ( .D(\init_mac/N132 ), 
        .CLK(clk), .RESET_B(rst_n), .Q(\init_mac/acc_reg [7]) );
  sky130_fd_sc_hd__dfrtp_1 \init_mac/acc_reg_reg[8]  ( .D(\init_mac/N133 ), 
        .CLK(clk), .RESET_B(rst_n), .Q(\init_mac/acc_reg [8]) );
  sky130_fd_sc_hd__dfrtp_1 \init_mac/acc_reg_reg[9]  ( .D(\init_mac/N134 ), 
        .CLK(clk), .RESET_B(rst_n), .Q(\init_mac/acc_reg [9]) );
  sky130_fd_sc_hd__dfrtp_1 \init_mac/acc_reg_reg[10]  ( .D(\init_mac/N135 ), 
        .CLK(clk), .RESET_B(rst_n), .Q(\init_mac/acc_reg [10]) );
  sky130_fd_sc_hd__dfrtp_1 \init_mac/acc_reg_reg[11]  ( .D(\init_mac/N136 ), 
        .CLK(clk), .RESET_B(rst_n), .Q(\init_mac/acc_reg [11]) );
  sky130_fd_sc_hd__dfrtp_1 \init_mac/acc_reg_reg[12]  ( .D(\init_mac/N137 ), 
        .CLK(clk), .RESET_B(rst_n), .Q(\init_mac/acc_reg [12]) );
  sky130_fd_sc_hd__dfrtp_1 \init_mac/acc_reg_reg[13]  ( .D(\init_mac/N138 ), 
        .CLK(clk), .RESET_B(rst_n), .Q(\init_mac/acc_reg [13]) );
  sky130_fd_sc_hd__dfrtp_1 \init_mac/acc_reg_reg[14]  ( .D(\init_mac/N139 ), 
        .CLK(clk), .RESET_B(rst_n), .Q(\init_mac/acc_reg [14]) );
  sky130_fd_sc_hd__dfrtp_1 \init_mac/acc_reg_reg[15]  ( .D(\init_mac/N140 ), 
        .CLK(clk), .RESET_B(rst_n), .Q(\init_mac/acc_reg [15]) );
  sky130_fd_sc_hd__dfrtp_1 \init_mac/acc_reg_reg[16]  ( .D(\init_mac/N141 ), 
        .CLK(clk), .RESET_B(rst_n), .Q(\init_mac/acc_reg [16]) );
  sky130_fd_sc_hd__dfrtp_1 \init_mac/acc_reg_reg[17]  ( .D(\init_mac/N142 ), 
        .CLK(clk), .RESET_B(rst_n), .Q(\init_mac/acc_reg [17]) );
  sky130_fd_sc_hd__dfrtp_1 \init_mac/acc_reg_reg[18]  ( .D(\init_mac/N143 ), 
        .CLK(clk), .RESET_B(rst_n), .Q(\init_mac/acc_reg [18]) );
  sky130_fd_sc_hd__dfrtp_1 \init_mac/acc_reg_reg[19]  ( .D(\init_mac/N144 ), 
        .CLK(clk), .RESET_B(rst_n), .Q(\init_mac/acc_reg [19]) );
  sky130_fd_sc_hd__dfrtp_1 \init_mac/acc_reg_reg[20]  ( .D(\init_mac/N145 ), 
        .CLK(clk), .RESET_B(rst_n), .Q(\init_mac/acc_reg [20]) );
  sky130_fd_sc_hd__dfrtp_1 \init_mac/acc_reg_reg[21]  ( .D(\init_mac/N146 ), 
        .CLK(clk), .RESET_B(rst_n), .Q(\init_mac/acc_reg [21]) );
  sky130_fd_sc_hd__dfrtp_1 \init_mac/acc_reg_reg[22]  ( .D(\init_mac/N147 ), 
        .CLK(clk), .RESET_B(rst_n), .Q(\init_mac/acc_reg [22]) );
  sky130_fd_sc_hd__dfrtp_1 \init_mac/acc_reg_reg[23]  ( .D(\init_mac/N148 ), 
        .CLK(clk), .RESET_B(rst_n), .Q(\init_mac/acc_reg [23]) );
  sky130_fd_sc_hd__dfrtp_1 \init_mac/acc_reg_reg[24]  ( .D(\init_mac/N149 ), 
        .CLK(clk), .RESET_B(rst_n), .Q(\init_mac/acc_reg [24]) );
  sky130_fd_sc_hd__dfrtp_1 \init_mac/acc_reg_reg[25]  ( .D(\init_mac/N150 ), 
        .CLK(clk), .RESET_B(rst_n), .Q(\init_mac/acc_reg [25]) );
  sky130_fd_sc_hd__dfrtp_1 \init_mac/acc_reg_reg[26]  ( .D(\init_mac/N151 ), 
        .CLK(clk), .RESET_B(rst_n), .Q(\init_mac/acc_reg [26]) );
  sky130_fd_sc_hd__dfrtp_1 \init_mac/acc_reg_reg[27]  ( .D(\init_mac/N152 ), 
        .CLK(clk), .RESET_B(rst_n), .Q(\init_mac/acc_reg [27]) );
  sky130_fd_sc_hd__dfrtp_1 \init_mac/acc_reg_reg[28]  ( .D(\init_mac/N153 ), 
        .CLK(clk), .RESET_B(rst_n), .Q(\init_mac/acc_reg [28]) );
  sky130_fd_sc_hd__dfrtp_1 \init_mac/acc_reg_reg[29]  ( .D(\init_mac/N154 ), 
        .CLK(clk), .RESET_B(rst_n), .Q(\init_mac/acc_reg [29]) );
  sky130_fd_sc_hd__dfrtp_1 \init_mac/acc_reg_reg[30]  ( .D(\init_mac/N155 ), 
        .CLK(clk), .RESET_B(rst_n), .Q(\init_mac/acc_reg [30]) );
  sky130_fd_sc_hd__dfrtp_1 \init_mac/acc_reg_reg[31]  ( .D(\init_mac/N156 ), 
        .CLK(clk), .RESET_B(rst_n), .Q(\init_mac/acc_reg [31]) );
  sky130_fd_sc_hd__dfrtp_1 \init_mac/valid_out_reg  ( .D(\init_mac/N124 ), 
        .CLK(clk), .RESET_B(rst_n), .Q(s2) );
  sky130_fd_sc_hd__dfrtp_1 s3_reg ( .D(s2), .CLK(clk), .RESET_B(rst_n), .Q(
        valid_out) );
  sky130_fd_sc_hd__dfrtp_1 \init_mac/acc_out_reg[8]  ( .D(n68), .CLK(clk), 
        .RESET_B(rst_n), .Q(mac_out[8]) );
  sky130_fd_sc_hd__dfrtp_1 \init_mac/acc_out_reg[9]  ( .D(n67), .CLK(clk), 
        .RESET_B(rst_n), .Q(mac_out[9]) );
  sky130_fd_sc_hd__dfrtp_1 \init_mac/acc_out_reg[10]  ( .D(n66), .CLK(clk), 
        .RESET_B(rst_n), .Q(mac_out[10]) );
  sky130_fd_sc_hd__dfrtp_1 \init_mac/acc_out_reg[11]  ( .D(n65), .CLK(clk), 
        .RESET_B(rst_n), .Q(mac_out[11]) );
  sky130_fd_sc_hd__dfrtp_1 \init_mac/acc_out_reg[12]  ( .D(n64), .CLK(clk), 
        .RESET_B(rst_n), .Q(mac_out[12]) );
  sky130_fd_sc_hd__dfrtp_1 \init_mac/acc_out_reg[13]  ( .D(n63), .CLK(clk), 
        .RESET_B(rst_n), .Q(mac_out[13]) );
  sky130_fd_sc_hd__dfrtp_1 \init_mac/acc_out_reg[14]  ( .D(n62), .CLK(clk), 
        .RESET_B(rst_n), .Q(mac_out[14]) );
  sky130_fd_sc_hd__dfrtp_1 \init_mac/acc_out_reg[15]  ( .D(n61), .CLK(clk), 
        .RESET_B(rst_n), .Q(mac_out[15]) );
  sky130_fd_sc_hd__dfrtp_1 \init_mac/prod_reg_reg[0]  ( .D(n60), .CLK(clk), 
        .RESET_B(rst_n), .Q(\init_mac/prod_ext [0]) );
  sky130_fd_sc_hd__dfrtp_1 \init_mac/prod_reg_reg[1]  ( .D(n59), .CLK(clk), 
        .RESET_B(rst_n), .Q(\init_mac/prod_ext [1]) );
  sky130_fd_sc_hd__dfrtp_1 \init_mac/prod_reg_reg[2]  ( .D(n58), .CLK(clk), 
        .RESET_B(rst_n), .Q(\init_mac/prod_ext [2]) );
  sky130_fd_sc_hd__dfrtp_1 \init_mac/prod_reg_reg[3]  ( .D(n57), .CLK(clk), 
        .RESET_B(rst_n), .Q(\init_mac/prod_ext [3]) );
  sky130_fd_sc_hd__dfrtp_1 \init_mac/prod_reg_reg[4]  ( .D(n56), .CLK(clk), 
        .RESET_B(rst_n), .Q(\init_mac/prod_ext [4]) );
  sky130_fd_sc_hd__dfrtp_1 \init_mac/prod_reg_reg[5]  ( .D(n55), .CLK(clk), 
        .RESET_B(rst_n), .Q(\init_mac/prod_ext [5]) );
  sky130_fd_sc_hd__dfrtp_1 \init_mac/prod_reg_reg[6]  ( .D(n54), .CLK(clk), 
        .RESET_B(rst_n), .Q(\init_mac/prod_ext [6]) );
  sky130_fd_sc_hd__dfrtp_1 \init_mac/prod_reg_reg[7]  ( .D(n53), .CLK(clk), 
        .RESET_B(rst_n), .Q(\init_mac/prod_ext [7]) );
  sky130_fd_sc_hd__dfrtp_1 \init_mac/prod_reg_reg[8]  ( .D(n52), .CLK(clk), 
        .RESET_B(rst_n), .Q(\init_mac/prod_ext [8]) );
  sky130_fd_sc_hd__dfrtp_1 \init_mac/prod_reg_reg[9]  ( .D(n51), .CLK(clk), 
        .RESET_B(rst_n), .Q(\init_mac/prod_ext [9]) );
  sky130_fd_sc_hd__dfrtp_1 \init_mac/prod_reg_reg[10]  ( .D(n50), .CLK(clk), 
        .RESET_B(rst_n), .Q(\init_mac/prod_ext [10]) );
  sky130_fd_sc_hd__dfrtp_1 \init_mac/prod_reg_reg[11]  ( .D(n49), .CLK(clk), 
        .RESET_B(rst_n), .Q(\init_mac/prod_ext [11]) );
  sky130_fd_sc_hd__dfrtp_1 \init_mac/prod_reg_reg[12]  ( .D(n48), .CLK(clk), 
        .RESET_B(rst_n), .Q(\init_mac/prod_ext [12]) );
  sky130_fd_sc_hd__dfrtp_1 \init_mac/prod_reg_reg[13]  ( .D(n47), .CLK(clk), 
        .RESET_B(rst_n), .Q(\init_mac/prod_ext [13]) );
  sky130_fd_sc_hd__dfrtp_1 \init_mac/prod_reg_reg[14]  ( .D(n46), .CLK(clk), 
        .RESET_B(rst_n), .Q(\init_mac/prod_ext [14]) );
  sky130_fd_sc_hd__dfrtp_1 \init_mac/prod_reg_reg[15]  ( .D(n45), .CLK(clk), 
        .RESET_B(rst_n), .Q(\init_mac/prod_ext_31 ) );
  sky130_fd_sc_hd__dfrtp_1 \init_mac/tap_idx_reg[1]  ( .D(n44), .CLK(clk), 
        .RESET_B(rst_n), .Q(\init_mac/tap_idx [1]) );
  sky130_fd_sc_hd__dfrtp_1 \init_mac/tap_idx_reg[0]  ( .D(n43), .CLK(clk), 
        .RESET_B(rst_n), .Q(\init_mac/tap_idx [0]) );
  sky130_fd_sc_hd__dfrtp_1 \init_mac/acc_out_reg[31]  ( .D(n69), .CLK(clk), 
        .RESET_B(rst_n), .Q(mac_out[31]) );
  sky130_fd_sc_hd__dfrtp_1 \tap_idx_reg[1]  ( .D(\read_offset[0] ), .CLK(clk), 
        .RESET_B(rst_n), .Q(tap_idx[1]) );
  sky130_fd_sc_hd__dfrtp_4 s1_reg ( .D(s0), .CLK(clk), .RESET_B(rst_n), .Q(s1)
         );
  sky130_fd_sc_hd__clkbuf_1 U100 ( .A(n414), .X(n415) );
  sky130_fd_sc_hd__nor2_1 U101 ( .A(n324), .B(n336), .Y(n326) );
  sky130_fd_sc_hd__nand2_1 U102 ( .A(n299), .B(n298), .Y(n370) );
  sky130_fd_sc_hd__nand2_1 U103 ( .A(n251), .B(n250), .Y(n288) );
  sky130_fd_sc_hd__o22ai_1 U104 ( .A1(n93), .A2(n241), .B1(n242), .B2(n138), 
        .Y(n142) );
  sky130_fd_sc_hd__ha_2 U105 ( .A(n215), .B(n214), .COUT(n231), .SUM(n206) );
  sky130_fd_sc_hd__xor2_1 U106 ( .A(n185), .B(n109), .X(n138) );
  sky130_fd_sc_hd__inv_2 U107 ( .A(x[6]), .Y(n105) );
  sky130_fd_sc_hd__clkinv_1 U108 ( .A(n163), .Y(n144) );
  sky130_fd_sc_hd__clkinv_1 U109 ( .A(n239), .Y(n91) );
  sky130_fd_sc_hd__clkinv_1 U110 ( .A(n334), .Y(n312) );
  sky130_fd_sc_hd__inv_2 U111 ( .A(n290), .Y(n348) );
  sky130_fd_sc_hd__inv_2 U112 ( .A(n422), .Y(n175) );
  sky130_fd_sc_hd__inv_2 U113 ( .A(n543), .Y(n545) );
  sky130_fd_sc_hd__and2_1 U114 ( .A(n168), .B(n115), .X(n79) );
  sky130_fd_sc_hd__clkbuf_1 U115 ( .A(n280), .X(n80) );
  sky130_fd_sc_hd__nand2_1 U116 ( .A(n213), .B(n81), .Y(n259) );
  sky130_fd_sc_hd__o21ai_1 U117 ( .A1(n117), .A2(n84), .B1(n212), .Y(n81) );
  sky130_fd_sc_hd__nand2_1 U118 ( .A(n82), .B(s1), .Y(n382) );
  sky130_fd_sc_hd__xnor2_1 U119 ( .A(n83), .B(n380), .Y(n82) );
  sky130_fd_sc_hd__inv_1 U120 ( .A(n379), .Y(n83) );
  sky130_fd_sc_hd__nand2_2 U121 ( .A(n156), .B(n451), .Y(n217) );
  sky130_fd_sc_hd__inv_1 U122 ( .A(n114), .Y(n428) );
  sky130_fd_sc_hd__o22ai_2 U123 ( .A1(n187), .A2(n282), .B1(n210), .B2(n280), 
        .Y(n84) );
  sky130_fd_sc_hd__nand2_1 U124 ( .A(n117), .B(n84), .Y(n213) );
  sky130_fd_sc_hd__xnor2_1 U125 ( .A(n84), .B(n212), .Y(n191) );
  sky130_fd_sc_hd__nor2_4 U126 ( .A(n300), .B(n301), .Y(n367) );
  sky130_fd_sc_hd__nand2_2 U127 ( .A(n85), .B(n240), .Y(n300) );
  sky130_fd_sc_hd__nand2_1 U128 ( .A(n87), .B(n268), .Y(n85) );
  sky130_fd_sc_hd__xor2_1 U129 ( .A(n245), .B(n86), .X(n268) );
  sky130_fd_sc_hd__xnor2_1 U130 ( .A(n247), .B(n249), .Y(n86) );
  sky130_fd_sc_hd__nand2_1 U131 ( .A(n88), .B(n239), .Y(n87) );
  sky130_fd_sc_hd__inv_1 U132 ( .A(n269), .Y(n88) );
  sky130_fd_sc_hd__o22ai_1 U133 ( .A1(n242), .A2(n89), .B1(n218), .B2(n241), 
        .Y(n234) );
  sky130_fd_sc_hd__o22a_1 U134 ( .A1(n242), .A2(n243), .B1(n89), .B2(n241), 
        .X(n247) );
  sky130_fd_sc_hd__xor2_1 U135 ( .A(n112), .B(x[5]), .X(n89) );
  sky130_fd_sc_hd__inv_2 U136 ( .A(n90), .Y(n299) );
  sky130_fd_sc_hd__nand2_1 U137 ( .A(n269), .B(n91), .Y(n240) );
  sky130_fd_sc_hd__xnor2_1 U138 ( .A(n268), .B(n92), .Y(n90) );
  sky130_fd_sc_hd__xnor2_1 U139 ( .A(n239), .B(n269), .Y(n92) );
  sky130_fd_sc_hd__nor2_1 U140 ( .A(n238), .B(n237), .Y(n239) );
  sky130_fd_sc_hd__inv_2 U141 ( .A(n109), .Y(n111) );
  sky130_fd_sc_hd__xor2_1 U142 ( .A(n141), .B(n142), .X(n154) );
  sky130_fd_sc_hd__xnor2_1 U143 ( .A(n109), .B(n98), .Y(n93) );
  sky130_fd_sc_hd__o22ai_1 U144 ( .A1(n97), .A2(n242), .B1(n103), .B2(n241), 
        .Y(n141) );
  sky130_fd_sc_hd__nand2_2 U145 ( .A(n95), .B(n94), .Y(n241) );
  sky130_fd_sc_hd__nand2_1 U146 ( .A(n110), .B(x[5]), .Y(n94) );
  sky130_fd_sc_hd__nand2_1 U147 ( .A(n103), .B(n96), .Y(n95) );
  sky130_fd_sc_hd__nand2_2 U148 ( .A(n110), .B(n96), .Y(n242) );
  sky130_fd_sc_hd__nand2_1 U149 ( .A(x[3]), .B(x[4]), .Y(n96) );
  sky130_fd_sc_hd__nand2_1 U150 ( .A(n111), .B(n98), .Y(n97) );
  sky130_fd_sc_hd__clkinv_1 U151 ( .A(n452), .Y(n98) );
  sky130_fd_sc_hd__o22ai_1 U152 ( .A1(n242), .A2(n218), .B1(n241), .B2(n184), 
        .Y(n117) );
  sky130_fd_sc_hd__o22ai_2 U153 ( .A1(n242), .A2(n184), .B1(n241), .B2(n138), 
        .Y(n194) );
  sky130_fd_sc_hd__nand2_2 U154 ( .A(n100), .B(n99), .Y(n110) );
  sky130_fd_sc_hd__clkinv_1 U155 ( .A(x[4]), .Y(n99) );
  sky130_fd_sc_hd__inv_2 U156 ( .A(x[3]), .Y(n100) );
  sky130_fd_sc_hd__nand2b_1 U157 ( .A_N(n293), .B(n295), .Y(n296) );
  sky130_fd_sc_hd__nand2b_1 U158 ( .A_N(n295), .B(n293), .Y(n101) );
  sky130_fd_sc_hd__xor2_1 U159 ( .A(n295), .B(n293), .X(n286) );
  sky130_fd_sc_hd__xnor2_2 U160 ( .A(x[6]), .B(x[5]), .Y(n186) );
  sky130_fd_sc_hd__nand2_2 U161 ( .A(n106), .B(n102), .Y(n282) );
  sky130_fd_sc_hd__nand2_1 U162 ( .A(n104), .B(n103), .Y(n102) );
  sky130_fd_sc_hd__inv_2 U163 ( .A(x[5]), .Y(n103) );
  sky130_fd_sc_hd__nand2_1 U164 ( .A(n105), .B(x[7]), .Y(n104) );
  sky130_fd_sc_hd__nand2_1 U165 ( .A(n107), .B(x[5]), .Y(n106) );
  sky130_fd_sc_hd__nand2_1 U166 ( .A(n108), .B(x[6]), .Y(n107) );
  sky130_fd_sc_hd__inv_2 U167 ( .A(x[7]), .Y(n108) );
  sky130_fd_sc_hd__inv_2 U168 ( .A(x[5]), .Y(n109) );
  sky130_fd_sc_hd__clkinv_1 U169 ( .A(n257), .Y(n112) );
  sky130_fd_sc_hd__xor2_1 U170 ( .A(n113), .B(x[5]), .X(n243) );
  sky130_fd_sc_hd__clkinv_1 U171 ( .A(n273), .Y(n113) );
  sky130_fd_sc_hd__xnor2_1 U172 ( .A(n115), .B(n168), .Y(n114) );
  sky130_fd_sc_hd__o22ai_2 U173 ( .A1(n100), .A2(n252), .B1(n166), .B2(n253), 
        .Y(n115) );
  sky130_fd_sc_hd__clkbuf_1 U174 ( .A(n117), .X(n116) );
  sky130_fd_sc_hd__xnor2_1 U175 ( .A(n116), .B(n191), .Y(n204) );
  sky130_fd_sc_hd__nand2_1 U176 ( .A(n118), .B(n223), .Y(n385) );
  sky130_fd_sc_hd__nor2_2 U177 ( .A(n223), .B(n118), .Y(n383) );
  sky130_fd_sc_hd__xnor2_1 U178 ( .A(n263), .B(n220), .Y(n118) );
  sky130_fd_sc_hd__inv_1 U179 ( .A(n254), .Y(n255) );
  sky130_fd_sc_hd__o21a_1 U180 ( .A1(n379), .A2(n318), .B1(n317), .X(n319) );
  sky130_fd_sc_hd__and2_1 U181 ( .A(n142), .B(n141), .X(n119) );
  sky130_fd_sc_hd__xnor2_1 U182 ( .A(n228), .B(n271), .Y(n254) );
  sky130_fd_sc_hd__nor2b_1 U183 ( .B_N(n452), .A(n242), .Y(n170) );
  sky130_fd_sc_hd__clkinv_1 U185 ( .A(tap_idx[0]), .Y(n120) );
  sky130_fd_sc_hd__nor2_1 U186 ( .A(tap_idx[1]), .B(n120), .Y(\read_offset[0] ) );
  sky130_fd_sc_hd__clkinv_1 U187 ( .A(\init_mac/do_acc ), .Y(n121) );
  sky130_fd_sc_hd__clkinv_1 U188 ( .A(\init_mac/tap_idx [0]), .Y(n124) );
  sky130_fd_sc_hd__nand2_1 U189 ( .A(n124), .B(\init_mac/tap_idx [1]), .Y(n542) );
  sky130_fd_sc_hd__nor2_1 U190 ( .A(n121), .B(n542), .Y(\init_mac/N124 ) );
  sky130_fd_sc_hd__nor2_1 U191 ( .A(tap_idx[0]), .B(tap_idx[1]), .Y(n699) );
  sky130_fd_sc_hd__clkinv_1 U192 ( .A(\init_mac/tap_idx [1]), .Y(n123) );
  sky130_fd_sc_hd__nand3_1 U193 ( .A(n123), .B(\init_mac/do_acc ), .C(
        \init_mac/tap_idx [0]), .Y(n122) );
  sky130_fd_sc_hd__o21ai_1 U194 ( .A1(\init_mac/do_acc ), .A2(n123), .B1(n122), 
        .Y(n44) );
  sky130_fd_sc_hd__nand2_1 U195 ( .A(\init_mac/do_acc ), .B(n124), .Y(n125) );
  sky130_fd_sc_hd__o22ai_1 U196 ( .A1(n125), .A2(\init_mac/tap_idx [1]), .B1(
        \init_mac/do_acc ), .B2(n124), .Y(n43) );
  sky130_fd_sc_hd__clkinv_1 U197 ( .A(\init_mac/prod_ext_31 ), .Y(n321) );
  sky130_fd_sc_hd__a222oi_1 U198 ( .A1(tap_idx[1]), .A2(\weights[2][2] ), .B1(
        n699), .B2(\weights[0][2] ), .C1(\read_offset[0] ), .C2(
        \weights[1][2] ), .Y(n126) );
  sky130_fd_sc_hd__clkinv_1 U199 ( .A(n126), .Y(n209) );
  sky130_fd_sc_hd__xnor2_1 U200 ( .A(x[3]), .B(n209), .Y(n145) );
  sky130_fd_sc_hd__xor2_1 U201 ( .A(x[3]), .B(x[2]), .X(n127) );
  sky130_fd_sc_hd__xnor2_2 U202 ( .A(x[2]), .B(x[1]), .Y(n146) );
  sky130_fd_sc_hd__nand2_2 U203 ( .A(n127), .B(n146), .Y(n252) );
  sky130_fd_sc_hd__buf_6 U204 ( .A(n146), .X(n253) );
  sky130_fd_sc_hd__buf_6 U205 ( .A(x[3]), .X(n228) );
  sky130_fd_sc_hd__a222oi_1 U206 ( .A1(tap_idx[1]), .A2(\weights[2][3] ), .B1(
        n699), .B2(\weights[0][3] ), .C1(\read_offset[0] ), .C2(
        \weights[1][3] ), .Y(n128) );
  sky130_fd_sc_hd__clkinv_1 U207 ( .A(n128), .Y(n226) );
  sky130_fd_sc_hd__xnor2_1 U208 ( .A(n228), .B(n226), .Y(n140) );
  sky130_fd_sc_hd__o22ai_1 U209 ( .A1(n145), .A2(n252), .B1(n253), .B2(n140), 
        .Y(n153) );
  sky130_fd_sc_hd__clkbuf_1 U210 ( .A(n153), .X(n133) );
  sky130_fd_sc_hd__inv_2 U211 ( .A(x[0]), .Y(n451) );
  sky130_fd_sc_hd__buf_6 U212 ( .A(x[1]), .X(n156) );
  sky130_fd_sc_hd__a222oi_1 U213 ( .A1(tap_idx[1]), .A2(\weights[2][4] ), .B1(
        n699), .B2(\weights[0][4] ), .C1(\read_offset[0] ), .C2(
        \weights[1][4] ), .Y(n129) );
  sky130_fd_sc_hd__clkinv_1 U214 ( .A(n129), .Y(n257) );
  sky130_fd_sc_hd__xnor2_1 U215 ( .A(n156), .B(n257), .Y(n149) );
  sky130_fd_sc_hd__a222oi_1 U216 ( .A1(tap_idx[1]), .A2(\weights[2][5] ), .B1(
        n699), .B2(\weights[0][5] ), .C1(\read_offset[0] ), .C2(
        \weights[1][5] ), .Y(n130) );
  sky130_fd_sc_hd__clkinv_1 U217 ( .A(n130), .Y(n273) );
  sky130_fd_sc_hd__xnor2_1 U218 ( .A(n156), .B(n273), .Y(n137) );
  sky130_fd_sc_hd__o22ai_1 U219 ( .A1(n217), .A2(n149), .B1(n451), .B2(n137), 
        .Y(n152) );
  sky130_fd_sc_hd__a222oi_1 U220 ( .A1(tap_idx[1]), .A2(\weights[2][0] ), .B1(
        n699), .B2(\weights[0][0] ), .C1(\read_offset[0] ), .C2(
        \weights[1][0] ), .Y(n131) );
  sky130_fd_sc_hd__clkinv_1 U221 ( .A(n131), .Y(n452) );
  sky130_fd_sc_hd__a222oi_1 U222 ( .A1(tap_idx[1]), .A2(\weights[2][1] ), .B1(
        n699), .B2(\weights[0][1] ), .C1(\read_offset[0] ), .C2(
        \weights[1][1] ), .Y(n132) );
  sky130_fd_sc_hd__clkinv_1 U223 ( .A(n132), .Y(n185) );
  sky130_fd_sc_hd__o21ai_1 U224 ( .A1(n133), .A2(n152), .B1(n154), .Y(n135) );
  sky130_fd_sc_hd__nand2_1 U225 ( .A(n152), .B(n133), .Y(n134) );
  sky130_fd_sc_hd__nand2_1 U226 ( .A(n135), .B(n134), .Y(n178) );
  sky130_fd_sc_hd__a222oi_1 U227 ( .A1(tap_idx[1]), .A2(\weights[2][6] ), .B1(
        n699), .B2(\weights[0][6] ), .C1(\read_offset[0] ), .C2(
        \weights[1][6] ), .Y(n136) );
  sky130_fd_sc_hd__clkinv_1 U228 ( .A(n136), .Y(n270) );
  sky130_fd_sc_hd__xnor2_1 U229 ( .A(x[1]), .B(n270), .Y(n198) );
  sky130_fd_sc_hd__o22ai_1 U230 ( .A1(n137), .A2(n217), .B1(n451), .B2(n198), 
        .Y(n192) );
  sky130_fd_sc_hd__buf_6 U231 ( .A(n186), .X(n280) );
  sky130_fd_sc_hd__nor2b_1 U232 ( .B_N(n452), .A(n280), .Y(n193) );
  sky130_fd_sc_hd__xnor2_1 U233 ( .A(x[5]), .B(n209), .Y(n184) );
  sky130_fd_sc_hd__xnor2_1 U234 ( .A(n193), .B(n194), .Y(n139) );
  sky130_fd_sc_hd__xnor2_1 U235 ( .A(n192), .B(n139), .Y(n180) );
  sky130_fd_sc_hd__clkbuf_1 U236 ( .A(n252), .X(n162) );
  sky130_fd_sc_hd__xnor2_1 U237 ( .A(n228), .B(n257), .Y(n190) );
  sky130_fd_sc_hd__o22ai_1 U238 ( .A1(n162), .A2(n140), .B1(n253), .B2(n190), 
        .Y(n181) );
  sky130_fd_sc_hd__xnor2_1 U239 ( .A(n181), .B(n119), .Y(n143) );
  sky130_fd_sc_hd__xnor2_1 U240 ( .A(n180), .B(n143), .Y(n179) );
  sky130_fd_sc_hd__nor2_1 U241 ( .A(n178), .B(n179), .Y(n402) );
  sky130_fd_sc_hd__xnor2_1 U242 ( .A(n228), .B(n185), .Y(n163) );
  sky130_fd_sc_hd__nand2_1 U243 ( .A(n256), .B(n144), .Y(n148) );
  sky130_fd_sc_hd__or2_1 U244 ( .A(n146), .B(n145), .X(n147) );
  sky130_fd_sc_hd__nand2_1 U245 ( .A(n148), .B(n147), .Y(n172) );
  sky130_fd_sc_hd__xnor2_1 U246 ( .A(n156), .B(n226), .Y(n164) );
  sky130_fd_sc_hd__o22ai_1 U247 ( .A1(n451), .A2(n149), .B1(n164), .B2(n217), 
        .Y(n169) );
  sky130_fd_sc_hd__o21ai_1 U248 ( .A1(n170), .A2(n172), .B1(n169), .Y(n151) );
  sky130_fd_sc_hd__nand2_1 U249 ( .A(n172), .B(n170), .Y(n150) );
  sky130_fd_sc_hd__nand2_1 U250 ( .A(n151), .B(n150), .Y(n176) );
  sky130_fd_sc_hd__xnor2_1 U251 ( .A(n153), .B(n152), .Y(n155) );
  sky130_fd_sc_hd__xnor2_1 U252 ( .A(n155), .B(n154), .Y(n177) );
  sky130_fd_sc_hd__nor2_1 U253 ( .A(n176), .B(n177), .Y(n411) );
  sky130_fd_sc_hd__xnor2_1 U254 ( .A(n156), .B(n185), .Y(n158) );
  sky130_fd_sc_hd__o22ai_2 U255 ( .A1(n451), .A2(n158), .B1(n452), .B2(n217), 
        .Y(n444) );
  sky130_fd_sc_hd__nand2b_1 U256 ( .A_N(n452), .B(n156), .Y(n157) );
  sky130_fd_sc_hd__nand2_1 U257 ( .A(n217), .B(n157), .Y(n445) );
  sky130_fd_sc_hd__nand2_1 U258 ( .A(n444), .B(n445), .Y(n446) );
  sky130_fd_sc_hd__nor2b_1 U259 ( .B_N(n452), .A(n253), .Y(n159) );
  sky130_fd_sc_hd__xnor2_1 U260 ( .A(x[1]), .B(n209), .Y(n165) );
  sky130_fd_sc_hd__o22ai_1 U261 ( .A1(n158), .A2(n217), .B1(n451), .B2(n165), 
        .Y(n160) );
  sky130_fd_sc_hd__nor2_1 U262 ( .A(n159), .B(n160), .Y(n437) );
  sky130_fd_sc_hd__nand2_1 U263 ( .A(n160), .B(n159), .Y(n438) );
  sky130_fd_sc_hd__o21a_1 U264 ( .A1(n446), .A2(n437), .B1(n438), .X(n432) );
  sky130_fd_sc_hd__xnor2_1 U265 ( .A(n228), .B(n452), .Y(n161) );
  sky130_fd_sc_hd__o22ai_1 U266 ( .A1(n253), .A2(n163), .B1(n162), .B2(n161), 
        .Y(n429) );
  sky130_fd_sc_hd__o22ai_1 U267 ( .A1(n217), .A2(n165), .B1(n451), .B2(n164), 
        .Y(n168) );
  sky130_fd_sc_hd__nand2b_1 U268 ( .A_N(n452), .B(n228), .Y(n166) );
  sky130_fd_sc_hd__nor2_1 U269 ( .A(n429), .B(n428), .Y(n167) );
  sky130_fd_sc_hd__nand2_1 U270 ( .A(n428), .B(n429), .Y(n430) );
  sky130_fd_sc_hd__o21ai_2 U271 ( .A1(n432), .A2(n167), .B1(n430), .Y(n420) );
  sky130_fd_sc_hd__xnor2_1 U272 ( .A(n170), .B(n169), .Y(n171) );
  sky130_fd_sc_hd__xnor2_1 U273 ( .A(n172), .B(n171), .Y(n174) );
  sky130_fd_sc_hd__nor2_1 U274 ( .A(n79), .B(n174), .Y(n173) );
  sky130_fd_sc_hd__inv_1 U275 ( .A(n173), .Y(n421) );
  sky130_fd_sc_hd__nand2_1 U276 ( .A(n174), .B(n79), .Y(n422) );
  sky130_fd_sc_hd__a21oi_2 U277 ( .A1(n420), .A2(n421), .B1(n175), .Y(n414) );
  sky130_fd_sc_hd__nand2_1 U278 ( .A(n177), .B(n176), .Y(n412) );
  sky130_fd_sc_hd__o21a_1 U279 ( .A1(n411), .A2(n414), .B1(n412), .X(n405) );
  sky130_fd_sc_hd__nand2_1 U280 ( .A(n179), .B(n178), .Y(n403) );
  sky130_fd_sc_hd__o21ai_2 U281 ( .A1(n402), .A2(n405), .B1(n403), .Y(n388) );
  sky130_fd_sc_hd__o21ai_1 U282 ( .A1(n119), .A2(n181), .B1(n180), .Y(n183) );
  sky130_fd_sc_hd__nand2_1 U283 ( .A(n119), .B(n181), .Y(n182) );
  sky130_fd_sc_hd__nand2_1 U284 ( .A(n183), .B(n182), .Y(n221) );
  sky130_fd_sc_hd__xnor2_1 U285 ( .A(x[5]), .B(n226), .Y(n218) );
  sky130_fd_sc_hd__xnor2_1 U286 ( .A(x[7]), .B(n185), .Y(n210) );
  sky130_fd_sc_hd__xnor2_1 U287 ( .A(x[7]), .B(n452), .Y(n187) );
  sky130_fd_sc_hd__inv_2 U288 ( .A(n253), .Y(n189) );
  sky130_fd_sc_hd__xnor2_1 U289 ( .A(n228), .B(n273), .Y(n211) );
  sky130_fd_sc_hd__inv_2 U290 ( .A(n211), .Y(n188) );
  sky130_fd_sc_hd__o2bb2ai_1 U291 ( .B1(n252), .B2(n190), .A1_N(n189), .A2_N(
        n188), .Y(n212) );
  sky130_fd_sc_hd__o21ai_1 U292 ( .A1(n193), .A2(n194), .B1(n192), .Y(n196) );
  sky130_fd_sc_hd__nand2_1 U293 ( .A(n194), .B(n193), .Y(n195) );
  sky130_fd_sc_hd__nand2_1 U294 ( .A(n196), .B(n195), .Y(n205) );
  sky130_fd_sc_hd__a222oi_1 U295 ( .A1(tap_idx[1]), .A2(\weights[2][7] ), .B1(
        n699), .B2(\weights[0][7] ), .C1(\read_offset[0] ), .C2(
        \weights[1][7] ), .Y(n197) );
  sky130_fd_sc_hd__clkinv_1 U296 ( .A(n197), .Y(n271) );
  sky130_fd_sc_hd__xnor2_1 U297 ( .A(x[1]), .B(n271), .Y(n216) );
  sky130_fd_sc_hd__o22ai_1 U298 ( .A1(n451), .A2(n216), .B1(n198), .B2(n217), 
        .Y(n215) );
  sky130_fd_sc_hd__nand2b_1 U299 ( .A_N(n452), .B(x[7]), .Y(n200) );
  sky130_fd_sc_hd__clkinv_1 U300 ( .A(x[7]), .Y(n199) );
  sky130_fd_sc_hd__o22ai_2 U301 ( .A1(n186), .A2(n200), .B1(n199), .B2(n282), 
        .Y(n214) );
  sky130_fd_sc_hd__xnor2_1 U302 ( .A(n205), .B(n206), .Y(n201) );
  sky130_fd_sc_hd__xnor2_1 U303 ( .A(n204), .B(n201), .Y(n222) );
  sky130_fd_sc_hd__nor2_1 U304 ( .A(n221), .B(n222), .Y(n387) );
  sky130_fd_sc_hd__nor2_1 U305 ( .A(n205), .B(n206), .Y(n202) );
  sky130_fd_sc_hd__inv_1 U306 ( .A(n202), .Y(n203) );
  sky130_fd_sc_hd__nand2_1 U307 ( .A(n204), .B(n203), .Y(n208) );
  sky130_fd_sc_hd__nand2_1 U308 ( .A(n206), .B(n205), .Y(n207) );
  sky130_fd_sc_hd__nand2_1 U309 ( .A(n208), .B(n207), .Y(n223) );
  sky130_fd_sc_hd__xnor2_1 U310 ( .A(x[7]), .B(n209), .Y(n227) );
  sky130_fd_sc_hd__o22ai_2 U311 ( .A1(n282), .A2(n210), .B1(n280), .B2(n227), 
        .Y(n238) );
  sky130_fd_sc_hd__xnor2_1 U312 ( .A(n228), .B(n270), .Y(n229) );
  sky130_fd_sc_hd__o22ai_1 U313 ( .A1(n252), .A2(n211), .B1(n253), .B2(n229), 
        .Y(n237) );
  sky130_fd_sc_hd__xnor2_1 U314 ( .A(n238), .B(n237), .Y(n260) );
  sky130_fd_sc_hd__xnor2_1 U315 ( .A(n260), .B(n259), .Y(n220) );
  sky130_fd_sc_hd__a21o_1 U316 ( .A1(n217), .A2(n451), .B1(n216), .X(n233) );
  sky130_fd_sc_hd__xnor2_1 U317 ( .A(n233), .B(n234), .Y(n219) );
  sky130_fd_sc_hd__xnor2_1 U318 ( .A(n231), .B(n219), .Y(n263) );
  sky130_fd_sc_hd__nor2_2 U319 ( .A(n387), .B(n383), .Y(n225) );
  sky130_fd_sc_hd__nand2_1 U320 ( .A(n222), .B(n221), .Y(n389) );
  sky130_fd_sc_hd__o21ai_2 U321 ( .A1(n383), .A2(n389), .B1(n385), .Y(n224) );
  sky130_fd_sc_hd__a21oi_4 U322 ( .A1(n388), .A2(n225), .B1(n224), .Y(n379) );
  sky130_fd_sc_hd__xnor2_1 U323 ( .A(x[7]), .B(n226), .Y(n258) );
  sky130_fd_sc_hd__o22ai_2 U324 ( .A1(n258), .A2(n280), .B1(n282), .B2(n227), 
        .Y(n249) );
  sky130_fd_sc_hd__o22ai_2 U325 ( .A1(n253), .A2(n254), .B1(n252), .B2(n229), 
        .Y(n285) );
  sky130_fd_sc_hd__inv_1 U326 ( .A(n285), .Y(n245) );
  sky130_fd_sc_hd__inv_2 U327 ( .A(n233), .Y(n230) );
  sky130_fd_sc_hd__nand2b_1 U328 ( .A_N(n234), .B(n230), .Y(n232) );
  sky130_fd_sc_hd__nand2_1 U329 ( .A(n232), .B(n231), .Y(n236) );
  sky130_fd_sc_hd__nand2_1 U330 ( .A(n234), .B(n233), .Y(n235) );
  sky130_fd_sc_hd__nand2_1 U331 ( .A(n236), .B(n235), .Y(n269) );
  sky130_fd_sc_hd__clkbuf_1 U332 ( .A(n241), .X(n276) );
  sky130_fd_sc_hd__xnor2_1 U333 ( .A(n111), .B(n270), .Y(n272) );
  sky130_fd_sc_hd__o22ai_1 U334 ( .A1(n276), .A2(n243), .B1(n242), .B2(n272), 
        .Y(n289) );
  sky130_fd_sc_hd__inv_2 U335 ( .A(n249), .Y(n244) );
  sky130_fd_sc_hd__nand2_1 U336 ( .A(n244), .B(n247), .Y(n246) );
  sky130_fd_sc_hd__nand2_1 U337 ( .A(n246), .B(n245), .Y(n251) );
  sky130_fd_sc_hd__inv_1 U338 ( .A(n247), .Y(n248) );
  sky130_fd_sc_hd__nand2_1 U339 ( .A(n249), .B(n248), .Y(n250) );
  sky130_fd_sc_hd__inv_1 U340 ( .A(n252), .Y(n256) );
  sky130_fd_sc_hd__o21ai_2 U341 ( .A1(n256), .A2(n189), .B1(n255), .Y(n284) );
  sky130_fd_sc_hd__xnor2_1 U342 ( .A(x[7]), .B(n257), .Y(n281) );
  sky130_fd_sc_hd__o22ai_1 U343 ( .A1(n282), .A2(n258), .B1(n280), .B2(n281), 
        .Y(n283) );
  sky130_fd_sc_hd__clkbuf_1 U344 ( .A(n259), .X(n265) );
  sky130_fd_sc_hd__clkbuf_1 U345 ( .A(n260), .X(n264) );
  sky130_fd_sc_hd__inv_2 U346 ( .A(n264), .Y(n261) );
  sky130_fd_sc_hd__nand2b_1 U347 ( .A_N(n265), .B(n261), .Y(n262) );
  sky130_fd_sc_hd__nand2_1 U348 ( .A(n263), .B(n262), .Y(n267) );
  sky130_fd_sc_hd__nand2_1 U349 ( .A(n265), .B(n264), .Y(n266) );
  sky130_fd_sc_hd__nand2_1 U350 ( .A(n267), .B(n266), .Y(n298) );
  sky130_fd_sc_hd__nor2_2 U351 ( .A(n298), .B(n299), .Y(n376) );
  sky130_fd_sc_hd__nor2_2 U352 ( .A(n367), .B(n376), .Y(n358) );
  sky130_fd_sc_hd__xnor2_1 U353 ( .A(x[7]), .B(n271), .Y(n277) );
  sky130_fd_sc_hd__clkbuf_1 U354 ( .A(n282), .X(n278) );
  sky130_fd_sc_hd__xnor2_1 U355 ( .A(x[7]), .B(n270), .Y(n274) );
  sky130_fd_sc_hd__o22ai_1 U356 ( .A1(n80), .A2(n277), .B1(n278), .B2(n274), 
        .Y(n309) );
  sky130_fd_sc_hd__inv_2 U357 ( .A(n309), .Y(n307) );
  sky130_fd_sc_hd__xnor2_1 U358 ( .A(n111), .B(n271), .Y(n275) );
  sky130_fd_sc_hd__o22ai_2 U359 ( .A1(n242), .A2(n275), .B1(n276), .B2(n272), 
        .Y(n293) );
  sky130_fd_sc_hd__xnor2_1 U360 ( .A(x[7]), .B(n273), .Y(n279) );
  sky130_fd_sc_hd__o22ai_1 U361 ( .A1(n282), .A2(n279), .B1(n280), .B2(n274), 
        .Y(n292) );
  sky130_fd_sc_hd__a21o_1 U362 ( .A1(n276), .A2(n242), .B1(n275), .X(n291) );
  sky130_fd_sc_hd__nor2_1 U363 ( .A(n307), .B(n308), .Y(n324) );
  sky130_fd_sc_hd__inv_1 U364 ( .A(n324), .Y(n335) );
  sky130_fd_sc_hd__a21o_1 U365 ( .A1(n278), .A2(n80), .B1(n277), .X(n310) );
  sky130_fd_sc_hd__or2_0 U366 ( .A(n309), .B(n310), .X(n323) );
  sky130_fd_sc_hd__nand2_1 U367 ( .A(n335), .B(n323), .Y(n314) );
  sky130_fd_sc_hd__o22ai_1 U368 ( .A1(n282), .A2(n281), .B1(n280), .B2(n279), 
        .Y(n295) );
  sky130_fd_sc_hd__fah_1 U369 ( .A(n285), .B(n284), .CI(n283), .COUT(n294), 
        .SUM(n287) );
  sky130_fd_sc_hd__xnor2_1 U370 ( .A(n286), .B(n294), .Y(n302) );
  sky130_fd_sc_hd__fah_1 U371 ( .A(n289), .B(n288), .CI(n287), .COUT(n303), 
        .SUM(n301) );
  sky130_fd_sc_hd__nor2_1 U372 ( .A(n302), .B(n303), .Y(n290) );
  sky130_fd_sc_hd__fah_1 U373 ( .A(n293), .B(n292), .CI(n291), .COUT(n308), 
        .SUM(n304) );
  sky130_fd_sc_hd__nand2_1 U374 ( .A(n294), .B(n101), .Y(n297) );
  sky130_fd_sc_hd__nand2_1 U375 ( .A(n297), .B(n296), .Y(n305) );
  sky130_fd_sc_hd__nor2_1 U376 ( .A(n304), .B(n305), .Y(n306) );
  sky130_fd_sc_hd__inv_1 U377 ( .A(n306), .Y(n347) );
  sky130_fd_sc_hd__nand2_2 U378 ( .A(n348), .B(n347), .Y(n336) );
  sky130_fd_sc_hd__nor2_1 U379 ( .A(n314), .B(n336), .Y(n316) );
  sky130_fd_sc_hd__nand2_1 U380 ( .A(n358), .B(n316), .Y(n318) );
  sky130_fd_sc_hd__nand2_1 U381 ( .A(n301), .B(n300), .Y(n368) );
  sky130_fd_sc_hd__o21ai_2 U382 ( .A1(n367), .A2(n370), .B1(n368), .Y(n359) );
  sky130_fd_sc_hd__nand2_1 U383 ( .A(n303), .B(n302), .Y(n349) );
  sky130_fd_sc_hd__nand2_1 U384 ( .A(n305), .B(n304), .Y(n346) );
  sky130_fd_sc_hd__o21a_1 U385 ( .A1(n349), .A2(n306), .B1(n346), .X(n337) );
  sky130_fd_sc_hd__nand2_1 U386 ( .A(n308), .B(n307), .Y(n334) );
  sky130_fd_sc_hd__nand2_1 U387 ( .A(n310), .B(n309), .Y(n322) );
  sky130_fd_sc_hd__clkinv_1 U388 ( .A(n322), .Y(n311) );
  sky130_fd_sc_hd__a21oi_1 U389 ( .A1(n312), .A2(n323), .B1(n311), .Y(n313) );
  sky130_fd_sc_hd__o21ai_1 U390 ( .A1(n337), .A2(n314), .B1(n313), .Y(n315) );
  sky130_fd_sc_hd__a21oi_1 U391 ( .A1(n316), .A2(n359), .B1(n315), .Y(n317) );
  sky130_fd_sc_hd__nand2_1 U392 ( .A(n319), .B(s1), .Y(n320) );
  sky130_fd_sc_hd__o21ai_1 U393 ( .A1(s1), .A2(n321), .B1(n320), .Y(n45) );
  sky130_fd_sc_hd__nand2_1 U394 ( .A(n323), .B(n322), .Y(n330) );
  sky130_fd_sc_hd__nand2_1 U395 ( .A(n326), .B(n358), .Y(n328) );
  sky130_fd_sc_hd__o21ai_1 U396 ( .A1(n324), .A2(n337), .B1(n334), .Y(n325) );
  sky130_fd_sc_hd__a21oi_1 U397 ( .A1(n359), .A2(n326), .B1(n325), .Y(n327) );
  sky130_fd_sc_hd__o21ai_1 U398 ( .A1(n328), .A2(n379), .B1(n327), .Y(n329) );
  sky130_fd_sc_hd__xnor2_1 U399 ( .A(n330), .B(n329), .Y(n331) );
  sky130_fd_sc_hd__nand2_1 U400 ( .A(n331), .B(s1), .Y(n333) );
  sky130_fd_sc_hd__clkinv_1 U401 ( .A(s1), .Y(n454) );
  sky130_fd_sc_hd__nand2_1 U402 ( .A(n454), .B(\init_mac/prod_ext [14]), .Y(
        n332) );
  sky130_fd_sc_hd__nand2_1 U403 ( .A(n333), .B(n332), .Y(n46) );
  sky130_fd_sc_hd__nand2_1 U404 ( .A(n335), .B(n334), .Y(n342) );
  sky130_fd_sc_hd__inv_1 U405 ( .A(n336), .Y(n338) );
  sky130_fd_sc_hd__nand2_1 U406 ( .A(n358), .B(n338), .Y(n340) );
  sky130_fd_sc_hd__a21boi_1 U407 ( .A1(n359), .A2(n338), .B1_N(n337), .Y(n339)
         );
  sky130_fd_sc_hd__o21ai_1 U408 ( .A1(n340), .A2(n379), .B1(n339), .Y(n341) );
  sky130_fd_sc_hd__xnor2_1 U409 ( .A(n342), .B(n341), .Y(n343) );
  sky130_fd_sc_hd__nand2_1 U410 ( .A(n343), .B(s1), .Y(n345) );
  sky130_fd_sc_hd__nand2_1 U411 ( .A(n454), .B(\init_mac/prod_ext [13]), .Y(
        n344) );
  sky130_fd_sc_hd__nand2_1 U412 ( .A(n345), .B(n344), .Y(n47) );
  sky130_fd_sc_hd__nand2_1 U413 ( .A(n347), .B(n346), .Y(n353) );
  sky130_fd_sc_hd__clkbuf_1 U414 ( .A(n348), .X(n357) );
  sky130_fd_sc_hd__nand2_1 U415 ( .A(n358), .B(n357), .Y(n351) );
  sky130_fd_sc_hd__a21boi_1 U416 ( .A1(n359), .A2(n357), .B1_N(n349), .Y(n350)
         );
  sky130_fd_sc_hd__o21ai_1 U417 ( .A1(n351), .A2(n379), .B1(n350), .Y(n352) );
  sky130_fd_sc_hd__xnor2_1 U418 ( .A(n353), .B(n352), .Y(n354) );
  sky130_fd_sc_hd__nand2_1 U419 ( .A(n354), .B(s1), .Y(n356) );
  sky130_fd_sc_hd__nand2_1 U420 ( .A(n454), .B(\init_mac/prod_ext [12]), .Y(
        n355) );
  sky130_fd_sc_hd__nand2_1 U421 ( .A(n356), .B(n355), .Y(n48) );
  sky130_fd_sc_hd__nand2_1 U422 ( .A(n357), .B(n349), .Y(n363) );
  sky130_fd_sc_hd__inv_1 U423 ( .A(n358), .Y(n361) );
  sky130_fd_sc_hd__inv_1 U424 ( .A(n359), .Y(n360) );
  sky130_fd_sc_hd__o21ai_1 U425 ( .A1(n361), .A2(n379), .B1(n360), .Y(n362) );
  sky130_fd_sc_hd__xnor2_1 U426 ( .A(n363), .B(n362), .Y(n364) );
  sky130_fd_sc_hd__nand2_1 U427 ( .A(n364), .B(s1), .Y(n366) );
  sky130_fd_sc_hd__nand2_1 U428 ( .A(n454), .B(\init_mac/prod_ext [11]), .Y(
        n365) );
  sky130_fd_sc_hd__nand2_1 U429 ( .A(n366), .B(n365), .Y(n49) );
  sky130_fd_sc_hd__inv_1 U430 ( .A(n367), .Y(n369) );
  sky130_fd_sc_hd__nand2_1 U431 ( .A(n369), .B(n368), .Y(n372) );
  sky130_fd_sc_hd__clkbuf_1 U432 ( .A(n370), .X(n377) );
  sky130_fd_sc_hd__o21ai_1 U433 ( .A1(n376), .A2(n379), .B1(n377), .Y(n371) );
  sky130_fd_sc_hd__xnor2_1 U434 ( .A(n372), .B(n371), .Y(n373) );
  sky130_fd_sc_hd__nand2_1 U435 ( .A(n373), .B(s1), .Y(n375) );
  sky130_fd_sc_hd__nand2_1 U436 ( .A(n454), .B(\init_mac/prod_ext [10]), .Y(
        n374) );
  sky130_fd_sc_hd__nand2_1 U437 ( .A(n375), .B(n374), .Y(n50) );
  sky130_fd_sc_hd__inv_1 U438 ( .A(n376), .Y(n378) );
  sky130_fd_sc_hd__nand2_1 U439 ( .A(n378), .B(n377), .Y(n380) );
  sky130_fd_sc_hd__nand2_1 U440 ( .A(n454), .B(\init_mac/prod_ext [9]), .Y(
        n381) );
  sky130_fd_sc_hd__nand2_1 U441 ( .A(n382), .B(n381), .Y(n51) );
  sky130_fd_sc_hd__clkbuf_1 U442 ( .A(n383), .X(n384) );
  sky130_fd_sc_hd__inv_2 U443 ( .A(n384), .Y(n386) );
  sky130_fd_sc_hd__nand2_1 U444 ( .A(n386), .B(n385), .Y(n391) );
  sky130_fd_sc_hd__clkbuf_1 U445 ( .A(n387), .X(n395) );
  sky130_fd_sc_hd__inv_1 U446 ( .A(n388), .Y(n398) );
  sky130_fd_sc_hd__o21ai_1 U447 ( .A1(n395), .A2(n398), .B1(n389), .Y(n390) );
  sky130_fd_sc_hd__xnor2_1 U448 ( .A(n391), .B(n390), .Y(n392) );
  sky130_fd_sc_hd__nand2_1 U449 ( .A(n392), .B(s1), .Y(n394) );
  sky130_fd_sc_hd__nand2_1 U450 ( .A(n454), .B(\init_mac/prod_ext [8]), .Y(
        n393) );
  sky130_fd_sc_hd__nand2_1 U451 ( .A(n394), .B(n393), .Y(n52) );
  sky130_fd_sc_hd__inv_2 U452 ( .A(n395), .Y(n396) );
  sky130_fd_sc_hd__nand2_1 U453 ( .A(n396), .B(n389), .Y(n397) );
  sky130_fd_sc_hd__xor2_1 U454 ( .A(n398), .B(n397), .X(n399) );
  sky130_fd_sc_hd__nand2_1 U455 ( .A(n399), .B(s1), .Y(n401) );
  sky130_fd_sc_hd__nand2_1 U456 ( .A(n454), .B(\init_mac/prod_ext [7]), .Y(
        n400) );
  sky130_fd_sc_hd__nand2_1 U457 ( .A(n401), .B(n400), .Y(n53) );
  sky130_fd_sc_hd__inv_1 U458 ( .A(n402), .Y(n404) );
  sky130_fd_sc_hd__nand2_1 U459 ( .A(n404), .B(n403), .Y(n407) );
  sky130_fd_sc_hd__buf_2 U460 ( .A(n405), .X(n406) );
  sky130_fd_sc_hd__xor2_1 U461 ( .A(n407), .B(n406), .X(n408) );
  sky130_fd_sc_hd__nand2_1 U462 ( .A(n408), .B(s1), .Y(n410) );
  sky130_fd_sc_hd__nand2_1 U463 ( .A(n454), .B(\init_mac/prod_ext [6]), .Y(
        n409) );
  sky130_fd_sc_hd__nand2_1 U464 ( .A(n410), .B(n409), .Y(n54) );
  sky130_fd_sc_hd__clkinv_1 U465 ( .A(n411), .Y(n413) );
  sky130_fd_sc_hd__nand2_1 U466 ( .A(n413), .B(n412), .Y(n416) );
  sky130_fd_sc_hd__xor2_1 U467 ( .A(n416), .B(n415), .X(n417) );
  sky130_fd_sc_hd__nand2_1 U468 ( .A(n417), .B(s1), .Y(n419) );
  sky130_fd_sc_hd__nand2_1 U469 ( .A(n454), .B(\init_mac/prod_ext [5]), .Y(
        n418) );
  sky130_fd_sc_hd__nand2_1 U470 ( .A(n419), .B(n418), .Y(n55) );
  sky130_fd_sc_hd__clkbuf_1 U471 ( .A(n420), .X(n424) );
  sky130_fd_sc_hd__nand2_1 U472 ( .A(n421), .B(n422), .Y(n423) );
  sky130_fd_sc_hd__xnor2_1 U473 ( .A(n424), .B(n423), .Y(n425) );
  sky130_fd_sc_hd__nand2_1 U474 ( .A(n425), .B(s1), .Y(n427) );
  sky130_fd_sc_hd__nand2_1 U475 ( .A(n454), .B(\init_mac/prod_ext [4]), .Y(
        n426) );
  sky130_fd_sc_hd__nand2_1 U476 ( .A(n427), .B(n426), .Y(n56) );
  sky130_fd_sc_hd__or2_0 U477 ( .A(n429), .B(n428), .X(n431) );
  sky130_fd_sc_hd__nand2_1 U478 ( .A(n431), .B(n430), .Y(n433) );
  sky130_fd_sc_hd__xor2_1 U479 ( .A(n433), .B(n432), .X(n434) );
  sky130_fd_sc_hd__nand2_1 U480 ( .A(n434), .B(s1), .Y(n436) );
  sky130_fd_sc_hd__nand2_1 U481 ( .A(n454), .B(\init_mac/prod_ext [3]), .Y(
        n435) );
  sky130_fd_sc_hd__nand2_1 U482 ( .A(n436), .B(n435), .Y(n57) );
  sky130_fd_sc_hd__inv_1 U483 ( .A(n437), .Y(n439) );
  sky130_fd_sc_hd__nand2_1 U484 ( .A(n439), .B(n438), .Y(n440) );
  sky130_fd_sc_hd__xor2_1 U485 ( .A(n440), .B(n446), .X(n441) );
  sky130_fd_sc_hd__nand2_1 U486 ( .A(n441), .B(s1), .Y(n443) );
  sky130_fd_sc_hd__nand2_1 U487 ( .A(n454), .B(\init_mac/prod_ext [2]), .Y(
        n442) );
  sky130_fd_sc_hd__nand2_1 U488 ( .A(n443), .B(n442), .Y(n58) );
  sky130_fd_sc_hd__or2_0 U489 ( .A(n445), .B(n444), .X(n447) );
  sky130_fd_sc_hd__and2_0 U490 ( .A(n447), .B(n446), .X(n448) );
  sky130_fd_sc_hd__nand2_1 U491 ( .A(n448), .B(s1), .Y(n450) );
  sky130_fd_sc_hd__nand2_1 U492 ( .A(n454), .B(\init_mac/prod_ext [1]), .Y(
        n449) );
  sky130_fd_sc_hd__nand2_1 U493 ( .A(n450), .B(n449), .Y(n59) );
  sky130_fd_sc_hd__nor2b_1 U494 ( .B_N(n452), .A(n451), .Y(n453) );
  sky130_fd_sc_hd__nand2_1 U495 ( .A(n453), .B(s1), .Y(n456) );
  sky130_fd_sc_hd__nand2_1 U496 ( .A(n454), .B(\init_mac/prod_ext [0]), .Y(
        n455) );
  sky130_fd_sc_hd__nand2_1 U497 ( .A(n456), .B(n455), .Y(n60) );
  sky130_fd_sc_hd__clkinv_1 U498 ( .A(mac_out[31]), .Y(n478) );
  sky130_fd_sc_hd__xor2_1 U499 ( .A(\init_mac/prod_ext_31 ), .B(
        \init_mac/acc_reg [31]), .X(n476) );
  sky130_fd_sc_hd__nor2_1 U500 ( .A(\init_mac/acc_reg [29]), .B(
        \init_mac/prod_ext_31 ), .Y(n548) );
  sky130_fd_sc_hd__nor2_1 U501 ( .A(\init_mac/acc_reg [27]), .B(
        \init_mac/prod_ext_31 ), .Y(n559) );
  sky130_fd_sc_hd__nor2_1 U502 ( .A(\init_mac/acc_reg [25]), .B(
        \init_mac/prod_ext_31 ), .Y(n570) );
  sky130_fd_sc_hd__nor2_1 U503 ( .A(\init_mac/acc_reg [23]), .B(
        \init_mac/prod_ext_31 ), .Y(n581) );
  sky130_fd_sc_hd__nor2_1 U504 ( .A(\init_mac/acc_reg [21]), .B(
        \init_mac/prod_ext_31 ), .Y(n592) );
  sky130_fd_sc_hd__nor2_1 U505 ( .A(\init_mac/acc_reg [19]), .B(
        \init_mac/prod_ext_31 ), .Y(n603) );
  sky130_fd_sc_hd__nor2_1 U506 ( .A(\init_mac/acc_reg [17]), .B(
        \init_mac/prod_ext_31 ), .Y(n614) );
  sky130_fd_sc_hd__nor2_1 U507 ( .A(\init_mac/acc_reg [15]), .B(
        \init_mac/prod_ext_31 ), .Y(n479) );
  sky130_fd_sc_hd__nor2_1 U508 ( .A(\init_mac/acc_reg [13]), .B(
        \init_mac/prod_ext [13]), .Y(n492) );
  sky130_fd_sc_hd__nor2_1 U509 ( .A(\init_mac/acc_reg [5]), .B(
        \init_mac/prod_ext [5]), .Y(n666) );
  sky130_fd_sc_hd__nor2_1 U510 ( .A(\init_mac/acc_reg [4]), .B(
        \init_mac/prod_ext [4]), .Y(n664) );
  sky130_fd_sc_hd__nor2_1 U511 ( .A(n666), .B(n664), .Y(n654) );
  sky130_fd_sc_hd__nor2_1 U512 ( .A(\init_mac/acc_reg [7]), .B(
        \init_mac/prod_ext [7]), .Y(n649) );
  sky130_fd_sc_hd__nor2_1 U513 ( .A(\init_mac/acc_reg [6]), .B(
        \init_mac/prod_ext [6]), .Y(n658) );
  sky130_fd_sc_hd__nor2_1 U514 ( .A(n649), .B(n658), .Y(n460) );
  sky130_fd_sc_hd__nand2_1 U515 ( .A(n654), .B(n460), .Y(n462) );
  sky130_fd_sc_hd__nor2_1 U516 ( .A(\init_mac/acc_reg [3]), .B(
        \init_mac/prod_ext [3]), .Y(n677) );
  sky130_fd_sc_hd__nor2_1 U517 ( .A(\init_mac/acc_reg [2]), .B(
        \init_mac/prod_ext [2]), .Y(n684) );
  sky130_fd_sc_hd__nor2_1 U518 ( .A(n677), .B(n684), .Y(n458) );
  sky130_fd_sc_hd__nand2_1 U519 ( .A(\init_mac/prod_ext [0]), .B(
        \init_mac/acc_reg [0]), .Y(n695) );
  sky130_fd_sc_hd__nor2_1 U520 ( .A(\init_mac/acc_reg [1]), .B(
        \init_mac/prod_ext [1]), .Y(n690) );
  sky130_fd_sc_hd__nand2_1 U521 ( .A(\init_mac/prod_ext [1]), .B(
        \init_mac/acc_reg [1]), .Y(n691) );
  sky130_fd_sc_hd__o21ai_1 U522 ( .A1(n695), .A2(n690), .B1(n691), .Y(n680) );
  sky130_fd_sc_hd__nand2_1 U523 ( .A(\init_mac/prod_ext [2]), .B(
        \init_mac/acc_reg [2]), .Y(n685) );
  sky130_fd_sc_hd__nand2_1 U524 ( .A(\init_mac/prod_ext [3]), .B(
        \init_mac/acc_reg [3]), .Y(n678) );
  sky130_fd_sc_hd__o21ai_1 U525 ( .A1(n685), .A2(n677), .B1(n678), .Y(n457) );
  sky130_fd_sc_hd__a21oi_1 U526 ( .A1(n458), .A2(n680), .B1(n457), .Y(n652) );
  sky130_fd_sc_hd__nand2_1 U527 ( .A(\init_mac/prod_ext [4]), .B(
        \init_mac/acc_reg [4]), .Y(n672) );
  sky130_fd_sc_hd__nand2_1 U528 ( .A(\init_mac/prod_ext [5]), .B(
        \init_mac/acc_reg [5]), .Y(n667) );
  sky130_fd_sc_hd__o21ai_1 U529 ( .A1(n672), .A2(n666), .B1(n667), .Y(n653) );
  sky130_fd_sc_hd__nand2_1 U530 ( .A(\init_mac/prod_ext [6]), .B(
        \init_mac/acc_reg [6]), .Y(n659) );
  sky130_fd_sc_hd__nand2_1 U531 ( .A(\init_mac/prod_ext [7]), .B(
        \init_mac/acc_reg [7]), .Y(n650) );
  sky130_fd_sc_hd__o21ai_1 U532 ( .A1(n659), .A2(n649), .B1(n650), .Y(n459) );
  sky130_fd_sc_hd__a21oi_1 U533 ( .A1(n460), .A2(n653), .B1(n459), .Y(n461) );
  sky130_fd_sc_hd__o21ai_1 U534 ( .A1(n462), .A2(n652), .B1(n461), .Y(n502) );
  sky130_fd_sc_hd__nor2_1 U535 ( .A(\init_mac/acc_reg [12]), .B(
        \init_mac/prod_ext [12]), .Y(n499) );
  sky130_fd_sc_hd__nor2_1 U536 ( .A(\init_mac/acc_reg [9]), .B(
        \init_mac/prod_ext [9]), .Y(n528) );
  sky130_fd_sc_hd__nor2_1 U537 ( .A(\init_mac/acc_reg [8]), .B(
        \init_mac/prod_ext [8]), .Y(n535) );
  sky130_fd_sc_hd__nor2_1 U538 ( .A(n528), .B(n535), .Y(n509) );
  sky130_fd_sc_hd__nor2_1 U539 ( .A(\init_mac/acc_reg [11]), .B(
        \init_mac/prod_ext [11]), .Y(n515) );
  sky130_fd_sc_hd__nor2_1 U540 ( .A(\init_mac/acc_reg [10]), .B(
        \init_mac/prod_ext [10]), .Y(n513) );
  sky130_fd_sc_hd__nor2_1 U541 ( .A(n515), .B(n513), .Y(n464) );
  sky130_fd_sc_hd__nand2_1 U542 ( .A(n509), .B(n464), .Y(n504) );
  sky130_fd_sc_hd__nor2_1 U543 ( .A(n499), .B(n504), .Y(n466) );
  sky130_fd_sc_hd__nand2_1 U544 ( .A(\init_mac/prod_ext [8]), .B(
        \init_mac/acc_reg [8]), .Y(n536) );
  sky130_fd_sc_hd__nand2_1 U545 ( .A(\init_mac/prod_ext [9]), .B(
        \init_mac/acc_reg [9]), .Y(n529) );
  sky130_fd_sc_hd__o21ai_1 U546 ( .A1(n536), .A2(n528), .B1(n529), .Y(n510) );
  sky130_fd_sc_hd__nand2_1 U547 ( .A(\init_mac/prod_ext [10]), .B(
        \init_mac/acc_reg [10]), .Y(n522) );
  sky130_fd_sc_hd__nand2_1 U548 ( .A(\init_mac/prod_ext [11]), .B(
        \init_mac/acc_reg [11]), .Y(n516) );
  sky130_fd_sc_hd__o21ai_1 U549 ( .A1(n522), .A2(n515), .B1(n516), .Y(n463) );
  sky130_fd_sc_hd__a21oi_1 U550 ( .A1(n464), .A2(n510), .B1(n463), .Y(n503) );
  sky130_fd_sc_hd__nand2_1 U551 ( .A(\init_mac/prod_ext [12]), .B(
        \init_mac/acc_reg [12]), .Y(n500) );
  sky130_fd_sc_hd__o21ai_1 U552 ( .A1(n499), .A2(n503), .B1(n500), .Y(n465) );
  sky130_fd_sc_hd__a21oi_1 U553 ( .A1(n502), .A2(n466), .B1(n465), .Y(n496) );
  sky130_fd_sc_hd__nand2_1 U554 ( .A(\init_mac/prod_ext [13]), .B(
        \init_mac/acc_reg [13]), .Y(n493) );
  sky130_fd_sc_hd__o21ai_1 U555 ( .A1(n492), .A2(n496), .B1(n493), .Y(n488) );
  sky130_fd_sc_hd__or2_0 U556 ( .A(\init_mac/acc_reg [14]), .B(
        \init_mac/prod_ext [14]), .X(n487) );
  sky130_fd_sc_hd__nand2_1 U557 ( .A(\init_mac/prod_ext [14]), .B(
        \init_mac/acc_reg [14]), .Y(n486) );
  sky130_fd_sc_hd__clkinv_1 U558 ( .A(n486), .Y(n467) );
  sky130_fd_sc_hd__a21oi_1 U559 ( .A1(n488), .A2(n487), .B1(n467), .Y(n483) );
  sky130_fd_sc_hd__nand2_1 U560 ( .A(\init_mac/prod_ext_31 ), .B(
        \init_mac/acc_reg [15]), .Y(n480) );
  sky130_fd_sc_hd__o21ai_1 U561 ( .A1(n479), .A2(n483), .B1(n480), .Y(n622) );
  sky130_fd_sc_hd__or2_0 U562 ( .A(\init_mac/acc_reg [16]), .B(
        \init_mac/prod_ext_31 ), .X(n621) );
  sky130_fd_sc_hd__nand2_1 U563 ( .A(\init_mac/prod_ext_31 ), .B(
        \init_mac/acc_reg [16]), .Y(n620) );
  sky130_fd_sc_hd__clkinv_1 U564 ( .A(n620), .Y(n468) );
  sky130_fd_sc_hd__a21oi_1 U565 ( .A1(n622), .A2(n621), .B1(n468), .Y(n618) );
  sky130_fd_sc_hd__nand2_1 U566 ( .A(\init_mac/prod_ext_31 ), .B(
        \init_mac/acc_reg [17]), .Y(n615) );
  sky130_fd_sc_hd__o21ai_1 U567 ( .A1(n614), .A2(n618), .B1(n615), .Y(n611) );
  sky130_fd_sc_hd__or2_0 U568 ( .A(\init_mac/acc_reg [18]), .B(
        \init_mac/prod_ext_31 ), .X(n610) );
  sky130_fd_sc_hd__nand2_1 U569 ( .A(\init_mac/prod_ext_31 ), .B(
        \init_mac/acc_reg [18]), .Y(n609) );
  sky130_fd_sc_hd__clkinv_1 U570 ( .A(n609), .Y(n469) );
  sky130_fd_sc_hd__a21oi_1 U571 ( .A1(n611), .A2(n610), .B1(n469), .Y(n607) );
  sky130_fd_sc_hd__nand2_1 U572 ( .A(\init_mac/prod_ext_31 ), .B(
        \init_mac/acc_reg [19]), .Y(n604) );
  sky130_fd_sc_hd__o21ai_1 U573 ( .A1(n603), .A2(n607), .B1(n604), .Y(n600) );
  sky130_fd_sc_hd__or2_0 U574 ( .A(\init_mac/acc_reg [20]), .B(
        \init_mac/prod_ext_31 ), .X(n599) );
  sky130_fd_sc_hd__nand2_1 U575 ( .A(\init_mac/prod_ext_31 ), .B(
        \init_mac/acc_reg [20]), .Y(n598) );
  sky130_fd_sc_hd__clkinv_1 U576 ( .A(n598), .Y(n470) );
  sky130_fd_sc_hd__a21oi_1 U577 ( .A1(n600), .A2(n599), .B1(n470), .Y(n596) );
  sky130_fd_sc_hd__nand2_1 U578 ( .A(\init_mac/prod_ext_31 ), .B(
        \init_mac/acc_reg [21]), .Y(n593) );
  sky130_fd_sc_hd__o21ai_1 U579 ( .A1(n592), .A2(n596), .B1(n593), .Y(n589) );
  sky130_fd_sc_hd__or2_0 U580 ( .A(\init_mac/acc_reg [22]), .B(
        \init_mac/prod_ext_31 ), .X(n588) );
  sky130_fd_sc_hd__nand2_1 U581 ( .A(\init_mac/prod_ext_31 ), .B(
        \init_mac/acc_reg [22]), .Y(n587) );
  sky130_fd_sc_hd__clkinv_1 U582 ( .A(n587), .Y(n471) );
  sky130_fd_sc_hd__a21oi_1 U583 ( .A1(n589), .A2(n588), .B1(n471), .Y(n585) );
  sky130_fd_sc_hd__nand2_1 U584 ( .A(\init_mac/prod_ext_31 ), .B(
        \init_mac/acc_reg [23]), .Y(n582) );
  sky130_fd_sc_hd__o21ai_1 U585 ( .A1(n581), .A2(n585), .B1(n582), .Y(n578) );
  sky130_fd_sc_hd__or2_0 U586 ( .A(\init_mac/acc_reg [24]), .B(
        \init_mac/prod_ext_31 ), .X(n577) );
  sky130_fd_sc_hd__nand2_1 U587 ( .A(\init_mac/prod_ext_31 ), .B(
        \init_mac/acc_reg [24]), .Y(n576) );
  sky130_fd_sc_hd__clkinv_1 U588 ( .A(n576), .Y(n472) );
  sky130_fd_sc_hd__a21oi_1 U589 ( .A1(n578), .A2(n577), .B1(n472), .Y(n574) );
  sky130_fd_sc_hd__nand2_1 U590 ( .A(\init_mac/prod_ext_31 ), .B(
        \init_mac/acc_reg [25]), .Y(n571) );
  sky130_fd_sc_hd__o21ai_1 U591 ( .A1(n570), .A2(n574), .B1(n571), .Y(n567) );
  sky130_fd_sc_hd__or2_0 U592 ( .A(\init_mac/acc_reg [26]), .B(
        \init_mac/prod_ext_31 ), .X(n566) );
  sky130_fd_sc_hd__nand2_1 U593 ( .A(\init_mac/prod_ext_31 ), .B(
        \init_mac/acc_reg [26]), .Y(n565) );
  sky130_fd_sc_hd__clkinv_1 U594 ( .A(n565), .Y(n473) );
  sky130_fd_sc_hd__a21oi_1 U595 ( .A1(n567), .A2(n566), .B1(n473), .Y(n563) );
  sky130_fd_sc_hd__nand2_1 U596 ( .A(\init_mac/prod_ext_31 ), .B(
        \init_mac/acc_reg [27]), .Y(n560) );
  sky130_fd_sc_hd__o21ai_1 U597 ( .A1(n559), .A2(n563), .B1(n560), .Y(n556) );
  sky130_fd_sc_hd__or2_0 U598 ( .A(\init_mac/acc_reg [28]), .B(
        \init_mac/prod_ext_31 ), .X(n555) );
  sky130_fd_sc_hd__nand2_1 U599 ( .A(\init_mac/prod_ext_31 ), .B(
        \init_mac/acc_reg [28]), .Y(n554) );
  sky130_fd_sc_hd__clkinv_1 U600 ( .A(n554), .Y(n474) );
  sky130_fd_sc_hd__a21oi_1 U601 ( .A1(n556), .A2(n555), .B1(n474), .Y(n552) );
  sky130_fd_sc_hd__nand2_1 U602 ( .A(\init_mac/prod_ext_31 ), .B(
        \init_mac/acc_reg [29]), .Y(n549) );
  sky130_fd_sc_hd__o21ai_1 U603 ( .A1(n548), .A2(n552), .B1(n549), .Y(n546) );
  sky130_fd_sc_hd__xor2_1 U604 ( .A(n476), .B(n475), .X(n543) );
  sky130_fd_sc_hd__nand2_1 U605 ( .A(n543), .B(\init_mac/N124 ), .Y(n477) );
  sky130_fd_sc_hd__o21ai_1 U606 ( .A1(\init_mac/N124 ), .A2(n478), .B1(n477), 
        .Y(n69) );
  sky130_fd_sc_hd__clkinv_1 U607 ( .A(mac_out[15]), .Y(n485) );
  sky130_fd_sc_hd__clkinv_1 U608 ( .A(n479), .Y(n481) );
  sky130_fd_sc_hd__nand2_1 U609 ( .A(n481), .B(n480), .Y(n482) );
  sky130_fd_sc_hd__xor2_1 U610 ( .A(n483), .B(n482), .X(n625) );
  sky130_fd_sc_hd__nand2_1 U611 ( .A(n625), .B(\init_mac/N124 ), .Y(n484) );
  sky130_fd_sc_hd__o21ai_1 U612 ( .A1(\init_mac/N124 ), .A2(n485), .B1(n484), 
        .Y(n61) );
  sky130_fd_sc_hd__clkinv_1 U613 ( .A(mac_out[14]), .Y(n491) );
  sky130_fd_sc_hd__nand2_1 U614 ( .A(n487), .B(n486), .Y(n489) );
  sky130_fd_sc_hd__xnor2_1 U615 ( .A(n489), .B(n488), .Y(n628) );
  sky130_fd_sc_hd__nand2_1 U616 ( .A(n628), .B(\init_mac/N124 ), .Y(n490) );
  sky130_fd_sc_hd__o21ai_1 U617 ( .A1(\init_mac/N124 ), .A2(n491), .B1(n490), 
        .Y(n62) );
  sky130_fd_sc_hd__clkinv_1 U618 ( .A(mac_out[13]), .Y(n498) );
  sky130_fd_sc_hd__clkinv_1 U619 ( .A(n492), .Y(n494) );
  sky130_fd_sc_hd__nand2_1 U620 ( .A(n494), .B(n493), .Y(n495) );
  sky130_fd_sc_hd__xor2_1 U621 ( .A(n496), .B(n495), .X(n631) );
  sky130_fd_sc_hd__nand2_1 U622 ( .A(n631), .B(\init_mac/N124 ), .Y(n497) );
  sky130_fd_sc_hd__o21ai_1 U623 ( .A1(\init_mac/N124 ), .A2(n498), .B1(n497), 
        .Y(n63) );
  sky130_fd_sc_hd__clkinv_1 U624 ( .A(mac_out[12]), .Y(n508) );
  sky130_fd_sc_hd__clkinv_1 U625 ( .A(n499), .Y(n501) );
  sky130_fd_sc_hd__nand2_1 U626 ( .A(n501), .B(n500), .Y(n506) );
  sky130_fd_sc_hd__clkinv_1 U627 ( .A(n502), .Y(n539) );
  sky130_fd_sc_hd__o21ai_1 U628 ( .A1(n504), .A2(n539), .B1(n503), .Y(n505) );
  sky130_fd_sc_hd__xnor2_1 U629 ( .A(n506), .B(n505), .Y(n634) );
  sky130_fd_sc_hd__nand2_1 U630 ( .A(n634), .B(\init_mac/N124 ), .Y(n507) );
  sky130_fd_sc_hd__o21ai_1 U631 ( .A1(\init_mac/N124 ), .A2(n508), .B1(n507), 
        .Y(n64) );
  sky130_fd_sc_hd__clkinv_1 U632 ( .A(mac_out[11]), .Y(n521) );
  sky130_fd_sc_hd__clkinv_1 U633 ( .A(n509), .Y(n512) );
  sky130_fd_sc_hd__clkinv_1 U634 ( .A(n510), .Y(n511) );
  sky130_fd_sc_hd__o21ai_1 U635 ( .A1(n512), .A2(n539), .B1(n511), .Y(n524) );
  sky130_fd_sc_hd__clkinv_1 U636 ( .A(n513), .Y(n523) );
  sky130_fd_sc_hd__clkinv_1 U637 ( .A(n522), .Y(n514) );
  sky130_fd_sc_hd__a21oi_1 U638 ( .A1(n524), .A2(n523), .B1(n514), .Y(n519) );
  sky130_fd_sc_hd__clkinv_1 U639 ( .A(n515), .Y(n517) );
  sky130_fd_sc_hd__nand2_1 U640 ( .A(n517), .B(n516), .Y(n518) );
  sky130_fd_sc_hd__xor2_1 U641 ( .A(n519), .B(n518), .X(n637) );
  sky130_fd_sc_hd__nand2_1 U642 ( .A(n637), .B(\init_mac/N124 ), .Y(n520) );
  sky130_fd_sc_hd__o21ai_1 U643 ( .A1(\init_mac/N124 ), .A2(n521), .B1(n520), 
        .Y(n65) );
  sky130_fd_sc_hd__clkinv_1 U644 ( .A(mac_out[10]), .Y(n527) );
  sky130_fd_sc_hd__nand2_1 U645 ( .A(n523), .B(n522), .Y(n525) );
  sky130_fd_sc_hd__xnor2_1 U646 ( .A(n525), .B(n524), .Y(n640) );
  sky130_fd_sc_hd__nand2_1 U647 ( .A(n640), .B(\init_mac/N124 ), .Y(n526) );
  sky130_fd_sc_hd__o21ai_1 U648 ( .A1(\init_mac/N124 ), .A2(n527), .B1(n526), 
        .Y(n66) );
  sky130_fd_sc_hd__clkinv_1 U649 ( .A(mac_out[9]), .Y(n534) );
  sky130_fd_sc_hd__clkinv_1 U650 ( .A(n528), .Y(n530) );
  sky130_fd_sc_hd__nand2_1 U651 ( .A(n530), .B(n529), .Y(n532) );
  sky130_fd_sc_hd__o21ai_1 U652 ( .A1(n535), .A2(n539), .B1(n536), .Y(n531) );
  sky130_fd_sc_hd__xnor2_1 U653 ( .A(n532), .B(n531), .Y(n643) );
  sky130_fd_sc_hd__nand2_1 U654 ( .A(n643), .B(\init_mac/N124 ), .Y(n533) );
  sky130_fd_sc_hd__o21ai_1 U655 ( .A1(\init_mac/N124 ), .A2(n534), .B1(n533), 
        .Y(n67) );
  sky130_fd_sc_hd__clkinv_1 U656 ( .A(mac_out[8]), .Y(n541) );
  sky130_fd_sc_hd__clkinv_1 U657 ( .A(n535), .Y(n537) );
  sky130_fd_sc_hd__nand2_1 U658 ( .A(n537), .B(n536), .Y(n538) );
  sky130_fd_sc_hd__xor2_1 U659 ( .A(n539), .B(n538), .X(n646) );
  sky130_fd_sc_hd__nand2_1 U660 ( .A(n646), .B(\init_mac/N124 ), .Y(n540) );
  sky130_fd_sc_hd__o21ai_1 U661 ( .A1(\init_mac/N124 ), .A2(n541), .B1(n540), 
        .Y(n68) );
  sky130_fd_sc_hd__nand2_1 U662 ( .A(n542), .B(\init_mac/do_acc ), .Y(n697) );
  sky130_fd_sc_hd__nand2_1 U663 ( .A(n697), .B(bias[31]), .Y(n544) );
  sky130_fd_sc_hd__o21ai_1 U664 ( .A1(n697), .A2(n545), .B1(n544), .Y(
        \init_mac/N156 ) );
  sky130_fd_sc_hd__fa_1 U665 ( .A(\init_mac/acc_reg [30]), .B(
        \init_mac/prod_ext_31 ), .CIN(n546), .COUT(n475), .SUM(n547) );
  sky130_fd_sc_hd__mux2_2 U666 ( .A0(n547), .A1(bias[30]), .S(n697), .X(
        \init_mac/N155 ) );
  sky130_fd_sc_hd__clkinv_1 U667 ( .A(n548), .Y(n550) );
  sky130_fd_sc_hd__nand2_1 U668 ( .A(n550), .B(n549), .Y(n551) );
  sky130_fd_sc_hd__xor2_1 U669 ( .A(n552), .B(n551), .X(n553) );
  sky130_fd_sc_hd__mux2_2 U670 ( .A0(n553), .A1(bias[29]), .S(n697), .X(
        \init_mac/N154 ) );
  sky130_fd_sc_hd__nand2_1 U671 ( .A(n555), .B(n554), .Y(n557) );
  sky130_fd_sc_hd__xnor2_1 U672 ( .A(n557), .B(n556), .Y(n558) );
  sky130_fd_sc_hd__mux2_2 U673 ( .A0(n558), .A1(bias[28]), .S(n697), .X(
        \init_mac/N153 ) );
  sky130_fd_sc_hd__clkinv_1 U674 ( .A(n559), .Y(n561) );
  sky130_fd_sc_hd__nand2_1 U675 ( .A(n561), .B(n560), .Y(n562) );
  sky130_fd_sc_hd__xor2_1 U676 ( .A(n563), .B(n562), .X(n564) );
  sky130_fd_sc_hd__mux2_2 U677 ( .A0(n564), .A1(bias[27]), .S(n697), .X(
        \init_mac/N152 ) );
  sky130_fd_sc_hd__nand2_1 U678 ( .A(n566), .B(n565), .Y(n568) );
  sky130_fd_sc_hd__xnor2_1 U679 ( .A(n568), .B(n567), .Y(n569) );
  sky130_fd_sc_hd__mux2_2 U680 ( .A0(n569), .A1(bias[26]), .S(n697), .X(
        \init_mac/N151 ) );
  sky130_fd_sc_hd__clkinv_1 U681 ( .A(n570), .Y(n572) );
  sky130_fd_sc_hd__nand2_1 U682 ( .A(n572), .B(n571), .Y(n573) );
  sky130_fd_sc_hd__xor2_1 U683 ( .A(n574), .B(n573), .X(n575) );
  sky130_fd_sc_hd__mux2_2 U684 ( .A0(n575), .A1(bias[25]), .S(n697), .X(
        \init_mac/N150 ) );
  sky130_fd_sc_hd__nand2_1 U685 ( .A(n577), .B(n576), .Y(n579) );
  sky130_fd_sc_hd__xnor2_1 U686 ( .A(n579), .B(n578), .Y(n580) );
  sky130_fd_sc_hd__mux2_2 U687 ( .A0(n580), .A1(bias[24]), .S(n697), .X(
        \init_mac/N149 ) );
  sky130_fd_sc_hd__clkinv_1 U688 ( .A(n581), .Y(n583) );
  sky130_fd_sc_hd__nand2_1 U689 ( .A(n583), .B(n582), .Y(n584) );
  sky130_fd_sc_hd__xor2_1 U690 ( .A(n585), .B(n584), .X(n586) );
  sky130_fd_sc_hd__mux2_2 U691 ( .A0(n586), .A1(bias[23]), .S(n697), .X(
        \init_mac/N148 ) );
  sky130_fd_sc_hd__nand2_1 U692 ( .A(n588), .B(n587), .Y(n590) );
  sky130_fd_sc_hd__xnor2_1 U693 ( .A(n590), .B(n589), .Y(n591) );
  sky130_fd_sc_hd__mux2_2 U694 ( .A0(n591), .A1(bias[22]), .S(n697), .X(
        \init_mac/N147 ) );
  sky130_fd_sc_hd__clkinv_1 U695 ( .A(n592), .Y(n594) );
  sky130_fd_sc_hd__nand2_1 U696 ( .A(n594), .B(n593), .Y(n595) );
  sky130_fd_sc_hd__xor2_1 U697 ( .A(n596), .B(n595), .X(n597) );
  sky130_fd_sc_hd__mux2_2 U698 ( .A0(n597), .A1(bias[21]), .S(n697), .X(
        \init_mac/N146 ) );
  sky130_fd_sc_hd__nand2_1 U699 ( .A(n599), .B(n598), .Y(n601) );
  sky130_fd_sc_hd__xnor2_1 U700 ( .A(n601), .B(n600), .Y(n602) );
  sky130_fd_sc_hd__mux2_2 U701 ( .A0(n602), .A1(bias[20]), .S(n697), .X(
        \init_mac/N145 ) );
  sky130_fd_sc_hd__clkinv_1 U702 ( .A(n603), .Y(n605) );
  sky130_fd_sc_hd__nand2_1 U703 ( .A(n605), .B(n604), .Y(n606) );
  sky130_fd_sc_hd__xor2_1 U704 ( .A(n607), .B(n606), .X(n608) );
  sky130_fd_sc_hd__mux2_2 U705 ( .A0(n608), .A1(bias[19]), .S(n697), .X(
        \init_mac/N144 ) );
  sky130_fd_sc_hd__nand2_1 U706 ( .A(n610), .B(n609), .Y(n612) );
  sky130_fd_sc_hd__xnor2_1 U707 ( .A(n612), .B(n611), .Y(n613) );
  sky130_fd_sc_hd__mux2_2 U708 ( .A0(n613), .A1(bias[18]), .S(n697), .X(
        \init_mac/N143 ) );
  sky130_fd_sc_hd__clkinv_1 U709 ( .A(n614), .Y(n616) );
  sky130_fd_sc_hd__nand2_1 U710 ( .A(n616), .B(n615), .Y(n617) );
  sky130_fd_sc_hd__xor2_1 U711 ( .A(n618), .B(n617), .X(n619) );
  sky130_fd_sc_hd__mux2_2 U712 ( .A0(n619), .A1(bias[17]), .S(n697), .X(
        \init_mac/N142 ) );
  sky130_fd_sc_hd__nand2_1 U713 ( .A(n621), .B(n620), .Y(n623) );
  sky130_fd_sc_hd__xnor2_1 U714 ( .A(n623), .B(n622), .Y(n624) );
  sky130_fd_sc_hd__mux2_2 U715 ( .A0(n624), .A1(bias[16]), .S(n697), .X(
        \init_mac/N141 ) );
  sky130_fd_sc_hd__clkinv_1 U716 ( .A(n625), .Y(n627) );
  sky130_fd_sc_hd__nand2_1 U717 ( .A(n697), .B(bias[15]), .Y(n626) );
  sky130_fd_sc_hd__o21ai_1 U718 ( .A1(n697), .A2(n627), .B1(n626), .Y(
        \init_mac/N140 ) );
  sky130_fd_sc_hd__clkinv_1 U719 ( .A(n628), .Y(n630) );
  sky130_fd_sc_hd__nand2_1 U720 ( .A(n697), .B(bias[14]), .Y(n629) );
  sky130_fd_sc_hd__o21ai_1 U721 ( .A1(n697), .A2(n630), .B1(n629), .Y(
        \init_mac/N139 ) );
  sky130_fd_sc_hd__clkinv_1 U722 ( .A(n631), .Y(n633) );
  sky130_fd_sc_hd__nand2_1 U723 ( .A(n697), .B(bias[13]), .Y(n632) );
  sky130_fd_sc_hd__o21ai_1 U724 ( .A1(n697), .A2(n633), .B1(n632), .Y(
        \init_mac/N138 ) );
  sky130_fd_sc_hd__clkinv_1 U725 ( .A(n634), .Y(n636) );
  sky130_fd_sc_hd__nand2_1 U726 ( .A(n697), .B(bias[12]), .Y(n635) );
  sky130_fd_sc_hd__o21ai_1 U727 ( .A1(n697), .A2(n636), .B1(n635), .Y(
        \init_mac/N137 ) );
  sky130_fd_sc_hd__clkinv_1 U728 ( .A(n637), .Y(n639) );
  sky130_fd_sc_hd__nand2_1 U729 ( .A(n697), .B(bias[11]), .Y(n638) );
  sky130_fd_sc_hd__o21ai_1 U730 ( .A1(n697), .A2(n639), .B1(n638), .Y(
        \init_mac/N136 ) );
  sky130_fd_sc_hd__clkinv_1 U731 ( .A(n640), .Y(n642) );
  sky130_fd_sc_hd__nand2_1 U732 ( .A(n697), .B(bias[10]), .Y(n641) );
  sky130_fd_sc_hd__o21ai_1 U733 ( .A1(n697), .A2(n642), .B1(n641), .Y(
        \init_mac/N135 ) );
  sky130_fd_sc_hd__clkinv_1 U734 ( .A(n643), .Y(n645) );
  sky130_fd_sc_hd__nand2_1 U735 ( .A(n697), .B(bias[9]), .Y(n644) );
  sky130_fd_sc_hd__o21ai_1 U736 ( .A1(n697), .A2(n645), .B1(n644), .Y(
        \init_mac/N134 ) );
  sky130_fd_sc_hd__clkinv_1 U737 ( .A(n646), .Y(n648) );
  sky130_fd_sc_hd__nand2_1 U738 ( .A(n697), .B(bias[8]), .Y(n647) );
  sky130_fd_sc_hd__o21ai_1 U739 ( .A1(n697), .A2(n648), .B1(n647), .Y(
        \init_mac/N133 ) );
  sky130_fd_sc_hd__clkinv_1 U740 ( .A(n649), .Y(n651) );
  sky130_fd_sc_hd__nand2_1 U741 ( .A(n651), .B(n650), .Y(n656) );
  sky130_fd_sc_hd__clkinv_1 U742 ( .A(n652), .Y(n674) );
  sky130_fd_sc_hd__a21oi_1 U743 ( .A1(n674), .A2(n654), .B1(n653), .Y(n662) );
  sky130_fd_sc_hd__o21ai_1 U744 ( .A1(n658), .A2(n662), .B1(n659), .Y(n655) );
  sky130_fd_sc_hd__xnor2_1 U745 ( .A(n656), .B(n655), .Y(n657) );
  sky130_fd_sc_hd__mux2_2 U746 ( .A0(n657), .A1(bias[7]), .S(n697), .X(
        \init_mac/N132 ) );
  sky130_fd_sc_hd__clkinv_1 U747 ( .A(n658), .Y(n660) );
  sky130_fd_sc_hd__nand2_1 U748 ( .A(n660), .B(n659), .Y(n661) );
  sky130_fd_sc_hd__xor2_1 U749 ( .A(n662), .B(n661), .X(n663) );
  sky130_fd_sc_hd__mux2_2 U750 ( .A0(n663), .A1(bias[6]), .S(n697), .X(
        \init_mac/N131 ) );
  sky130_fd_sc_hd__clkinv_1 U751 ( .A(n664), .Y(n673) );
  sky130_fd_sc_hd__clkinv_1 U752 ( .A(n672), .Y(n665) );
  sky130_fd_sc_hd__a21oi_1 U753 ( .A1(n674), .A2(n673), .B1(n665), .Y(n670) );
  sky130_fd_sc_hd__clkinv_1 U754 ( .A(n666), .Y(n668) );
  sky130_fd_sc_hd__nand2_1 U755 ( .A(n668), .B(n667), .Y(n669) );
  sky130_fd_sc_hd__xor2_1 U756 ( .A(n670), .B(n669), .X(n671) );
  sky130_fd_sc_hd__mux2_2 U757 ( .A0(n671), .A1(bias[5]), .S(n697), .X(
        \init_mac/N130 ) );
  sky130_fd_sc_hd__nand2_1 U758 ( .A(n673), .B(n672), .Y(n675) );
  sky130_fd_sc_hd__xnor2_1 U759 ( .A(n675), .B(n674), .Y(n676) );
  sky130_fd_sc_hd__mux2_2 U760 ( .A0(n676), .A1(bias[4]), .S(n697), .X(
        \init_mac/N129 ) );
  sky130_fd_sc_hd__clkinv_1 U761 ( .A(n677), .Y(n679) );
  sky130_fd_sc_hd__nand2_1 U762 ( .A(n679), .B(n678), .Y(n682) );
  sky130_fd_sc_hd__clkinv_1 U763 ( .A(n680), .Y(n688) );
  sky130_fd_sc_hd__o21ai_1 U764 ( .A1(n684), .A2(n688), .B1(n685), .Y(n681) );
  sky130_fd_sc_hd__xnor2_1 U765 ( .A(n682), .B(n681), .Y(n683) );
  sky130_fd_sc_hd__mux2_2 U766 ( .A0(n683), .A1(bias[3]), .S(n697), .X(
        \init_mac/N128 ) );
  sky130_fd_sc_hd__clkinv_1 U767 ( .A(n684), .Y(n686) );
  sky130_fd_sc_hd__nand2_1 U768 ( .A(n686), .B(n685), .Y(n687) );
  sky130_fd_sc_hd__xor2_1 U769 ( .A(n688), .B(n687), .X(n689) );
  sky130_fd_sc_hd__mux2_2 U770 ( .A0(n689), .A1(bias[2]), .S(n697), .X(
        \init_mac/N127 ) );
  sky130_fd_sc_hd__clkinv_1 U771 ( .A(n690), .Y(n692) );
  sky130_fd_sc_hd__nand2_1 U772 ( .A(n692), .B(n691), .Y(n693) );
  sky130_fd_sc_hd__xor2_1 U773 ( .A(n693), .B(n695), .X(n694) );
  sky130_fd_sc_hd__mux2_2 U774 ( .A0(n694), .A1(bias[1]), .S(n697), .X(
        \init_mac/N126 ) );
  sky130_fd_sc_hd__or2_0 U775 ( .A(\init_mac/acc_reg [0]), .B(
        \init_mac/prod_ext [0]), .X(n696) );
  sky130_fd_sc_hd__and2_0 U776 ( .A(n696), .B(n695), .X(n698) );
  sky130_fd_sc_hd__mux2_2 U777 ( .A0(n698), .A1(bias[0]), .S(n697), .X(
        \init_mac/N125 ) );
  sky130_fd_sc_hd__nor2b_1 U778 ( .B_N(mac_out[8]), .A(mac_out[31]), .Y(
        write_in[0]) );
  sky130_fd_sc_hd__nor2b_1 U779 ( .B_N(mac_out[9]), .A(mac_out[31]), .Y(
        write_in[1]) );
  sky130_fd_sc_hd__nor2b_1 U780 ( .B_N(mac_out[10]), .A(mac_out[31]), .Y(
        write_in[2]) );
  sky130_fd_sc_hd__nor2b_1 U781 ( .B_N(mac_out[11]), .A(mac_out[31]), .Y(
        write_in[3]) );
  sky130_fd_sc_hd__nor2b_1 U782 ( .B_N(mac_out[12]), .A(mac_out[31]), .Y(
        write_in[4]) );
  sky130_fd_sc_hd__nor2b_1 U783 ( .B_N(mac_out[13]), .A(mac_out[31]), .Y(
        write_in[5]) );
  sky130_fd_sc_hd__nor2b_1 U784 ( .B_N(mac_out[14]), .A(mac_out[31]), .Y(
        write_in[6]) );
  sky130_fd_sc_hd__nor2b_1 U785 ( .B_N(mac_out[15]), .A(mac_out[31]), .Y(
        write_in[7]) );
endmodule


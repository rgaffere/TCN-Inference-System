/////////////////////////////////////////////////////////////
// Created by: Synopsys DC Ultra(TM) in wire load mode
// Version   : X-2025.06-SP4
// Date      : Wed May 20 22:44:51 2026
/////////////////////////////////////////////////////////////


module conv_MAC ( clk, rst_n, valid_in, x, w, bias, valid_out, acc_out );
  input [7:0] x;
  input [7:0] w;
  input [31:0] bias;
  output [31:0] acc_out;
  input clk, rst_n, valid_in;
  output valid_out;
  wire   prod_ext_31, do_acc, N125, N126, N127, N128, N129, N130, N131, N132,
         N133, N134, N135, N136, N137, N138, N139, N140, N141, N142, N143,
         N144, N145, N146, N147, N148, N149, N150, N151, N152, N153, N154,
         N155, N156, n6, n7, n8, n9, n10, n11, n12, n13, n14, n15, n16, n17,
         n18, n19, n20, n21, n22, n23, n24, n25, n26, n27, n28, n29, n30, n31,
         n32, n33, n34, n35, n36, n37, n38, n39, n40, n41, n42, n43, n44, n45,
         n46, n47, n48, n49, n50, n51, n52, n53, n54, n55, n56, n57, n58, n59,
         n60, n61, n62, n63, n64, n65, n66, n67, n68, n69, n70, n71, n72, n73,
         n74, n75, n76, n77, n78, n79, n80, n81, n82, n83, n84, n85, n86, n87,
         n88, n89, n90, n91, n92, n93, n94, n95, n96, n97, n98, n99, n100,
         n101, n102, n103, n104, n105, n106, n107, n108, n109, n110, n111,
         n112, n113, n114, n115, n116, n117, n118, n119, n120, n121, n122,
         n123, n124, n125, n126, n127, n128, n129, n130, n131, n132, n133,
         n134, n135, n136, n137, n138, n139, n140, n141, n142, n143, n144,
         n145, n146, n147, n148, n149, n150, n151, n152, n153, n154, n155,
         n156, n157, n158, n159, n160, n161, n162, n163, n164, n165, n166,
         n167, n168, n169, n170, n171, n172, n173, n174, n175, n176, n177,
         n178, n179, n180, n181, n182, n183, n184, n185, n186, n187, n188,
         n189, n190, n191, n192, n193, n194, n195, n196, n197, n198, n199,
         n200, n201, n202, n203, n204, n205, n206, n207, n208, n209, n210,
         n211, n212, n213, n214, n215, n216, n217, n218, n219, n220, n221,
         n222, n223, n224, n225, n226, n227, n228, n229, n230, n231, n232,
         n233, n234, n235, n236, n237, n238, n239, n240, n241, n242, n243,
         n244, n245, n246, n247, n248, n249, n250, n251, n252, n253, n254,
         n255, n256, n257, n258, n259, n260, n261, n262, n263, n264, n265,
         n266, n267, n268, n269, n270, n271, n272, n273, n274, n275, n276,
         n277, n278, n279, n280, n281, n282, n283, n284, n285, n286, n287,
         n288, n289, n290, n291, n292, n293, n294, n295, n296, n297, n298,
         n299, n300, n301, n302, n303, n304, n305, n306, n307, n308, n309,
         n310, n311, n312, n313, n314, n315, n316, n317, n318, n319, n320,
         n321, n322, n323, n324, n325, n326, n327, n328, n329, n330, n331,
         n332, n333, n334, n335, n336, n337, n338, n339, n340, n341, n342,
         n343, n344, n345, n346, n347, n348, n349, n350, n351, n352, n353,
         n354, n355, n356, n357, n358, n359, n360;
  wire   [14:0] prod_ext;
  wire   [31:0] acc_reg;
  wire   [1:0] tap_idx;

  DFFR_X1 do_acc_reg ( .D(valid_in), .CK(clk), .RN(rst_n), .Q(do_acc) );
  DFFR_X1 \acc_out_reg[0]  ( .D(n55), .CK(clk), .RN(rst_n), .Q(acc_out[0]), 
        .QN(n360) );
  DFFR_X1 \acc_out_reg[1]  ( .D(n54), .CK(clk), .RN(rst_n), .Q(acc_out[1]) );
  DFFR_X1 \acc_out_reg[2]  ( .D(n53), .CK(clk), .RN(rst_n), .Q(acc_out[2]) );
  DFFR_X1 \acc_out_reg[3]  ( .D(n52), .CK(clk), .RN(rst_n), .Q(acc_out[3]) );
  DFFR_X1 \acc_out_reg[4]  ( .D(n51), .CK(clk), .RN(rst_n), .Q(acc_out[4]) );
  DFFR_X1 \acc_out_reg[5]  ( .D(n50), .CK(clk), .RN(rst_n), .Q(acc_out[5]) );
  DFFR_X1 \acc_out_reg[6]  ( .D(n49), .CK(clk), .RN(rst_n), .Q(acc_out[6]) );
  DFFR_X1 \acc_out_reg[7]  ( .D(n48), .CK(clk), .RN(rst_n), .Q(acc_out[7]) );
  DFFR_X1 \acc_out_reg[8]  ( .D(n47), .CK(clk), .RN(rst_n), .Q(acc_out[8]) );
  DFFR_X1 \acc_out_reg[9]  ( .D(n46), .CK(clk), .RN(rst_n), .Q(acc_out[9]) );
  DFFR_X1 \acc_out_reg[10]  ( .D(n45), .CK(clk), .RN(rst_n), .Q(acc_out[10])
         );
  DFFR_X1 \acc_out_reg[11]  ( .D(n44), .CK(clk), .RN(rst_n), .Q(acc_out[11])
         );
  DFFR_X1 \acc_out_reg[12]  ( .D(n43), .CK(clk), .RN(rst_n), .Q(acc_out[12])
         );
  DFFR_X1 \acc_out_reg[13]  ( .D(n42), .CK(clk), .RN(rst_n), .Q(acc_out[13])
         );
  DFFR_X1 \acc_out_reg[14]  ( .D(n41), .CK(clk), .RN(rst_n), .Q(acc_out[14])
         );
  DFFR_X1 \acc_out_reg[15]  ( .D(n40), .CK(clk), .RN(rst_n), .Q(acc_out[15])
         );
  DFFR_X1 \acc_out_reg[16]  ( .D(n39), .CK(clk), .RN(rst_n), .Q(acc_out[16])
         );
  DFFR_X1 \acc_out_reg[17]  ( .D(n38), .CK(clk), .RN(rst_n), .Q(acc_out[17])
         );
  DFFR_X1 \acc_out_reg[18]  ( .D(n37), .CK(clk), .RN(rst_n), .Q(acc_out[18])
         );
  DFFR_X1 \acc_out_reg[19]  ( .D(n36), .CK(clk), .RN(rst_n), .Q(acc_out[19])
         );
  DFFR_X1 \acc_out_reg[20]  ( .D(n35), .CK(clk), .RN(rst_n), .Q(acc_out[20])
         );
  DFFR_X1 \acc_out_reg[21]  ( .D(n34), .CK(clk), .RN(rst_n), .Q(acc_out[21])
         );
  DFFR_X1 \acc_out_reg[22]  ( .D(n33), .CK(clk), .RN(rst_n), .Q(acc_out[22])
         );
  DFFR_X1 \acc_out_reg[23]  ( .D(n32), .CK(clk), .RN(rst_n), .Q(acc_out[23])
         );
  DFFR_X1 \acc_out_reg[24]  ( .D(n31), .CK(clk), .RN(rst_n), .Q(acc_out[24])
         );
  DFFR_X1 \acc_out_reg[25]  ( .D(n30), .CK(clk), .RN(rst_n), .Q(acc_out[25])
         );
  DFFR_X1 \acc_out_reg[26]  ( .D(n29), .CK(clk), .RN(rst_n), .Q(acc_out[26])
         );
  DFFR_X1 \acc_out_reg[27]  ( .D(n28), .CK(clk), .RN(rst_n), .Q(acc_out[27])
         );
  DFFR_X1 \acc_out_reg[28]  ( .D(n27), .CK(clk), .RN(rst_n), .Q(acc_out[28])
         );
  DFFR_X1 \acc_out_reg[29]  ( .D(n26), .CK(clk), .RN(rst_n), .Q(acc_out[29])
         );
  DFFR_X1 \acc_out_reg[30]  ( .D(n25), .CK(clk), .RN(rst_n), .Q(acc_out[30])
         );
  DFFR_X1 \acc_out_reg[31]  ( .D(n24), .CK(clk), .RN(rst_n), .Q(acc_out[31])
         );
  DFFR_X1 valid_out_reg ( .D(n348), .CK(clk), .RN(rst_n), .Q(valid_out) );
  DFFR_X1 \acc_reg_reg[31]  ( .D(N156), .CK(clk), .RN(rst_n), .Q(acc_reg[31])
         );
  DFFR_X1 \acc_reg_reg[30]  ( .D(N155), .CK(clk), .RN(rst_n), .Q(acc_reg[30])
         );
  DFFR_X1 \acc_reg_reg[29]  ( .D(N154), .CK(clk), .RN(rst_n), .Q(acc_reg[29])
         );
  DFFR_X1 \acc_reg_reg[28]  ( .D(N153), .CK(clk), .RN(rst_n), .Q(acc_reg[28])
         );
  DFFR_X1 \acc_reg_reg[27]  ( .D(N152), .CK(clk), .RN(rst_n), .Q(acc_reg[27])
         );
  DFFR_X1 \acc_reg_reg[26]  ( .D(N151), .CK(clk), .RN(rst_n), .Q(acc_reg[26])
         );
  DFFR_X1 \acc_reg_reg[25]  ( .D(N150), .CK(clk), .RN(rst_n), .Q(acc_reg[25])
         );
  DFFR_X1 \acc_reg_reg[24]  ( .D(N149), .CK(clk), .RN(rst_n), .Q(acc_reg[24])
         );
  DFFR_X1 \acc_reg_reg[23]  ( .D(N148), .CK(clk), .RN(rst_n), .Q(acc_reg[23])
         );
  DFFR_X1 \acc_reg_reg[22]  ( .D(N147), .CK(clk), .RN(rst_n), .Q(acc_reg[22])
         );
  DFFR_X1 \acc_reg_reg[21]  ( .D(N146), .CK(clk), .RN(rst_n), .Q(acc_reg[21])
         );
  DFFR_X1 \acc_reg_reg[20]  ( .D(N145), .CK(clk), .RN(rst_n), .Q(acc_reg[20])
         );
  DFFR_X1 \acc_reg_reg[19]  ( .D(N144), .CK(clk), .RN(rst_n), .Q(acc_reg[19])
         );
  DFFR_X1 \acc_reg_reg[18]  ( .D(N143), .CK(clk), .RN(rst_n), .Q(acc_reg[18])
         );
  DFFR_X1 \acc_reg_reg[17]  ( .D(N142), .CK(clk), .RN(rst_n), .Q(acc_reg[17])
         );
  DFFR_X1 \acc_reg_reg[16]  ( .D(N141), .CK(clk), .RN(rst_n), .Q(acc_reg[16])
         );
  DFFR_X1 \acc_reg_reg[15]  ( .D(N140), .CK(clk), .RN(rst_n), .Q(acc_reg[15])
         );
  DFFR_X1 \acc_reg_reg[14]  ( .D(N139), .CK(clk), .RN(rst_n), .Q(acc_reg[14])
         );
  DFFR_X1 \acc_reg_reg[13]  ( .D(N138), .CK(clk), .RN(rst_n), .Q(acc_reg[13])
         );
  DFFR_X1 \acc_reg_reg[12]  ( .D(N137), .CK(clk), .RN(rst_n), .Q(acc_reg[12])
         );
  DFFR_X1 \acc_reg_reg[11]  ( .D(N136), .CK(clk), .RN(rst_n), .Q(acc_reg[11])
         );
  DFFR_X1 \acc_reg_reg[10]  ( .D(N135), .CK(clk), .RN(rst_n), .Q(acc_reg[10])
         );
  DFFR_X1 \acc_reg_reg[9]  ( .D(N134), .CK(clk), .RN(rst_n), .Q(acc_reg[9]) );
  DFFR_X1 \acc_reg_reg[8]  ( .D(N133), .CK(clk), .RN(rst_n), .Q(acc_reg[8]) );
  DFFR_X1 \acc_reg_reg[7]  ( .D(N132), .CK(clk), .RN(rst_n), .Q(acc_reg[7]) );
  DFFR_X1 \acc_reg_reg[6]  ( .D(N131), .CK(clk), .RN(rst_n), .Q(acc_reg[6]) );
  DFFR_X1 \acc_reg_reg[5]  ( .D(N130), .CK(clk), .RN(rst_n), .Q(acc_reg[5]) );
  DFFR_X1 \acc_reg_reg[4]  ( .D(N129), .CK(clk), .RN(rst_n), .Q(acc_reg[4]) );
  DFFR_X1 \acc_reg_reg[3]  ( .D(N128), .CK(clk), .RN(rst_n), .Q(acc_reg[3]) );
  DFFR_X1 \acc_reg_reg[2]  ( .D(N127), .CK(clk), .RN(rst_n), .Q(acc_reg[2]) );
  DFFR_X1 \acc_reg_reg[1]  ( .D(N126), .CK(clk), .RN(rst_n), .Q(acc_reg[1]) );
  DFFR_X1 \acc_reg_reg[0]  ( .D(N125), .CK(clk), .RN(rst_n), .Q(acc_reg[0]) );
  DFFR_X1 \tap_idx_reg[1]  ( .D(n23), .CK(clk), .RN(rst_n), .Q(tap_idx[1]), 
        .QN(n346) );
  DFFR_X1 \tap_idx_reg[0]  ( .D(n22), .CK(clk), .RN(rst_n), .Q(tap_idx[0]), 
        .QN(n345) );
  DFFR_X1 \prod_reg_reg[14]  ( .D(n20), .CK(clk), .RN(rst_n), .Q(prod_ext[14])
         );
  DFFR_X1 \prod_reg_reg[13]  ( .D(n19), .CK(clk), .RN(rst_n), .Q(prod_ext[13]), 
        .QN(n350) );
  DFFR_X1 \prod_reg_reg[12]  ( .D(n18), .CK(clk), .RN(rst_n), .Q(prod_ext[12]), 
        .QN(n351) );
  DFFR_X1 \prod_reg_reg[11]  ( .D(n17), .CK(clk), .RN(rst_n), .Q(prod_ext[11]), 
        .QN(n352) );
  DFFR_X1 \prod_reg_reg[10]  ( .D(n16), .CK(clk), .RN(rst_n), .Q(prod_ext[10])
         );
  DFFR_X1 \prod_reg_reg[9]  ( .D(n15), .CK(clk), .RN(rst_n), .Q(prod_ext[9]), 
        .QN(n353) );
  DFFR_X1 \prod_reg_reg[8]  ( .D(n14), .CK(clk), .RN(rst_n), .Q(prod_ext[8]), 
        .QN(n354) );
  DFFR_X1 \prod_reg_reg[7]  ( .D(n13), .CK(clk), .RN(rst_n), .Q(prod_ext[7]), 
        .QN(n355) );
  DFFR_X1 \prod_reg_reg[6]  ( .D(n12), .CK(clk), .RN(rst_n), .Q(prod_ext[6]), 
        .QN(n356) );
  DFFR_X1 \prod_reg_reg[5]  ( .D(n11), .CK(clk), .RN(rst_n), .Q(prod_ext[5]), 
        .QN(n357) );
  DFFR_X1 \prod_reg_reg[4]  ( .D(n10), .CK(clk), .RN(rst_n), .Q(prod_ext[4]), 
        .QN(n358) );
  DFFR_X1 \prod_reg_reg[3]  ( .D(n9), .CK(clk), .RN(rst_n), .Q(prod_ext[3]), 
        .QN(n359) );
  DFFR_X1 \prod_reg_reg[2]  ( .D(n8), .CK(clk), .RN(rst_n), .Q(prod_ext[2]) );
  DFFR_X1 \prod_reg_reg[1]  ( .D(n7), .CK(clk), .RN(rst_n), .Q(prod_ext[1]), 
        .QN(n349) );
  DFFR_X1 \prod_reg_reg[0]  ( .D(n6), .CK(clk), .RN(rst_n), .Q(prod_ext[0]) );
  DFFR_X2 \prod_reg_reg[15]  ( .D(n21), .CK(clk), .RN(rst_n), .Q(prod_ext_31), 
        .QN(n347) );
  AOI221_X2 U92 ( .B1(n131), .B2(n154), .C1(x[6]), .C2(x[7]), .A(n267), .ZN(
        n266) );
  AOI221_X2 U93 ( .B1(n133), .B2(n185), .C1(x[2]), .C2(x[3]), .A(n211), .ZN(
        n209) );
  AOI221_X4 U94 ( .B1(n137), .B2(n171), .C1(x[4]), .C2(x[5]), .A(n253), .ZN(
        n252) );
  INV_X1 U95 ( .A(valid_in), .ZN(n335) );
  CLKBUF_X1 U96 ( .A(n95), .Z(n90) );
  AND3_X1 U97 ( .A1(n345), .A2(do_acc), .A3(tap_idx[1]), .ZN(n348) );
  INV_X1 U98 ( .A(w[0]), .ZN(n184) );
  INV_X1 U99 ( .A(x[0]), .ZN(n168) );
  NOR2_X1 U100 ( .A1(n184), .A2(n168), .ZN(n56) );
  MUX2_X1 U101 ( .A(n56), .B(prod_ext[0]), .S(n335), .Z(n6) );
  NOR2_X1 U102 ( .A1(n345), .A2(tap_idx[1]), .ZN(n57) );
  MUX2_X1 U103 ( .A(tap_idx[1]), .B(n57), .S(do_acc), .Z(n23) );
  NAND2_X1 U104 ( .A1(prod_ext[0]), .A2(acc_reg[0]), .ZN(n59) );
  OAI21_X1 U105 ( .B1(prod_ext[0]), .B2(acc_reg[0]), .A(n59), .ZN(n96) );
  OAI21_X1 U106 ( .B1(tap_idx[0]), .B2(n346), .A(do_acc), .ZN(n95) );
  NAND2_X1 U107 ( .A1(n90), .A2(bias[0]), .ZN(n58) );
  OAI21_X1 U108 ( .B1(n96), .B2(n90), .A(n58), .ZN(N125) );
  INV_X1 U109 ( .A(n59), .ZN(n60) );
  MUX2_X1 U110 ( .A(n97), .B(bias[1]), .S(n90), .Z(N126) );
  FA_X1 U111 ( .A(prod_ext[1]), .B(acc_reg[1]), .CI(n60), .CO(n61), .S(n97) );
  MUX2_X1 U112 ( .A(n98), .B(bias[2]), .S(n90), .Z(N127) );
  FA_X1 U113 ( .A(prod_ext[2]), .B(acc_reg[2]), .CI(n61), .CO(n62), .S(n98) );
  MUX2_X1 U114 ( .A(n99), .B(bias[3]), .S(n90), .Z(N128) );
  FA_X1 U115 ( .A(prod_ext[3]), .B(acc_reg[3]), .CI(n62), .CO(n63), .S(n99) );
  MUX2_X1 U116 ( .A(n100), .B(bias[4]), .S(n90), .Z(N129) );
  FA_X1 U117 ( .A(prod_ext[4]), .B(acc_reg[4]), .CI(n63), .CO(n64), .S(n100)
         );
  MUX2_X1 U118 ( .A(n101), .B(bias[5]), .S(n90), .Z(N130) );
  FA_X1 U119 ( .A(prod_ext[5]), .B(acc_reg[5]), .CI(n64), .CO(n65), .S(n101)
         );
  MUX2_X1 U120 ( .A(n102), .B(bias[6]), .S(n90), .Z(N131) );
  FA_X1 U121 ( .A(prod_ext[6]), .B(acc_reg[6]), .CI(n65), .CO(n66), .S(n102)
         );
  MUX2_X1 U122 ( .A(n103), .B(bias[7]), .S(n90), .Z(N132) );
  FA_X1 U123 ( .A(prod_ext[7]), .B(acc_reg[7]), .CI(n66), .CO(n67), .S(n103)
         );
  MUX2_X1 U124 ( .A(n104), .B(bias[8]), .S(n95), .Z(N133) );
  FA_X1 U125 ( .A(prod_ext[8]), .B(acc_reg[8]), .CI(n67), .CO(n68), .S(n104)
         );
  MUX2_X1 U126 ( .A(n105), .B(bias[9]), .S(n95), .Z(N134) );
  FA_X1 U127 ( .A(prod_ext[9]), .B(acc_reg[9]), .CI(n68), .CO(n69), .S(n105)
         );
  MUX2_X1 U128 ( .A(n106), .B(bias[10]), .S(n95), .Z(N135) );
  FA_X1 U129 ( .A(prod_ext[10]), .B(acc_reg[10]), .CI(n69), .CO(n70), .S(n106)
         );
  MUX2_X1 U130 ( .A(n107), .B(bias[11]), .S(n95), .Z(N136) );
  FA_X1 U131 ( .A(prod_ext[11]), .B(acc_reg[11]), .CI(n70), .CO(n71), .S(n107)
         );
  MUX2_X1 U132 ( .A(n108), .B(bias[12]), .S(n95), .Z(N137) );
  FA_X1 U133 ( .A(prod_ext[12]), .B(acc_reg[12]), .CI(n71), .CO(n72), .S(n108)
         );
  MUX2_X1 U134 ( .A(n109), .B(bias[13]), .S(n95), .Z(N138) );
  FA_X1 U135 ( .A(prod_ext[13]), .B(acc_reg[13]), .CI(n72), .CO(n73), .S(n109)
         );
  MUX2_X1 U136 ( .A(n110), .B(bias[14]), .S(n95), .Z(N139) );
  FA_X1 U137 ( .A(prod_ext[14]), .B(acc_reg[14]), .CI(n73), .CO(n74), .S(n110)
         );
  MUX2_X1 U138 ( .A(n111), .B(bias[15]), .S(n90), .Z(N140) );
  FA_X1 U139 ( .A(acc_reg[15]), .B(prod_ext_31), .CI(n74), .CO(n75), .S(n111)
         );
  MUX2_X1 U140 ( .A(n112), .B(bias[16]), .S(n95), .Z(N141) );
  FA_X1 U141 ( .A(acc_reg[16]), .B(prod_ext_31), .CI(n75), .CO(n76), .S(n112)
         );
  MUX2_X1 U142 ( .A(n113), .B(bias[17]), .S(n90), .Z(N142) );
  FA_X1 U143 ( .A(acc_reg[17]), .B(prod_ext_31), .CI(n76), .CO(n77), .S(n113)
         );
  MUX2_X1 U144 ( .A(n115), .B(bias[18]), .S(n90), .Z(N143) );
  FA_X1 U145 ( .A(acc_reg[18]), .B(prod_ext_31), .CI(n77), .CO(n78), .S(n115)
         );
  MUX2_X1 U146 ( .A(n116), .B(bias[19]), .S(n95), .Z(N144) );
  FA_X1 U147 ( .A(acc_reg[19]), .B(prod_ext_31), .CI(n78), .CO(n79), .S(n116)
         );
  MUX2_X1 U148 ( .A(n117), .B(bias[20]), .S(n90), .Z(N145) );
  FA_X1 U149 ( .A(acc_reg[20]), .B(prod_ext_31), .CI(n79), .CO(n80), .S(n117)
         );
  MUX2_X1 U150 ( .A(n118), .B(bias[21]), .S(n90), .Z(N146) );
  FA_X1 U151 ( .A(acc_reg[21]), .B(prod_ext_31), .CI(n80), .CO(n81), .S(n118)
         );
  MUX2_X1 U152 ( .A(n119), .B(bias[22]), .S(n90), .Z(N147) );
  FA_X1 U153 ( .A(acc_reg[22]), .B(prod_ext_31), .CI(n81), .CO(n82), .S(n119)
         );
  MUX2_X1 U154 ( .A(n120), .B(bias[23]), .S(n90), .Z(N148) );
  FA_X1 U155 ( .A(acc_reg[23]), .B(prod_ext_31), .CI(n82), .CO(n83), .S(n120)
         );
  MUX2_X1 U156 ( .A(n121), .B(bias[24]), .S(n90), .Z(N149) );
  FA_X1 U157 ( .A(acc_reg[24]), .B(prod_ext_31), .CI(n83), .CO(n84), .S(n121)
         );
  MUX2_X1 U158 ( .A(n122), .B(bias[25]), .S(n90), .Z(N150) );
  FA_X1 U159 ( .A(acc_reg[25]), .B(prod_ext_31), .CI(n84), .CO(n85), .S(n122)
         );
  MUX2_X1 U160 ( .A(n123), .B(bias[26]), .S(n90), .Z(N151) );
  FA_X1 U161 ( .A(acc_reg[26]), .B(prod_ext_31), .CI(n85), .CO(n86), .S(n123)
         );
  MUX2_X1 U162 ( .A(n124), .B(bias[27]), .S(n90), .Z(N152) );
  FA_X1 U163 ( .A(acc_reg[27]), .B(prod_ext_31), .CI(n86), .CO(n87), .S(n124)
         );
  MUX2_X1 U164 ( .A(n125), .B(bias[28]), .S(n90), .Z(N153) );
  FA_X1 U165 ( .A(acc_reg[28]), .B(prod_ext_31), .CI(n87), .CO(n88), .S(n125)
         );
  MUX2_X1 U166 ( .A(n126), .B(bias[29]), .S(n90), .Z(N154) );
  NOR2_X1 U167 ( .A1(prod_ext_31), .A2(acc_reg[30]), .ZN(n91) );
  AOI21_X1 U168 ( .B1(acc_reg[30]), .B2(prod_ext_31), .A(n91), .ZN(n89) );
  FA_X1 U169 ( .A(acc_reg[29]), .B(prod_ext_31), .CI(n88), .CO(n93), .S(n126)
         );
  XOR2_X1 U170 ( .A(n89), .B(n93), .Z(n127) );
  MUX2_X1 U171 ( .A(n127), .B(bias[30]), .S(n90), .Z(N155) );
  INV_X1 U172 ( .A(n93), .ZN(n92) );
  AOI221_X1 U173 ( .B1(prod_ext_31), .B2(n93), .C1(acc_reg[30]), .C2(n92), .A(
        n91), .ZN(n94) );
  XOR2_X1 U174 ( .A(acc_reg[31]), .B(n94), .Z(n129) );
  MUX2_X1 U175 ( .A(n129), .B(bias[31]), .S(n95), .Z(N156) );
  INV_X1 U176 ( .A(n348), .ZN(n114) );
  AOI22_X1 U177 ( .A1(n348), .A2(n96), .B1(n360), .B2(n114), .ZN(n55) );
  MUX2_X1 U178 ( .A(n97), .B(acc_out[1]), .S(n114), .Z(n54) );
  MUX2_X1 U179 ( .A(n98), .B(acc_out[2]), .S(n114), .Z(n53) );
  MUX2_X1 U180 ( .A(n99), .B(acc_out[3]), .S(n114), .Z(n52) );
  MUX2_X1 U181 ( .A(n100), .B(acc_out[4]), .S(n114), .Z(n51) );
  MUX2_X1 U182 ( .A(n101), .B(acc_out[5]), .S(n114), .Z(n50) );
  MUX2_X1 U183 ( .A(n102), .B(acc_out[6]), .S(n114), .Z(n49) );
  MUX2_X1 U184 ( .A(n103), .B(acc_out[7]), .S(n114), .Z(n48) );
  MUX2_X1 U185 ( .A(n104), .B(acc_out[8]), .S(n114), .Z(n47) );
  MUX2_X1 U186 ( .A(n105), .B(acc_out[9]), .S(n114), .Z(n46) );
  MUX2_X1 U187 ( .A(n106), .B(acc_out[10]), .S(n114), .Z(n45) );
  MUX2_X1 U188 ( .A(n107), .B(acc_out[11]), .S(n114), .Z(n44) );
  MUX2_X1 U189 ( .A(n108), .B(acc_out[12]), .S(n114), .Z(n43) );
  INV_X1 U190 ( .A(n348), .ZN(n128) );
  MUX2_X1 U191 ( .A(n109), .B(acc_out[13]), .S(n128), .Z(n42) );
  MUX2_X1 U192 ( .A(n110), .B(acc_out[14]), .S(n114), .Z(n41) );
  MUX2_X1 U193 ( .A(n111), .B(acc_out[15]), .S(n128), .Z(n40) );
  MUX2_X1 U194 ( .A(n112), .B(acc_out[16]), .S(n114), .Z(n39) );
  MUX2_X1 U195 ( .A(n113), .B(acc_out[17]), .S(n128), .Z(n38) );
  MUX2_X1 U196 ( .A(n115), .B(acc_out[18]), .S(n114), .Z(n37) );
  MUX2_X1 U197 ( .A(n116), .B(acc_out[19]), .S(n128), .Z(n36) );
  MUX2_X1 U198 ( .A(n117), .B(acc_out[20]), .S(n128), .Z(n35) );
  MUX2_X1 U199 ( .A(n118), .B(acc_out[21]), .S(n128), .Z(n34) );
  MUX2_X1 U200 ( .A(n119), .B(acc_out[22]), .S(n128), .Z(n33) );
  MUX2_X1 U201 ( .A(n120), .B(acc_out[23]), .S(n128), .Z(n32) );
  MUX2_X1 U202 ( .A(n121), .B(acc_out[24]), .S(n128), .Z(n31) );
  MUX2_X1 U203 ( .A(n122), .B(acc_out[25]), .S(n128), .Z(n30) );
  MUX2_X1 U204 ( .A(n123), .B(acc_out[26]), .S(n128), .Z(n29) );
  MUX2_X1 U205 ( .A(n124), .B(acc_out[27]), .S(n128), .Z(n28) );
  MUX2_X1 U206 ( .A(n125), .B(acc_out[28]), .S(n128), .Z(n27) );
  MUX2_X1 U207 ( .A(n126), .B(acc_out[29]), .S(n128), .Z(n26) );
  MUX2_X1 U208 ( .A(n127), .B(acc_out[30]), .S(n128), .Z(n25) );
  MUX2_X1 U209 ( .A(n129), .B(acc_out[31]), .S(n128), .Z(n24) );
  NAND2_X1 U210 ( .A1(do_acc), .A2(n345), .ZN(n130) );
  OAI22_X1 U211 ( .A1(tap_idx[1]), .A2(n130), .B1(do_acc), .B2(n345), .ZN(n22)
         );
  INV_X1 U212 ( .A(w[6]), .ZN(n136) );
  INV_X1 U213 ( .A(x[7]), .ZN(n154) );
  AOI22_X1 U214 ( .A1(x[7]), .A2(n136), .B1(w[6]), .B2(n154), .ZN(n256) );
  INV_X1 U215 ( .A(x[6]), .ZN(n131) );
  INV_X1 U216 ( .A(x[5]), .ZN(n171) );
  OAI22_X1 U217 ( .A1(n171), .A2(x[6]), .B1(n131), .B2(x[5]), .ZN(n267) );
  INV_X1 U218 ( .A(n266), .ZN(n255) );
  INV_X1 U219 ( .A(w[7]), .ZN(n157) );
  AOI22_X1 U220 ( .A1(w[7]), .A2(x[7]), .B1(n154), .B2(n157), .ZN(n265) );
  NAND2_X1 U221 ( .A1(n267), .A2(n265), .ZN(n132) );
  OAI21_X1 U222 ( .B1(n256), .B2(n255), .A(n132), .ZN(n272) );
  INV_X1 U223 ( .A(x[1]), .ZN(n337) );
  INV_X1 U224 ( .A(x[2]), .ZN(n133) );
  OAI22_X1 U225 ( .A1(n337), .A2(n133), .B1(x[2]), .B2(x[1]), .ZN(n182) );
  INV_X1 U226 ( .A(n182), .ZN(n211) );
  INV_X1 U227 ( .A(x[3]), .ZN(n185) );
  AOI22_X1 U228 ( .A1(x[3]), .A2(w[7]), .B1(n157), .B2(n185), .ZN(n135) );
  AOI22_X1 U229 ( .A1(x[3]), .A2(w[6]), .B1(n136), .B2(n185), .ZN(n149) );
  AOI22_X1 U230 ( .A1(n211), .A2(n135), .B1(n209), .B2(n149), .ZN(n134) );
  INV_X1 U231 ( .A(n134), .ZN(n151) );
  OAI21_X1 U232 ( .B1(n211), .B2(n209), .A(n135), .ZN(n142) );
  INV_X1 U233 ( .A(x[4]), .ZN(n137) );
  AOI22_X1 U234 ( .A1(x[3]), .A2(x[4]), .B1(n137), .B2(n185), .ZN(n253) );
  OAI22_X1 U235 ( .A1(n171), .A2(w[6]), .B1(n136), .B2(x[5]), .ZN(n139) );
  INV_X1 U236 ( .A(w[5]), .ZN(n179) );
  OAI22_X1 U237 ( .A1(n171), .A2(w[5]), .B1(n179), .B2(x[5]), .ZN(n145) );
  AOI22_X1 U238 ( .A1(n253), .A2(n139), .B1(n252), .B2(n145), .ZN(n138) );
  INV_X1 U239 ( .A(n138), .ZN(n141) );
  AOI22_X1 U240 ( .A1(x[5]), .A2(w[7]), .B1(n157), .B2(n171), .ZN(n251) );
  AOI22_X1 U241 ( .A1(n253), .A2(n251), .B1(n252), .B2(n139), .ZN(n250) );
  INV_X1 U242 ( .A(n267), .ZN(n257) );
  AOI22_X1 U243 ( .A1(x[7]), .A2(n179), .B1(w[5]), .B2(n154), .ZN(n254) );
  INV_X1 U244 ( .A(w[4]), .ZN(n166) );
  OAI22_X1 U245 ( .A1(n154), .A2(w[4]), .B1(n166), .B2(x[7]), .ZN(n144) );
  INV_X1 U246 ( .A(n144), .ZN(n140) );
  OAI22_X1 U247 ( .A1(n257), .A2(n254), .B1(n255), .B2(n140), .ZN(n245) );
  FA_X1 U248 ( .A(n151), .B(n142), .CI(n141), .CO(n246), .S(n143) );
  INV_X1 U249 ( .A(n143), .ZN(n243) );
  INV_X1 U250 ( .A(w[3]), .ZN(n167) );
  AOI22_X1 U251 ( .A1(w[3]), .A2(x[7]), .B1(n154), .B2(n167), .ZN(n146) );
  AOI22_X1 U252 ( .A1(n267), .A2(n144), .B1(n266), .B2(n146), .ZN(n242) );
  AOI22_X1 U253 ( .A1(x[5]), .A2(w[4]), .B1(n166), .B2(n171), .ZN(n158) );
  AOI22_X1 U254 ( .A1(n253), .A2(n145), .B1(n252), .B2(n158), .ZN(n152) );
  INV_X1 U255 ( .A(w[2]), .ZN(n176) );
  AOI22_X1 U256 ( .A1(w[2]), .A2(x[7]), .B1(n154), .B2(n176), .ZN(n148) );
  AOI22_X1 U257 ( .A1(n267), .A2(n146), .B1(n266), .B2(n148), .ZN(n150) );
  INV_X1 U258 ( .A(n147), .ZN(n290) );
  INV_X1 U259 ( .A(w[1]), .ZN(n340) );
  AOI22_X1 U260 ( .A1(w[1]), .A2(x[7]), .B1(n154), .B2(n340), .ZN(n156) );
  AOI22_X1 U261 ( .A1(n267), .A2(n148), .B1(n266), .B2(n156), .ZN(n227) );
  AOI22_X1 U262 ( .A1(x[3]), .A2(w[5]), .B1(n179), .B2(n185), .ZN(n210) );
  AOI22_X1 U263 ( .A1(n211), .A2(n149), .B1(n209), .B2(n210), .ZN(n226) );
  NAND2_X1 U264 ( .A1(n227), .A2(n226), .ZN(n162) );
  FA_X1 U265 ( .A(n152), .B(n151), .CI(n150), .CO(n241), .S(n153) );
  INV_X1 U266 ( .A(n153), .ZN(n161) );
  AOI221_X1 U267 ( .B1(w[0]), .B2(x[7]), .C1(n184), .C2(n154), .A(n255), .ZN(
        n155) );
  AOI21_X1 U268 ( .B1(n156), .B2(n267), .A(n155), .ZN(n218) );
  OAI221_X1 U269 ( .B1(n266), .B2(n267), .C1(n266), .C2(n184), .A(x[7]), .ZN(
        n217) );
  NOR2_X1 U270 ( .A1(n218), .A2(n217), .ZN(n224) );
  NAND2_X1 U271 ( .A1(n337), .A2(x[0]), .ZN(n338) );
  INV_X1 U272 ( .A(n338), .ZN(n216) );
  AOI22_X1 U273 ( .A1(w[7]), .A2(n216), .B1(x[1]), .B2(n157), .ZN(n223) );
  OAI22_X1 U274 ( .A1(n167), .A2(x[5]), .B1(n171), .B2(w[3]), .ZN(n213) );
  AOI22_X1 U275 ( .A1(n253), .A2(n158), .B1(n252), .B2(n213), .ZN(n159) );
  INV_X1 U276 ( .A(n159), .ZN(n222) );
  FA_X1 U277 ( .A(n162), .B(n161), .CI(n160), .CO(n289), .S(n163) );
  INV_X1 U278 ( .A(n163), .ZN(n294) );
  AOI22_X1 U279 ( .A1(w[1]), .A2(x[5]), .B1(n171), .B2(n340), .ZN(n172) );
  OAI221_X1 U280 ( .B1(n184), .B2(n171), .C1(w[0]), .C2(x[5]), .A(n252), .ZN(
        n164) );
  INV_X1 U281 ( .A(n164), .ZN(n165) );
  AOI21_X1 U282 ( .B1(n253), .B2(n172), .A(n165), .ZN(n175) );
  OAI221_X1 U283 ( .B1(n252), .B2(n253), .C1(n252), .C2(n184), .A(x[5]), .ZN(
        n174) );
  NOR2_X1 U284 ( .A1(n175), .A2(n174), .ZN(n204) );
  AOI22_X1 U285 ( .A1(x[3]), .A2(w[4]), .B1(n166), .B2(n185), .ZN(n208) );
  AOI22_X1 U286 ( .A1(x[3]), .A2(w[3]), .B1(n167), .B2(n185), .ZN(n177) );
  AOI22_X1 U287 ( .A1(n211), .A2(n208), .B1(n209), .B2(n177), .ZN(n221) );
  NAND2_X1 U288 ( .A1(w[0]), .A2(n267), .ZN(n220) );
  NAND2_X1 U289 ( .A1(x[1]), .A2(x[0]), .ZN(n339) );
  NAND2_X1 U290 ( .A1(x[1]), .A2(n168), .ZN(n214) );
  OAI22_X1 U291 ( .A1(w[6]), .A2(n339), .B1(w[5]), .B2(n214), .ZN(n169) );
  AOI21_X1 U292 ( .B1(n216), .B2(w[6]), .A(n169), .ZN(n219) );
  INV_X1 U293 ( .A(n170), .ZN(n203) );
  AOI22_X1 U294 ( .A1(w[2]), .A2(x[5]), .B1(n171), .B2(n176), .ZN(n212) );
  AOI22_X1 U295 ( .A1(n253), .A2(n212), .B1(n252), .B2(n172), .ZN(n173) );
  INV_X1 U296 ( .A(n173), .ZN(n202) );
  AOI21_X1 U297 ( .B1(n175), .B2(n174), .A(n204), .ZN(n201) );
  OAI22_X1 U298 ( .A1(n185), .A2(w[2]), .B1(n176), .B2(x[3]), .ZN(n191) );
  AOI22_X1 U299 ( .A1(n211), .A2(n177), .B1(n209), .B2(n191), .ZN(n178) );
  INV_X1 U300 ( .A(n178), .ZN(n200) );
  OAI221_X1 U301 ( .B1(x[1]), .B2(w[5]), .C1(n337), .C2(n179), .A(x[0]), .ZN(
        n180) );
  OAI21_X1 U302 ( .B1(w[4]), .B2(n214), .A(n180), .ZN(n199) );
  OAI22_X1 U303 ( .A1(w[1]), .A2(n214), .B1(w[2]), .B2(n339), .ZN(n181) );
  AOI21_X1 U304 ( .B1(n216), .B2(w[2]), .A(n181), .ZN(n333) );
  AOI211_X1 U305 ( .C1(w[1]), .C2(x[0]), .A(w[0]), .B(n337), .ZN(n344) );
  AOI21_X1 U306 ( .B1(w[0]), .B2(n211), .A(n344), .ZN(n334) );
  NOR2_X1 U307 ( .A1(n333), .A2(n334), .ZN(n332) );
  AOI221_X1 U308 ( .B1(x[2]), .B2(n182), .C1(w[0]), .C2(n211), .A(n185), .ZN(
        n194) );
  NOR2_X1 U309 ( .A1(n332), .A2(n194), .ZN(n326) );
  AOI22_X1 U310 ( .A1(x[3]), .A2(w[1]), .B1(n340), .B2(n185), .ZN(n190) );
  INV_X1 U311 ( .A(n209), .ZN(n183) );
  AOI221_X1 U312 ( .B1(x[3]), .B2(w[0]), .C1(n185), .C2(n184), .A(n183), .ZN(
        n186) );
  AOI21_X1 U313 ( .B1(n211), .B2(n190), .A(n186), .ZN(n189) );
  OAI22_X1 U314 ( .A1(w[3]), .A2(n339), .B1(w[2]), .B2(n214), .ZN(n187) );
  AOI21_X1 U315 ( .B1(n216), .B2(w[3]), .A(n187), .ZN(n188) );
  XNOR2_X1 U316 ( .A(n189), .B(n188), .ZN(n330) );
  OR2_X1 U317 ( .A1(n189), .A2(n188), .ZN(n195) );
  OAI21_X1 U318 ( .B1(n326), .B2(n330), .A(n195), .ZN(n321) );
  AOI22_X1 U319 ( .A1(n211), .A2(n191), .B1(n209), .B2(n190), .ZN(n198) );
  NAND2_X1 U320 ( .A1(w[0]), .A2(n253), .ZN(n197) );
  OAI22_X1 U321 ( .A1(w[3]), .A2(n214), .B1(w[4]), .B2(n339), .ZN(n192) );
  AOI21_X1 U322 ( .B1(n216), .B2(w[4]), .A(n192), .ZN(n196) );
  INV_X1 U323 ( .A(n193), .ZN(n324) );
  NAND2_X1 U324 ( .A1(n332), .A2(n194), .ZN(n328) );
  NOR2_X1 U325 ( .A1(n195), .A2(n328), .ZN(n320) );
  AOI21_X1 U326 ( .B1(n321), .B2(n324), .A(n320), .ZN(n318) );
  FA_X1 U327 ( .A(n198), .B(n197), .CI(n196), .CO(n315), .S(n193) );
  FA_X1 U328 ( .A(n201), .B(n200), .CI(n199), .CO(n206), .S(n316) );
  INV_X1 U329 ( .A(n316), .ZN(n313) );
  AOI222_X1 U330 ( .A1(n318), .A2(n315), .B1(n318), .B2(n313), .C1(n315), .C2(
        n313), .ZN(n205) );
  NOR2_X1 U331 ( .A1(n206), .A2(n205), .ZN(n308) );
  INV_X1 U332 ( .A(n308), .ZN(n207) );
  FA_X1 U333 ( .A(n204), .B(n203), .CI(n202), .CO(n306), .S(n311) );
  AND2_X1 U334 ( .A1(n206), .A2(n205), .ZN(n309) );
  AOI21_X1 U335 ( .B1(n207), .B2(n311), .A(n309), .ZN(n304) );
  INV_X1 U336 ( .A(n304), .ZN(n301) );
  AOI22_X1 U337 ( .A1(n211), .A2(n210), .B1(n209), .B2(n208), .ZN(n230) );
  AOI22_X1 U338 ( .A1(n253), .A2(n213), .B1(n252), .B2(n212), .ZN(n229) );
  OAI22_X1 U339 ( .A1(w[7]), .A2(n339), .B1(w[6]), .B2(n214), .ZN(n215) );
  AOI21_X1 U340 ( .B1(n216), .B2(w[7]), .A(n215), .ZN(n228) );
  XNOR2_X1 U341 ( .A(n218), .B(n217), .ZN(n232) );
  FA_X1 U342 ( .A(n221), .B(n220), .CI(n219), .CO(n231), .S(n170) );
  INV_X1 U343 ( .A(n303), .ZN(n302) );
  AOI222_X1 U344 ( .A1(n306), .A2(n301), .B1(n306), .B2(n302), .C1(n301), .C2(
        n302), .ZN(n234) );
  INV_X1 U345 ( .A(n234), .ZN(n299) );
  FA_X1 U346 ( .A(n224), .B(n223), .CI(n222), .CO(n160), .S(n225) );
  INV_X1 U347 ( .A(n225), .ZN(n239) );
  XOR2_X1 U348 ( .A(n227), .B(n226), .Z(n238) );
  FA_X1 U349 ( .A(n230), .B(n229), .CI(n228), .CO(n237), .S(n233) );
  INV_X1 U350 ( .A(n235), .ZN(n297) );
  FA_X1 U351 ( .A(n233), .B(n232), .CI(n231), .CO(n296), .S(n303) );
  AOI21_X1 U352 ( .B1(n235), .B2(n234), .A(n296), .ZN(n236) );
  AOI21_X1 U353 ( .B1(n299), .B2(n297), .A(n236), .ZN(n293) );
  FA_X1 U354 ( .A(n239), .B(n238), .CI(n237), .CO(n292), .S(n235) );
  INV_X1 U355 ( .A(n240), .ZN(n288) );
  FA_X1 U356 ( .A(n243), .B(n242), .CI(n241), .CO(n244), .S(n147) );
  INV_X1 U357 ( .A(n244), .ZN(n248) );
  NAND2_X1 U358 ( .A1(n249), .A2(n248), .ZN(n282) );
  FA_X1 U359 ( .A(n246), .B(n250), .CI(n245), .CO(n259), .S(n247) );
  INV_X1 U360 ( .A(n247), .ZN(n286) );
  NOR2_X1 U361 ( .A1(n249), .A2(n248), .ZN(n284) );
  AOI21_X1 U362 ( .B1(n282), .B2(n286), .A(n284), .ZN(n258) );
  AND2_X1 U363 ( .A1(n259), .A2(n258), .ZN(n277) );
  INV_X1 U364 ( .A(n250), .ZN(n263) );
  OAI21_X1 U365 ( .B1(n253), .B2(n252), .A(n251), .ZN(n262) );
  OAI22_X1 U366 ( .A1(n257), .A2(n256), .B1(n255), .B2(n254), .ZN(n261) );
  NOR2_X1 U367 ( .A1(n259), .A2(n258), .ZN(n278) );
  INV_X1 U368 ( .A(n278), .ZN(n260) );
  OAI21_X1 U369 ( .B1(n277), .B2(n280), .A(n260), .ZN(n275) );
  FA_X1 U370 ( .A(n263), .B(n262), .CI(n261), .CO(n264), .S(n280) );
  INV_X1 U371 ( .A(n264), .ZN(n273) );
  AOI222_X1 U372 ( .A1(n275), .A2(n273), .B1(n275), .B2(n272), .C1(n273), .C2(
        n272), .ZN(n270) );
  OAI21_X1 U373 ( .B1(n267), .B2(n266), .A(n265), .ZN(n269) );
  AOI22_X1 U374 ( .A1(valid_in), .A2(n268), .B1(n347), .B2(n335), .ZN(n21) );
  FA_X1 U375 ( .A(n272), .B(n270), .CI(n269), .CO(n268), .S(n271) );
  MUX2_X1 U376 ( .A(n271), .B(prod_ext[14]), .S(n335), .Z(n20) );
  XOR2_X1 U377 ( .A(n273), .B(n272), .Z(n274) );
  XOR2_X1 U378 ( .A(n275), .B(n274), .Z(n276) );
  AOI22_X1 U379 ( .A1(valid_in), .A2(n276), .B1(n350), .B2(n335), .ZN(n19) );
  NOR2_X1 U380 ( .A1(n278), .A2(n277), .ZN(n279) );
  XNOR2_X1 U381 ( .A(n280), .B(n279), .ZN(n281) );
  AOI22_X1 U382 ( .A1(valid_in), .A2(n281), .B1(n351), .B2(n335), .ZN(n18) );
  INV_X1 U383 ( .A(n282), .ZN(n283) );
  NOR2_X1 U384 ( .A1(n284), .A2(n283), .ZN(n285) );
  XOR2_X1 U385 ( .A(n286), .B(n285), .Z(n287) );
  AOI22_X1 U386 ( .A1(valid_in), .A2(n287), .B1(n352), .B2(n335), .ZN(n17) );
  FA_X1 U387 ( .A(n290), .B(n289), .CI(n288), .CO(n249), .S(n291) );
  MUX2_X1 U388 ( .A(n291), .B(prod_ext[10]), .S(n335), .Z(n16) );
  FA_X1 U389 ( .A(n294), .B(n293), .CI(n292), .CO(n240), .S(n295) );
  AOI22_X1 U390 ( .A1(valid_in), .A2(n295), .B1(n353), .B2(n335), .ZN(n15) );
  XOR2_X1 U391 ( .A(n297), .B(n296), .Z(n298) );
  XOR2_X1 U392 ( .A(n299), .B(n298), .Z(n300) );
  AOI22_X1 U393 ( .A1(valid_in), .A2(n300), .B1(n354), .B2(n335), .ZN(n14) );
  AOI22_X1 U394 ( .A1(n304), .A2(n303), .B1(n302), .B2(n301), .ZN(n305) );
  XNOR2_X1 U395 ( .A(n306), .B(n305), .ZN(n307) );
  AOI22_X1 U396 ( .A1(valid_in), .A2(n307), .B1(n355), .B2(n335), .ZN(n13) );
  NOR2_X1 U397 ( .A1(n309), .A2(n308), .ZN(n310) );
  XNOR2_X1 U398 ( .A(n311), .B(n310), .ZN(n312) );
  AOI22_X1 U399 ( .A1(valid_in), .A2(n312), .B1(n356), .B2(n335), .ZN(n12) );
  INV_X1 U400 ( .A(n315), .ZN(n314) );
  AOI22_X1 U401 ( .A1(n316), .A2(n315), .B1(n314), .B2(n313), .ZN(n317) );
  XNOR2_X1 U402 ( .A(n318), .B(n317), .ZN(n319) );
  AOI22_X1 U403 ( .A1(valid_in), .A2(n319), .B1(n357), .B2(n335), .ZN(n11) );
  INV_X1 U404 ( .A(n320), .ZN(n322) );
  NAND2_X1 U405 ( .A1(n322), .A2(n321), .ZN(n323) );
  XOR2_X1 U406 ( .A(n324), .B(n323), .Z(n325) );
  AOI22_X1 U407 ( .A1(valid_in), .A2(n325), .B1(n358), .B2(n335), .ZN(n10) );
  INV_X1 U408 ( .A(n326), .ZN(n327) );
  NAND2_X1 U409 ( .A1(n328), .A2(n327), .ZN(n329) );
  XNOR2_X1 U410 ( .A(n330), .B(n329), .ZN(n331) );
  AOI22_X1 U411 ( .A1(valid_in), .A2(n331), .B1(n359), .B2(n335), .ZN(n9) );
  AOI21_X1 U412 ( .B1(n334), .B2(n333), .A(n332), .ZN(n336) );
  MUX2_X1 U413 ( .A(n336), .B(prod_ext[2]), .S(n335), .Z(n8) );
  AOI21_X1 U414 ( .B1(w[0]), .B2(x[0]), .A(n337), .ZN(n342) );
  AOI22_X1 U415 ( .A1(n340), .A2(n339), .B1(n338), .B2(w[1]), .ZN(n341) );
  OAI21_X1 U416 ( .B1(n342), .B2(n341), .A(valid_in), .ZN(n343) );
  OAI22_X1 U417 ( .A1(n344), .A2(n343), .B1(valid_in), .B2(n349), .ZN(n7) );
endmodule


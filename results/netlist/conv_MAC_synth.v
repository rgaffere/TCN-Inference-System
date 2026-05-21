/////////////////////////////////////////////////////////////
// Created by: Synopsys DC Ultra(TM) in wire load mode
// Version   : X-2025.06-SP4
// Date      : Wed May 20 22:57:47 2026
/////////////////////////////////////////////////////////////


module conv_MAC ( clk, rst_n, valid_in, x, w, bias, valid_out, acc_out );
  input [7:0] x;
  input [7:0] w;
  input [31:0] bias;
  output [31:0] acc_out;
  input clk, rst_n, valid_in;
  output valid_out;
  wire   do_acc, N125, N126, N127, N128, N129, N130, N131, N132, N133, N134,
         N135, N136, N137, N138, N139, N140, N141, N142, N143, N144, N145,
         N146, N147, N148, N149, N150, N151, N152, N153, N154, N155, N156, n6,
         n7, n8, n9, n10, n11, n12, n13, n14, n15, n16, n17, n18, n19, n20,
         n21, n22, n23, n24, n25, n26, n27, n28, n29, n30, n31, n32, n33, n34,
         n35, n36, n37, n38, n39, n40, n41, n42, n43, n44, n45, n46, n47, n48,
         n49, n50, n51, n52, n53, n54, n55, \intadd_1/A[2] , \intadd_1/A[1] ,
         \intadd_1/A[0] , \intadd_1/B[2] , \intadd_1/B[1] , \intadd_1/B[0] ,
         \intadd_1/CI , \intadd_1/SUM[2] , \intadd_1/SUM[1] ,
         \intadd_1/SUM[0] , \intadd_1/n3 , \intadd_1/n2 , \intadd_1/n1 ,
         \intadd_2/A[2] , \intadd_2/A[1] , \intadd_2/A[0] , \intadd_2/B[2] ,
         \intadd_2/B[1] , \intadd_2/B[0] , \intadd_2/CI , \intadd_2/SUM[2] ,
         \intadd_2/SUM[1] , \intadd_2/SUM[0] , \intadd_2/n3 , \intadd_2/n2 ,
         \intadd_2/n1 , n56, n57, n58, n59, n60, n61, n62, n63, n64, n65, n66,
         n67, n68, n69, n70, n71, n72, n73, n74, n75, n76, n77, n78, n79, n80,
         n81, n82, n83, n84, n85, n86, n87, n88, n89, n90, n91, n92, n93, n94,
         n95, n96, n97, n98, n99, n100, n101, n102, n103, n104, n105, n106,
         n107, n108, n109, n110, n111, n112, n113, n114, n115, n116, n117,
         n118, n119, n120, n121, n122, n123, n124, n125, n126, n127, n128,
         n129, n130, n131, n132, n133, n134, n135, n136, n137, n138, n139,
         n140, n141, n142, n143, n144, n145, n146, n147, n148, n149, n150,
         n151, n152, n153, n154, n155, n156, n157, n158, n159, n160, n161,
         n162, n163, n164, n165, n166, n167, n168, n169, n170, n171, n172,
         n173, n174, n175, n176, n177, n178, n179, n180, n181, n182, n183,
         n184, n185, n186, n187, n188, n189, n190, n191, n192, n193, n194,
         n195, n196, n197, n198, n199, n200, n201, n202, n203, n204, n205,
         n206, n207, n208, n209, n210, n211, n212, n213, n214, n215, n216,
         n217, n218, n219, n220, n221, n222, n223, n224, n225, n226, n227,
         n228, n229, n230, n231, n232, n233, n234, n235, n236, n237, n238,
         n239, n240, n241, n242, n243, n244, n245, n246, n247, n248, n249,
         n250, n251, n252, n253, n254, n255, n256, n257, n258, n259, n260,
         n261, n262, n263, n264, n265, n266, n267, n268, n269, n270, n271,
         n272, n273, n274, n275, n276, n277, n278, n279, n280, n281, n282,
         n283, n284, n285, n286, n287, n288, n289, n290, n291, n292, n293,
         n294, n295, n296, n297, n298, n299, n300, n301, n302, n303, n304,
         n305, n306, n307, n308, n309, n310, n311, n312, n313, n314, n315,
         n316, n317, n318, n319, n320, n321, n322, n323, n324, n325, n326,
         n327, n328, n329, n330, n331, n332, n333, n334, n335, n336, n337,
         n338, n339, n340, n341, n342, n343, n344, n345, n346, n347, n348,
         n349, n350, n351, n352, n353, n354, n355, n356, n357, n358, n359,
         n360, n361, n362, n363, n364, n365, n366, n367, n368, n369, n370,
         n371, n372, n373, n374, n375, n376, n377, n378, n379, n380, n381,
         n382, n383, n384, n385, n386, n387, n388, n389, n390, n391, n392,
         n393, n394, n395, n396, n397, n398, n399, n400, n401, n402, n403,
         n404, n405, n406, n407, n408, n409, n410, n411, n412, n413, n414,
         n415, n416, n417, n418, n419, n420, n421, n422, n423, n424, n425,
         n426, n427, n428, n429, n430, n431, n432, n433, n434, n435, n436,
         n437, n438, n439, n440, n441, n442, n443, n444, n445, n446, n447,
         n448, n449, n450, n451, n452, n453, n454, n455, n456, n457, n458,
         n459, n460, n461;
  wire   [14:0] prod_ext;
  wire   [31:0] acc_reg;
  wire   [1:0] tap_idx;

  DFFR_X1 do_acc_reg ( .D(valid_in), .CK(clk), .RN(rst_n), .Q(do_acc) );
  DFFR_X1 \acc_out_reg[0]  ( .D(n55), .CK(clk), .RN(rst_n), .Q(acc_out[0]), 
        .QN(n458) );
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
  DFFR_X1 valid_out_reg ( .D(n443), .CK(clk), .RN(rst_n), .Q(valid_out) );
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
  DFFR_X1 \acc_reg_reg[0]  ( .D(N125), .CK(clk), .RN(rst_n), .Q(acc_reg[0]), 
        .QN(n460) );
  DFFR_X1 \tap_idx_reg[1]  ( .D(n23), .CK(clk), .RN(rst_n), .Q(tap_idx[1]), 
        .QN(n445) );
  DFFR_X1 \tap_idx_reg[0]  ( .D(n22), .CK(clk), .RN(rst_n), .Q(tap_idx[0]), 
        .QN(n446) );
  DFFR_X1 \prod_reg_reg[14]  ( .D(n20), .CK(clk), .RN(rst_n), .Q(prod_ext[14])
         );
  DFFR_X1 \prod_reg_reg[13]  ( .D(n19), .CK(clk), .RN(rst_n), .Q(prod_ext[13]), 
        .QN(n448) );
  DFFR_X1 \prod_reg_reg[12]  ( .D(n18), .CK(clk), .RN(rst_n), .Q(prod_ext[12]), 
        .QN(n449) );
  DFFR_X1 \prod_reg_reg[11]  ( .D(n17), .CK(clk), .RN(rst_n), .Q(prod_ext[11]), 
        .QN(n450) );
  DFFR_X1 \prod_reg_reg[10]  ( .D(n16), .CK(clk), .RN(rst_n), .Q(prod_ext[10])
         );
  DFFR_X1 \prod_reg_reg[9]  ( .D(n15), .CK(clk), .RN(rst_n), .Q(prod_ext[9]), 
        .QN(n451) );
  DFFR_X1 \prod_reg_reg[8]  ( .D(n14), .CK(clk), .RN(rst_n), .Q(prod_ext[8]), 
        .QN(n452) );
  DFFR_X1 \prod_reg_reg[7]  ( .D(n13), .CK(clk), .RN(rst_n), .Q(prod_ext[7]), 
        .QN(n453) );
  DFFR_X1 \prod_reg_reg[6]  ( .D(n12), .CK(clk), .RN(rst_n), .Q(prod_ext[6]), 
        .QN(n454) );
  DFFR_X1 \prod_reg_reg[5]  ( .D(n11), .CK(clk), .RN(rst_n), .Q(prod_ext[5]), 
        .QN(n457) );
  DFFR_X1 \prod_reg_reg[4]  ( .D(n10), .CK(clk), .RN(rst_n), .Q(prod_ext[4]), 
        .QN(n455) );
  DFFR_X1 \prod_reg_reg[3]  ( .D(n9), .CK(clk), .RN(rst_n), .Q(prod_ext[3]), 
        .QN(n456) );
  DFFR_X1 \prod_reg_reg[2]  ( .D(n8), .CK(clk), .RN(rst_n), .Q(prod_ext[2]) );
  DFFR_X1 \prod_reg_reg[1]  ( .D(n7), .CK(clk), .RN(rst_n), .Q(prod_ext[1]), 
        .QN(n459) );
  DFFR_X1 \prod_reg_reg[0]  ( .D(n6), .CK(clk), .RN(rst_n), .Q(prod_ext[0]), 
        .QN(n461) );
  FA_X1 \intadd_1/U4  ( .A(\intadd_1/A[0] ), .B(\intadd_1/B[0] ), .CI(
        \intadd_1/CI ), .CO(\intadd_1/n3 ), .S(\intadd_1/SUM[0] ) );
  FA_X1 \intadd_1/U3  ( .A(\intadd_1/A[1] ), .B(\intadd_1/B[1] ), .CI(
        \intadd_1/n3 ), .CO(\intadd_1/n2 ), .S(\intadd_1/SUM[1] ) );
  FA_X1 \intadd_1/U2  ( .A(\intadd_1/A[2] ), .B(\intadd_1/B[2] ), .CI(
        \intadd_1/n2 ), .CO(\intadd_1/n1 ), .S(\intadd_1/SUM[2] ) );
  FA_X1 \intadd_2/U4  ( .A(\intadd_2/A[0] ), .B(\intadd_2/B[0] ), .CI(
        \intadd_2/CI ), .CO(\intadd_2/n3 ), .S(\intadd_2/SUM[0] ) );
  FA_X1 \intadd_2/U3  ( .A(\intadd_2/A[1] ), .B(\intadd_2/B[1] ), .CI(
        \intadd_2/n3 ), .CO(\intadd_2/n2 ), .S(\intadd_2/SUM[1] ) );
  FA_X1 \intadd_2/U2  ( .A(\intadd_2/A[2] ), .B(\intadd_2/B[2] ), .CI(
        \intadd_2/n2 ), .CO(\intadd_2/n1 ), .S(\intadd_2/SUM[2] ) );
  DFFR_X2 \prod_reg_reg[15]  ( .D(n21), .CK(clk), .RN(rst_n), .Q(n444), .QN(
        n447) );
  AOI221_X2 U92 ( .B1(n71), .B2(n344), .C1(x[6]), .C2(x[7]), .A(n373), .ZN(
        n372) );
  AOI221_X2 U93 ( .B1(n69), .B2(n300), .C1(x[2]), .C2(x[3]), .A(n308), .ZN(
        n306) );
  AOI221_X4 U94 ( .B1(n66), .B2(n341), .C1(x[4]), .C2(x[5]), .A(n359), .ZN(
        n358) );
  NOR2_X1 U95 ( .A1(acc_reg[29]), .A2(n444), .ZN(n56) );
  AND2_X1 U96 ( .A1(n227), .A2(n58), .ZN(n57) );
  AND2_X1 U97 ( .A1(n223), .A2(n221), .ZN(n58) );
  AND2_X1 U98 ( .A1(n204), .A2(n65), .ZN(n59) );
  NOR2_X1 U99 ( .A1(acc_reg[5]), .A2(prod_ext[5]), .ZN(n60) );
  AND3_X1 U100 ( .A1(n92), .A2(n91), .A3(n90), .ZN(n165) );
  AND2_X1 U101 ( .A1(n191), .A2(n193), .ZN(n61) );
  AND3_X1 U102 ( .A1(n123), .A2(n122), .A3(n121), .ZN(n187) );
  NOR2_X1 U103 ( .A1(acc_reg[16]), .A2(n444), .ZN(n62) );
  AND2_X1 U104 ( .A1(n209), .A2(n59), .ZN(n63) );
  AND2_X1 U105 ( .A1(n197), .A2(n61), .ZN(n64) );
  AND3_X1 U106 ( .A1(n57), .A2(n147), .A3(n146), .ZN(n230) );
  AND2_X1 U107 ( .A1(n201), .A2(n64), .ZN(n65) );
  NAND2_X1 U108 ( .A1(n444), .A2(acc_reg[29]), .ZN(n148) );
  NAND2_X1 U109 ( .A1(n444), .A2(acc_reg[16]), .ZN(n124) );
  OAI21_X1 U110 ( .B1(n165), .B2(n60), .A(n93), .ZN(n167) );
  OAI21_X1 U111 ( .B1(n56), .B2(n230), .A(n148), .ZN(n233) );
  OAI21_X1 U112 ( .B1(n62), .B2(n187), .A(n124), .ZN(n190) );
  INV_X1 U113 ( .A(valid_in), .ZN(n433) );
  OAI21_X1 U114 ( .B1(tap_idx[0]), .B2(n445), .A(do_acc), .ZN(n212) );
  CLKBUF_X1 U115 ( .A(n212), .Z(n235) );
  XNOR2_X1 U116 ( .A(n187), .B(n188), .ZN(n252) );
  XNOR2_X1 U117 ( .A(n165), .B(n166), .ZN(n241) );
  AND3_X1 U118 ( .A1(n446), .A2(do_acc), .A3(tap_idx[1]), .ZN(n443) );
  INV_X1 U119 ( .A(\intadd_1/n1 ), .ZN(\intadd_2/B[2] ) );
  INV_X1 U120 ( .A(\intadd_2/SUM[0] ), .ZN(\intadd_1/A[1] ) );
  INV_X1 U121 ( .A(\intadd_2/SUM[1] ), .ZN(\intadd_1/A[2] ) );
  INV_X1 U122 ( .A(x[4]), .ZN(n66) );
  INV_X1 U123 ( .A(x[3]), .ZN(n300) );
  AOI22_X1 U124 ( .A1(x[3]), .A2(x[4]), .B1(n66), .B2(n300), .ZN(n359) );
  INV_X1 U125 ( .A(w[4]), .ZN(n274) );
  INV_X1 U126 ( .A(x[5]), .ZN(n341) );
  AOI22_X1 U127 ( .A1(x[5]), .A2(w[4]), .B1(n274), .B2(n341), .ZN(n68) );
  INV_X1 U128 ( .A(w[3]), .ZN(n283) );
  OAI22_X1 U129 ( .A1(n283), .A2(x[5]), .B1(n341), .B2(w[3]), .ZN(n276) );
  AOI22_X1 U130 ( .A1(n359), .A2(n68), .B1(n358), .B2(n276), .ZN(n67) );
  INV_X1 U131 ( .A(n67), .ZN(\intadd_2/CI ) );
  INV_X1 U132 ( .A(w[5]), .ZN(n345) );
  OAI22_X1 U133 ( .A1(n341), .A2(w[5]), .B1(n345), .B2(x[5]), .ZN(n74) );
  AOI22_X1 U134 ( .A1(n359), .A2(n74), .B1(n358), .B2(n68), .ZN(n79) );
  INV_X1 U135 ( .A(x[1]), .ZN(n435) );
  INV_X1 U136 ( .A(x[2]), .ZN(n69) );
  OAI22_X1 U137 ( .A1(n435), .A2(n69), .B1(x[2]), .B2(x[1]), .ZN(n297) );
  INV_X1 U138 ( .A(n297), .ZN(n308) );
  INV_X1 U139 ( .A(w[7]), .ZN(n342) );
  AOI22_X1 U140 ( .A1(x[3]), .A2(w[7]), .B1(n342), .B2(n300), .ZN(n73) );
  INV_X1 U141 ( .A(w[6]), .ZN(n336) );
  AOI22_X1 U142 ( .A1(x[3]), .A2(w[6]), .B1(n336), .B2(n300), .ZN(n271) );
  AOI22_X1 U143 ( .A1(n308), .A2(n73), .B1(n306), .B2(n271), .ZN(n70) );
  INV_X1 U144 ( .A(n70), .ZN(n340) );
  INV_X1 U145 ( .A(x[6]), .ZN(n71) );
  OAI22_X1 U146 ( .A1(n341), .A2(x[6]), .B1(n71), .B2(x[5]), .ZN(n373) );
  INV_X1 U147 ( .A(x[7]), .ZN(n344) );
  AOI22_X1 U148 ( .A1(w[3]), .A2(x[7]), .B1(n344), .B2(n283), .ZN(n77) );
  INV_X1 U149 ( .A(w[2]), .ZN(n292) );
  AOI22_X1 U150 ( .A1(w[2]), .A2(x[7]), .B1(n344), .B2(n292), .ZN(n270) );
  AOI22_X1 U151 ( .A1(n373), .A2(n77), .B1(n372), .B2(n270), .ZN(n78) );
  INV_X1 U152 ( .A(n72), .ZN(\intadd_2/B[1] ) );
  OAI21_X1 U153 ( .B1(n308), .B2(n306), .A(n73), .ZN(n339) );
  OAI22_X1 U154 ( .A1(n341), .A2(w[6]), .B1(n336), .B2(x[5]), .ZN(n343) );
  AOI22_X1 U155 ( .A1(n359), .A2(n343), .B1(n358), .B2(n74), .ZN(n75) );
  INV_X1 U156 ( .A(n75), .ZN(n338) );
  INV_X1 U157 ( .A(n76), .ZN(n350) );
  OAI22_X1 U158 ( .A1(n344), .A2(w[4]), .B1(n274), .B2(x[7]), .ZN(n346) );
  AOI22_X1 U159 ( .A1(n373), .A2(n346), .B1(n372), .B2(n77), .ZN(n349) );
  FA_X1 U160 ( .A(n79), .B(n340), .CI(n78), .CO(n348), .S(n72) );
  INV_X1 U161 ( .A(n80), .ZN(\intadd_2/A[2] ) );
  NAND2_X1 U162 ( .A1(n444), .A2(acc_reg[27]), .ZN(n227) );
  NAND2_X1 U163 ( .A1(n444), .A2(acc_reg[26]), .ZN(n223) );
  NAND2_X1 U164 ( .A1(n444), .A2(acc_reg[24]), .ZN(n138) );
  NAND2_X1 U165 ( .A1(n444), .A2(acc_reg[21]), .ZN(n209) );
  NAND2_X1 U166 ( .A1(n444), .A2(acc_reg[20]), .ZN(n204) );
  NAND2_X1 U167 ( .A1(n444), .A2(acc_reg[19]), .ZN(n201) );
  NAND2_X1 U168 ( .A1(n444), .A2(acc_reg[18]), .ZN(n197) );
  NAND2_X1 U169 ( .A1(n444), .A2(acc_reg[17]), .ZN(n191) );
  NOR2_X1 U170 ( .A1(n461), .A2(n460), .ZN(n155) );
  NAND2_X1 U171 ( .A1(n155), .A2(prod_ext[1]), .ZN(n83) );
  NAND2_X1 U172 ( .A1(n155), .A2(acc_reg[1]), .ZN(n82) );
  NAND2_X1 U173 ( .A1(prod_ext[1]), .A2(acc_reg[1]), .ZN(n81) );
  NAND3_X1 U174 ( .A1(n83), .A2(n82), .A3(n81), .ZN(n160) );
  NAND2_X1 U175 ( .A1(n160), .A2(prod_ext[2]), .ZN(n86) );
  NAND2_X1 U176 ( .A1(n160), .A2(acc_reg[2]), .ZN(n85) );
  NAND2_X1 U177 ( .A1(prod_ext[2]), .A2(acc_reg[2]), .ZN(n84) );
  NAND3_X1 U178 ( .A1(n86), .A2(n85), .A3(n84), .ZN(n161) );
  NAND2_X1 U179 ( .A1(n161), .A2(prod_ext[3]), .ZN(n89) );
  NAND2_X1 U180 ( .A1(n161), .A2(acc_reg[3]), .ZN(n88) );
  NAND2_X1 U181 ( .A1(prod_ext[3]), .A2(acc_reg[3]), .ZN(n87) );
  NAND3_X1 U182 ( .A1(n89), .A2(n88), .A3(n87), .ZN(n164) );
  NAND2_X1 U183 ( .A1(n164), .A2(acc_reg[4]), .ZN(n92) );
  NAND2_X1 U184 ( .A1(n164), .A2(prod_ext[4]), .ZN(n91) );
  NAND2_X1 U185 ( .A1(prod_ext[4]), .A2(acc_reg[4]), .ZN(n90) );
  NAND2_X1 U186 ( .A1(acc_reg[5]), .A2(prod_ext[5]), .ZN(n93) );
  NAND2_X1 U187 ( .A1(n167), .A2(prod_ext[6]), .ZN(n96) );
  NAND2_X1 U188 ( .A1(n167), .A2(acc_reg[6]), .ZN(n95) );
  NAND2_X1 U189 ( .A1(prod_ext[6]), .A2(acc_reg[6]), .ZN(n94) );
  NAND3_X1 U190 ( .A1(n96), .A2(n95), .A3(n94), .ZN(n170) );
  NAND2_X1 U191 ( .A1(n170), .A2(prod_ext[7]), .ZN(n99) );
  NAND2_X1 U192 ( .A1(n170), .A2(acc_reg[7]), .ZN(n98) );
  NAND2_X1 U193 ( .A1(prod_ext[7]), .A2(acc_reg[7]), .ZN(n97) );
  NAND3_X1 U194 ( .A1(n99), .A2(n98), .A3(n97), .ZN(n172) );
  NAND2_X1 U195 ( .A1(n172), .A2(prod_ext[8]), .ZN(n102) );
  NAND2_X1 U196 ( .A1(n172), .A2(acc_reg[8]), .ZN(n101) );
  NAND2_X1 U197 ( .A1(prod_ext[8]), .A2(acc_reg[8]), .ZN(n100) );
  NAND3_X1 U198 ( .A1(n102), .A2(n101), .A3(n100), .ZN(n173) );
  NAND2_X1 U199 ( .A1(n173), .A2(prod_ext[9]), .ZN(n105) );
  NAND2_X1 U200 ( .A1(n173), .A2(acc_reg[9]), .ZN(n104) );
  NAND2_X1 U201 ( .A1(prod_ext[9]), .A2(acc_reg[9]), .ZN(n103) );
  NAND3_X1 U202 ( .A1(n105), .A2(n104), .A3(n103), .ZN(n175) );
  NAND2_X1 U203 ( .A1(n175), .A2(prod_ext[10]), .ZN(n108) );
  NAND2_X1 U204 ( .A1(n175), .A2(acc_reg[10]), .ZN(n107) );
  NAND2_X1 U205 ( .A1(prod_ext[10]), .A2(acc_reg[10]), .ZN(n106) );
  NAND3_X1 U206 ( .A1(n108), .A2(n107), .A3(n106), .ZN(n177) );
  NAND2_X1 U207 ( .A1(n177), .A2(prod_ext[11]), .ZN(n111) );
  NAND2_X1 U208 ( .A1(n177), .A2(acc_reg[11]), .ZN(n110) );
  NAND2_X1 U209 ( .A1(prod_ext[11]), .A2(acc_reg[11]), .ZN(n109) );
  NAND3_X1 U210 ( .A1(n111), .A2(n110), .A3(n109), .ZN(n179) );
  NAND2_X1 U211 ( .A1(n179), .A2(prod_ext[12]), .ZN(n114) );
  NAND2_X1 U212 ( .A1(n179), .A2(acc_reg[12]), .ZN(n113) );
  NAND2_X1 U213 ( .A1(prod_ext[12]), .A2(acc_reg[12]), .ZN(n112) );
  NAND3_X1 U214 ( .A1(n114), .A2(n113), .A3(n112), .ZN(n181) );
  NAND2_X1 U215 ( .A1(n181), .A2(prod_ext[13]), .ZN(n117) );
  NAND2_X1 U216 ( .A1(n181), .A2(acc_reg[13]), .ZN(n116) );
  NAND2_X1 U217 ( .A1(prod_ext[13]), .A2(acc_reg[13]), .ZN(n115) );
  NAND3_X1 U218 ( .A1(n117), .A2(n116), .A3(n115), .ZN(n184) );
  NAND2_X1 U219 ( .A1(n184), .A2(acc_reg[14]), .ZN(n120) );
  NAND2_X1 U220 ( .A1(n184), .A2(prod_ext[14]), .ZN(n119) );
  NAND2_X1 U221 ( .A1(prod_ext[14]), .A2(acc_reg[14]), .ZN(n118) );
  NAND3_X1 U222 ( .A1(n120), .A2(n119), .A3(n118), .ZN(n185) );
  NAND2_X1 U223 ( .A1(n185), .A2(acc_reg[15]), .ZN(n123) );
  NAND2_X1 U224 ( .A1(n185), .A2(n444), .ZN(n122) );
  NAND2_X1 U225 ( .A1(n444), .A2(acc_reg[15]), .ZN(n121) );
  NAND2_X1 U226 ( .A1(n190), .A2(n444), .ZN(n193) );
  NAND2_X1 U227 ( .A1(n190), .A2(acc_reg[17]), .ZN(n192) );
  INV_X1 U228 ( .A(n192), .ZN(n125) );
  NAND2_X1 U229 ( .A1(n125), .A2(acc_reg[18]), .ZN(n196) );
  INV_X1 U230 ( .A(n196), .ZN(n126) );
  NAND2_X1 U231 ( .A1(n126), .A2(acc_reg[19]), .ZN(n200) );
  INV_X1 U232 ( .A(n200), .ZN(n127) );
  NAND2_X1 U233 ( .A1(n127), .A2(acc_reg[20]), .ZN(n205) );
  INV_X1 U234 ( .A(n205), .ZN(n128) );
  NAND2_X1 U235 ( .A1(n128), .A2(acc_reg[21]), .ZN(n208) );
  INV_X1 U236 ( .A(n208), .ZN(n129) );
  NAND2_X1 U237 ( .A1(n129), .A2(acc_reg[22]), .ZN(n131) );
  NAND2_X1 U238 ( .A1(n444), .A2(acc_reg[22]), .ZN(n130) );
  NAND3_X1 U239 ( .A1(n63), .A2(n131), .A3(n130), .ZN(n213) );
  NAND2_X1 U240 ( .A1(n213), .A2(n444), .ZN(n134) );
  INV_X1 U241 ( .A(n131), .ZN(n132) );
  NAND2_X1 U242 ( .A1(n132), .A2(acc_reg[23]), .ZN(n135) );
  NAND2_X1 U243 ( .A1(n444), .A2(acc_reg[23]), .ZN(n133) );
  NAND3_X1 U244 ( .A1(n134), .A2(n135), .A3(n133), .ZN(n215) );
  NAND2_X1 U245 ( .A1(n215), .A2(n444), .ZN(n137) );
  INV_X1 U246 ( .A(n135), .ZN(n136) );
  NAND2_X1 U247 ( .A1(n136), .A2(acc_reg[24]), .ZN(n139) );
  NAND3_X1 U248 ( .A1(n138), .A2(n137), .A3(n139), .ZN(n217) );
  NAND2_X1 U249 ( .A1(n217), .A2(n444), .ZN(n143) );
  INV_X1 U250 ( .A(n139), .ZN(n140) );
  NAND2_X1 U251 ( .A1(acc_reg[25]), .A2(n140), .ZN(n142) );
  NAND2_X1 U252 ( .A1(n444), .A2(acc_reg[25]), .ZN(n141) );
  NAND3_X1 U253 ( .A1(n143), .A2(n142), .A3(n141), .ZN(n219) );
  NAND2_X1 U254 ( .A1(n219), .A2(n444), .ZN(n221) );
  NAND2_X1 U255 ( .A1(n219), .A2(acc_reg[26]), .ZN(n222) );
  INV_X1 U256 ( .A(n222), .ZN(n144) );
  NAND2_X1 U257 ( .A1(acc_reg[27]), .A2(n144), .ZN(n226) );
  INV_X1 U258 ( .A(n226), .ZN(n145) );
  NAND2_X1 U259 ( .A1(n145), .A2(acc_reg[28]), .ZN(n147) );
  NAND2_X1 U260 ( .A1(n444), .A2(acc_reg[28]), .ZN(n146) );
  INV_X1 U261 ( .A(n233), .ZN(n149) );
  NOR2_X1 U262 ( .A1(n444), .A2(acc_reg[30]), .ZN(n232) );
  AOI221_X1 U263 ( .B1(n444), .B2(n233), .C1(acc_reg[30]), .C2(n149), .A(n232), 
        .ZN(n150) );
  XOR2_X1 U264 ( .A(n150), .B(acc_reg[31]), .Z(n152) );
  MUX2_X1 U265 ( .A(acc_out[31]), .B(n152), .S(n443), .Z(n24) );
  INV_X1 U266 ( .A(n235), .ZN(n151) );
  MUX2_X1 U267 ( .A(bias[31]), .B(n152), .S(n151), .Z(N156) );
  INV_X1 U268 ( .A(w[0]), .ZN(n299) );
  INV_X1 U269 ( .A(x[0]), .ZN(n277) );
  NOR2_X1 U270 ( .A1(n299), .A2(n277), .ZN(n153) );
  MUX2_X1 U271 ( .A(n153), .B(prod_ext[0]), .S(n433), .Z(n6) );
  NOR2_X1 U272 ( .A1(n446), .A2(tap_idx[1]), .ZN(n154) );
  MUX2_X1 U273 ( .A(tap_idx[1]), .B(n154), .S(do_acc), .Z(n23) );
  INV_X1 U274 ( .A(n155), .ZN(n156) );
  OAI21_X1 U275 ( .B1(prod_ext[0]), .B2(acc_reg[0]), .A(n156), .ZN(n236) );
  NAND2_X1 U276 ( .A1(n212), .A2(bias[0]), .ZN(n157) );
  OAI21_X1 U277 ( .B1(n236), .B2(n212), .A(n157), .ZN(N125) );
  XOR2_X1 U278 ( .A(prod_ext[1]), .B(acc_reg[1]), .Z(n158) );
  XOR2_X1 U279 ( .A(n155), .B(n158), .Z(n237) );
  MUX2_X1 U280 ( .A(n237), .B(bias[1]), .S(n235), .Z(N126) );
  XOR2_X1 U281 ( .A(prod_ext[2]), .B(acc_reg[2]), .Z(n159) );
  XOR2_X1 U282 ( .A(n160), .B(n159), .Z(n238) );
  MUX2_X1 U283 ( .A(n238), .B(bias[2]), .S(n235), .Z(N127) );
  XOR2_X1 U284 ( .A(prod_ext[3]), .B(acc_reg[3]), .Z(n162) );
  XOR2_X1 U285 ( .A(n161), .B(n162), .Z(n239) );
  MUX2_X1 U286 ( .A(n239), .B(bias[3]), .S(n212), .Z(N128) );
  XOR2_X1 U287 ( .A(prod_ext[4]), .B(acc_reg[4]), .Z(n163) );
  XOR2_X1 U288 ( .A(n164), .B(n163), .Z(n240) );
  MUX2_X1 U289 ( .A(n240), .B(bias[4]), .S(n212), .Z(N129) );
  XOR2_X1 U290 ( .A(acc_reg[5]), .B(prod_ext[5]), .Z(n166) );
  MUX2_X1 U291 ( .A(n241), .B(bias[5]), .S(n212), .Z(N130) );
  XOR2_X1 U292 ( .A(prod_ext[6]), .B(acc_reg[6]), .Z(n168) );
  XOR2_X1 U293 ( .A(n167), .B(n168), .Z(n242) );
  MUX2_X1 U294 ( .A(n242), .B(bias[6]), .S(n212), .Z(N131) );
  XOR2_X1 U295 ( .A(prod_ext[7]), .B(acc_reg[7]), .Z(n169) );
  XOR2_X1 U296 ( .A(n170), .B(n169), .Z(n243) );
  MUX2_X1 U297 ( .A(n243), .B(bias[7]), .S(n212), .Z(N132) );
  XOR2_X1 U298 ( .A(prod_ext[8]), .B(acc_reg[8]), .Z(n171) );
  XOR2_X1 U299 ( .A(n172), .B(n171), .Z(n244) );
  MUX2_X1 U300 ( .A(n244), .B(bias[8]), .S(n235), .Z(N133) );
  XOR2_X1 U301 ( .A(prod_ext[9]), .B(acc_reg[9]), .Z(n174) );
  XOR2_X1 U302 ( .A(n173), .B(n174), .Z(n245) );
  MUX2_X1 U303 ( .A(n245), .B(bias[9]), .S(n235), .Z(N134) );
  XOR2_X1 U304 ( .A(prod_ext[10]), .B(acc_reg[10]), .Z(n176) );
  XOR2_X1 U305 ( .A(n175), .B(n176), .Z(n246) );
  MUX2_X1 U306 ( .A(n246), .B(bias[10]), .S(n235), .Z(N135) );
  XOR2_X1 U307 ( .A(prod_ext[11]), .B(acc_reg[11]), .Z(n178) );
  XOR2_X1 U308 ( .A(n177), .B(n178), .Z(n247) );
  MUX2_X1 U309 ( .A(n247), .B(bias[11]), .S(n235), .Z(N136) );
  XOR2_X1 U310 ( .A(prod_ext[12]), .B(acc_reg[12]), .Z(n180) );
  XOR2_X1 U311 ( .A(n179), .B(n180), .Z(n248) );
  MUX2_X1 U312 ( .A(n248), .B(bias[12]), .S(n235), .Z(N137) );
  XOR2_X1 U313 ( .A(prod_ext[13]), .B(acc_reg[13]), .Z(n182) );
  XOR2_X1 U314 ( .A(n181), .B(n182), .Z(n249) );
  MUX2_X1 U315 ( .A(n249), .B(bias[13]), .S(n235), .Z(N138) );
  XOR2_X1 U316 ( .A(prod_ext[14]), .B(acc_reg[14]), .Z(n183) );
  XOR2_X1 U317 ( .A(n184), .B(n183), .Z(n250) );
  MUX2_X1 U318 ( .A(n250), .B(bias[14]), .S(n235), .Z(N139) );
  XOR2_X1 U319 ( .A(n444), .B(acc_reg[15]), .Z(n186) );
  XOR2_X1 U320 ( .A(n185), .B(n186), .Z(n251) );
  MUX2_X1 U321 ( .A(n251), .B(bias[15]), .S(n235), .Z(N140) );
  XOR2_X1 U322 ( .A(n444), .B(acc_reg[16]), .Z(n188) );
  MUX2_X1 U323 ( .A(n252), .B(bias[16]), .S(n235), .Z(N141) );
  XOR2_X1 U324 ( .A(n444), .B(acc_reg[17]), .Z(n189) );
  XOR2_X1 U325 ( .A(n190), .B(n189), .Z(n253) );
  MUX2_X1 U326 ( .A(n253), .B(bias[17]), .S(n235), .Z(N142) );
  NAND3_X1 U327 ( .A1(n193), .A2(n192), .A3(n191), .ZN(n195) );
  XOR2_X1 U328 ( .A(n444), .B(acc_reg[18]), .Z(n194) );
  XOR2_X1 U329 ( .A(n195), .B(n194), .Z(n254) );
  MUX2_X1 U330 ( .A(n254), .B(bias[18]), .S(n235), .Z(N143) );
  NAND3_X1 U331 ( .A1(n61), .A2(n196), .A3(n197), .ZN(n199) );
  XOR2_X1 U332 ( .A(n444), .B(acc_reg[19]), .Z(n198) );
  XOR2_X1 U333 ( .A(n199), .B(n198), .Z(n256) );
  MUX2_X1 U334 ( .A(n256), .B(bias[19]), .S(n235), .Z(N144) );
  NAND3_X1 U335 ( .A1(n64), .A2(n200), .A3(n201), .ZN(n203) );
  XOR2_X1 U336 ( .A(n444), .B(acc_reg[20]), .Z(n202) );
  XOR2_X1 U337 ( .A(n203), .B(n202), .Z(n257) );
  MUX2_X1 U338 ( .A(n257), .B(bias[20]), .S(n212), .Z(N145) );
  NAND3_X1 U339 ( .A1(n65), .A2(n205), .A3(n204), .ZN(n207) );
  XOR2_X1 U340 ( .A(n444), .B(acc_reg[21]), .Z(n206) );
  XOR2_X1 U341 ( .A(n207), .B(n206), .Z(n258) );
  MUX2_X1 U342 ( .A(n258), .B(bias[21]), .S(n212), .Z(N146) );
  NAND3_X1 U343 ( .A1(n59), .A2(n208), .A3(n209), .ZN(n211) );
  XOR2_X1 U344 ( .A(n444), .B(acc_reg[22]), .Z(n210) );
  XOR2_X1 U345 ( .A(n211), .B(n210), .Z(n259) );
  MUX2_X1 U346 ( .A(n259), .B(bias[22]), .S(n212), .Z(N147) );
  XOR2_X1 U347 ( .A(n444), .B(acc_reg[23]), .Z(n214) );
  XOR2_X1 U348 ( .A(n213), .B(n214), .Z(n260) );
  MUX2_X1 U349 ( .A(n260), .B(bias[23]), .S(n235), .Z(N148) );
  XOR2_X1 U350 ( .A(n444), .B(acc_reg[24]), .Z(n216) );
  XOR2_X1 U351 ( .A(n216), .B(n215), .Z(n261) );
  MUX2_X1 U352 ( .A(n261), .B(bias[24]), .S(n235), .Z(N149) );
  XOR2_X1 U353 ( .A(n444), .B(acc_reg[25]), .Z(n218) );
  XOR2_X1 U354 ( .A(n218), .B(n217), .Z(n262) );
  MUX2_X1 U355 ( .A(n262), .B(bias[25]), .S(n235), .Z(N150) );
  XOR2_X1 U356 ( .A(n444), .B(acc_reg[26]), .Z(n220) );
  XOR2_X1 U357 ( .A(n219), .B(n220), .Z(n263) );
  MUX2_X1 U358 ( .A(n263), .B(bias[26]), .S(n235), .Z(N151) );
  XOR2_X1 U359 ( .A(n444), .B(acc_reg[27]), .Z(n225) );
  NAND3_X1 U360 ( .A1(n221), .A2(n222), .A3(n223), .ZN(n224) );
  XOR2_X1 U361 ( .A(n225), .B(n224), .Z(n264) );
  MUX2_X1 U362 ( .A(n264), .B(bias[27]), .S(n235), .Z(N152) );
  XOR2_X1 U363 ( .A(n444), .B(acc_reg[28]), .Z(n229) );
  NAND3_X1 U364 ( .A1(n58), .A2(n227), .A3(n226), .ZN(n228) );
  XOR2_X1 U365 ( .A(n229), .B(n228), .Z(n265) );
  MUX2_X1 U366 ( .A(n265), .B(bias[28]), .S(n235), .Z(N153) );
  XOR2_X1 U367 ( .A(n444), .B(acc_reg[29]), .Z(n231) );
  XNOR2_X1 U368 ( .A(n230), .B(n231), .ZN(n266) );
  MUX2_X1 U369 ( .A(n266), .B(bias[29]), .S(n235), .Z(N154) );
  AOI21_X1 U370 ( .B1(acc_reg[30]), .B2(n444), .A(n232), .ZN(n234) );
  XOR2_X1 U371 ( .A(n234), .B(n233), .Z(n268) );
  MUX2_X1 U372 ( .A(n268), .B(bias[30]), .S(n235), .Z(N155) );
  INV_X1 U373 ( .A(n443), .ZN(n255) );
  AOI22_X1 U374 ( .A1(n443), .A2(n236), .B1(n458), .B2(n255), .ZN(n55) );
  INV_X1 U375 ( .A(n443), .ZN(n267) );
  MUX2_X1 U376 ( .A(n237), .B(acc_out[1]), .S(n267), .Z(n54) );
  MUX2_X1 U377 ( .A(n238), .B(acc_out[2]), .S(n255), .Z(n53) );
  MUX2_X1 U378 ( .A(n239), .B(acc_out[3]), .S(n267), .Z(n52) );
  MUX2_X1 U379 ( .A(n240), .B(acc_out[4]), .S(n255), .Z(n51) );
  MUX2_X1 U380 ( .A(n241), .B(acc_out[5]), .S(n267), .Z(n50) );
  MUX2_X1 U381 ( .A(n242), .B(acc_out[6]), .S(n255), .Z(n49) );
  MUX2_X1 U382 ( .A(n243), .B(acc_out[7]), .S(n267), .Z(n48) );
  MUX2_X1 U383 ( .A(n244), .B(acc_out[8]), .S(n255), .Z(n47) );
  MUX2_X1 U384 ( .A(n245), .B(acc_out[9]), .S(n255), .Z(n46) );
  MUX2_X1 U385 ( .A(n246), .B(acc_out[10]), .S(n255), .Z(n45) );
  MUX2_X1 U386 ( .A(n247), .B(acc_out[11]), .S(n255), .Z(n44) );
  MUX2_X1 U387 ( .A(n248), .B(acc_out[12]), .S(n255), .Z(n43) );
  MUX2_X1 U388 ( .A(n249), .B(acc_out[13]), .S(n255), .Z(n42) );
  MUX2_X1 U389 ( .A(n250), .B(acc_out[14]), .S(n255), .Z(n41) );
  MUX2_X1 U390 ( .A(n251), .B(acc_out[15]), .S(n255), .Z(n40) );
  MUX2_X1 U391 ( .A(n252), .B(acc_out[16]), .S(n255), .Z(n39) );
  MUX2_X1 U392 ( .A(n253), .B(acc_out[17]), .S(n255), .Z(n38) );
  MUX2_X1 U393 ( .A(n254), .B(acc_out[18]), .S(n255), .Z(n37) );
  MUX2_X1 U394 ( .A(n256), .B(acc_out[19]), .S(n255), .Z(n36) );
  MUX2_X1 U395 ( .A(n257), .B(acc_out[20]), .S(n267), .Z(n35) );
  MUX2_X1 U396 ( .A(n258), .B(acc_out[21]), .S(n267), .Z(n34) );
  MUX2_X1 U397 ( .A(n259), .B(acc_out[22]), .S(n267), .Z(n33) );
  MUX2_X1 U398 ( .A(n260), .B(acc_out[23]), .S(n267), .Z(n32) );
  MUX2_X1 U399 ( .A(n261), .B(acc_out[24]), .S(n267), .Z(n31) );
  MUX2_X1 U400 ( .A(n262), .B(acc_out[25]), .S(n267), .Z(n30) );
  MUX2_X1 U401 ( .A(n263), .B(acc_out[26]), .S(n267), .Z(n29) );
  MUX2_X1 U402 ( .A(n264), .B(acc_out[27]), .S(n267), .Z(n28) );
  MUX2_X1 U403 ( .A(n265), .B(acc_out[28]), .S(n267), .Z(n27) );
  MUX2_X1 U404 ( .A(n266), .B(acc_out[29]), .S(n267), .Z(n26) );
  MUX2_X1 U405 ( .A(n268), .B(acc_out[30]), .S(n267), .Z(n25) );
  NAND2_X1 U406 ( .A1(do_acc), .A2(n446), .ZN(n269) );
  OAI22_X1 U407 ( .A1(tap_idx[1]), .A2(n269), .B1(do_acc), .B2(n446), .ZN(n22)
         );
  INV_X1 U408 ( .A(w[1]), .ZN(n438) );
  AOI22_X1 U409 ( .A1(w[1]), .A2(x[7]), .B1(n344), .B2(n438), .ZN(n273) );
  AOI22_X1 U410 ( .A1(n373), .A2(n270), .B1(n372), .B2(n273), .ZN(n280) );
  AOI22_X1 U411 ( .A1(x[3]), .A2(w[5]), .B1(n345), .B2(n300), .ZN(n275) );
  AOI22_X1 U412 ( .A1(n308), .A2(n271), .B1(n306), .B2(n275), .ZN(n279) );
  NAND2_X1 U413 ( .A1(n280), .A2(n279), .ZN(\intadd_2/A[1] ) );
  INV_X1 U414 ( .A(n372), .ZN(n361) );
  AOI221_X1 U415 ( .B1(w[0]), .B2(x[7]), .C1(n299), .C2(n344), .A(n361), .ZN(
        n272) );
  AOI21_X1 U416 ( .B1(n273), .B2(n373), .A(n272), .ZN(n328) );
  OAI221_X1 U417 ( .B1(n372), .B2(n373), .C1(n372), .C2(n299), .A(x[7]), .ZN(
        n327) );
  NOR2_X1 U418 ( .A1(n328), .A2(n327), .ZN(\intadd_2/A[0] ) );
  NAND2_X1 U419 ( .A1(n435), .A2(x[0]), .ZN(n436) );
  INV_X1 U420 ( .A(n436), .ZN(n311) );
  AOI22_X1 U421 ( .A1(w[7]), .A2(n311), .B1(x[1]), .B2(n342), .ZN(
        \intadd_2/B[0] ) );
  AOI22_X1 U422 ( .A1(x[3]), .A2(w[4]), .B1(n274), .B2(n300), .ZN(n284) );
  AOI22_X1 U423 ( .A1(n308), .A2(n275), .B1(n306), .B2(n284), .ZN(
        \intadd_1/A[0] ) );
  AOI22_X1 U424 ( .A1(w[2]), .A2(x[5]), .B1(n341), .B2(n292), .ZN(n288) );
  AOI22_X1 U425 ( .A1(n359), .A2(n276), .B1(n358), .B2(n288), .ZN(
        \intadd_1/B[0] ) );
  NAND2_X1 U426 ( .A1(x[1]), .A2(x[0]), .ZN(n437) );
  NAND2_X1 U427 ( .A1(x[1]), .A2(n277), .ZN(n309) );
  OAI22_X1 U428 ( .A1(w[7]), .A2(n437), .B1(w[6]), .B2(n309), .ZN(n278) );
  AOI21_X1 U429 ( .B1(n311), .B2(w[7]), .A(n278), .ZN(\intadd_1/CI ) );
  XOR2_X1 U430 ( .A(n280), .B(n279), .Z(\intadd_1/B[1] ) );
  AOI22_X1 U431 ( .A1(w[1]), .A2(x[5]), .B1(n341), .B2(n438), .ZN(n287) );
  OAI221_X1 U432 ( .B1(n299), .B2(n341), .C1(w[0]), .C2(x[5]), .A(n358), .ZN(
        n281) );
  INV_X1 U433 ( .A(n281), .ZN(n282) );
  AOI21_X1 U434 ( .B1(n359), .B2(n287), .A(n282), .ZN(n291) );
  OAI221_X1 U435 ( .B1(n358), .B2(n359), .C1(n358), .C2(n299), .A(x[5]), .ZN(
        n290) );
  NOR2_X1 U436 ( .A1(n291), .A2(n290), .ZN(n323) );
  AOI22_X1 U437 ( .A1(x[3]), .A2(w[3]), .B1(n283), .B2(n300), .ZN(n293) );
  AOI22_X1 U438 ( .A1(n308), .A2(n284), .B1(n306), .B2(n293), .ZN(n331) );
  NAND2_X1 U439 ( .A1(w[0]), .A2(n373), .ZN(n330) );
  OAI22_X1 U440 ( .A1(w[6]), .A2(n437), .B1(w[5]), .B2(n309), .ZN(n285) );
  AOI21_X1 U441 ( .B1(n311), .B2(w[6]), .A(n285), .ZN(n329) );
  INV_X1 U442 ( .A(n286), .ZN(n322) );
  AOI22_X1 U443 ( .A1(n359), .A2(n288), .B1(n358), .B2(n287), .ZN(n289) );
  INV_X1 U444 ( .A(n289), .ZN(n321) );
  AOI21_X1 U445 ( .B1(n291), .B2(n290), .A(n323), .ZN(n320) );
  OAI22_X1 U446 ( .A1(n300), .A2(w[2]), .B1(n292), .B2(x[3]), .ZN(n307) );
  AOI22_X1 U447 ( .A1(n308), .A2(n293), .B1(n306), .B2(n307), .ZN(n294) );
  INV_X1 U448 ( .A(n294), .ZN(n319) );
  OAI221_X1 U449 ( .B1(x[1]), .B2(w[5]), .C1(n435), .C2(n345), .A(x[0]), .ZN(
        n295) );
  OAI21_X1 U450 ( .B1(w[4]), .B2(n309), .A(n295), .ZN(n318) );
  OAI22_X1 U451 ( .A1(w[1]), .A2(n309), .B1(w[2]), .B2(n437), .ZN(n296) );
  AOI21_X1 U452 ( .B1(n311), .B2(w[2]), .A(n296), .ZN(n431) );
  AOI211_X1 U453 ( .C1(w[1]), .C2(x[0]), .A(w[0]), .B(n435), .ZN(n442) );
  AOI21_X1 U454 ( .B1(w[0]), .B2(n308), .A(n442), .ZN(n432) );
  NOR2_X1 U455 ( .A1(n431), .A2(n432), .ZN(n430) );
  AOI221_X1 U456 ( .B1(x[2]), .B2(n297), .C1(w[0]), .C2(n308), .A(n300), .ZN(
        n313) );
  NOR2_X1 U457 ( .A1(n430), .A2(n313), .ZN(n424) );
  AOI22_X1 U458 ( .A1(x[3]), .A2(w[1]), .B1(n438), .B2(n300), .ZN(n305) );
  INV_X1 U459 ( .A(n306), .ZN(n298) );
  AOI221_X1 U460 ( .B1(x[3]), .B2(w[0]), .C1(n300), .C2(n299), .A(n298), .ZN(
        n301) );
  AOI21_X1 U461 ( .B1(n308), .B2(n305), .A(n301), .ZN(n304) );
  OAI22_X1 U462 ( .A1(w[3]), .A2(n437), .B1(w[2]), .B2(n309), .ZN(n302) );
  AOI21_X1 U463 ( .B1(n311), .B2(w[3]), .A(n302), .ZN(n303) );
  XNOR2_X1 U464 ( .A(n304), .B(n303), .ZN(n428) );
  OR2_X1 U465 ( .A1(n304), .A2(n303), .ZN(n314) );
  OAI21_X1 U466 ( .B1(n424), .B2(n428), .A(n314), .ZN(n419) );
  AOI22_X1 U467 ( .A1(n308), .A2(n307), .B1(n306), .B2(n305), .ZN(n317) );
  NAND2_X1 U468 ( .A1(w[0]), .A2(n359), .ZN(n316) );
  OAI22_X1 U469 ( .A1(w[3]), .A2(n309), .B1(w[4]), .B2(n437), .ZN(n310) );
  AOI21_X1 U470 ( .B1(n311), .B2(w[4]), .A(n310), .ZN(n315) );
  INV_X1 U471 ( .A(n312), .ZN(n422) );
  NAND2_X1 U472 ( .A1(n430), .A2(n313), .ZN(n426) );
  NOR2_X1 U473 ( .A1(n314), .A2(n426), .ZN(n418) );
  AOI21_X1 U474 ( .B1(n419), .B2(n422), .A(n418), .ZN(n416) );
  FA_X1 U475 ( .A(n317), .B(n316), .CI(n315), .CO(n413), .S(n312) );
  FA_X1 U476 ( .A(n320), .B(n319), .CI(n318), .CO(n325), .S(n414) );
  INV_X1 U477 ( .A(n414), .ZN(n411) );
  AOI222_X1 U478 ( .A1(n416), .A2(n413), .B1(n416), .B2(n411), .C1(n413), .C2(
        n411), .ZN(n324) );
  NOR2_X1 U479 ( .A1(n325), .A2(n324), .ZN(n406) );
  INV_X1 U480 ( .A(n406), .ZN(n326) );
  FA_X1 U481 ( .A(n323), .B(n322), .CI(n321), .CO(n404), .S(n409) );
  AND2_X1 U482 ( .A1(n325), .A2(n324), .ZN(n407) );
  AOI21_X1 U483 ( .B1(n326), .B2(n409), .A(n407), .ZN(n402) );
  INV_X1 U484 ( .A(n402), .ZN(n399) );
  XNOR2_X1 U485 ( .A(n328), .B(n327), .ZN(n333) );
  FA_X1 U486 ( .A(n331), .B(n330), .CI(n329), .CO(n332), .S(n286) );
  INV_X1 U487 ( .A(n401), .ZN(n400) );
  AOI222_X1 U488 ( .A1(n404), .A2(n399), .B1(n404), .B2(n400), .C1(n399), .C2(
        n400), .ZN(n334) );
  INV_X1 U489 ( .A(n334), .ZN(n397) );
  INV_X1 U490 ( .A(\intadd_1/SUM[1] ), .ZN(n395) );
  FA_X1 U491 ( .A(\intadd_1/SUM[0] ), .B(n333), .CI(n332), .CO(n394), .S(n401)
         );
  AOI21_X1 U492 ( .B1(\intadd_1/SUM[1] ), .B2(n334), .A(n394), .ZN(n335) );
  AOI21_X1 U493 ( .B1(n397), .B2(n395), .A(n335), .ZN(\intadd_1/B[2] ) );
  AOI22_X1 U494 ( .A1(x[7]), .A2(n336), .B1(w[6]), .B2(n344), .ZN(n362) );
  AOI22_X1 U495 ( .A1(w[7]), .A2(x[7]), .B1(n344), .B2(n342), .ZN(n371) );
  NAND2_X1 U496 ( .A1(n373), .A2(n371), .ZN(n337) );
  OAI21_X1 U497 ( .B1(n362), .B2(n361), .A(n337), .ZN(n378) );
  FA_X1 U498 ( .A(n340), .B(n339), .CI(n338), .CO(n353), .S(n76) );
  AOI22_X1 U499 ( .A1(x[5]), .A2(w[7]), .B1(n342), .B2(n341), .ZN(n357) );
  AOI22_X1 U500 ( .A1(n359), .A2(n357), .B1(n358), .B2(n343), .ZN(n356) );
  INV_X1 U501 ( .A(n373), .ZN(n363) );
  AOI22_X1 U502 ( .A1(x[7]), .A2(n345), .B1(w[5]), .B2(n344), .ZN(n360) );
  INV_X1 U503 ( .A(n346), .ZN(n347) );
  OAI22_X1 U504 ( .A1(n363), .A2(n360), .B1(n361), .B2(n347), .ZN(n352) );
  FA_X1 U505 ( .A(n350), .B(n349), .CI(n348), .CO(n351), .S(n80) );
  INV_X1 U506 ( .A(n351), .ZN(n355) );
  NAND2_X1 U507 ( .A1(\intadd_2/n1 ), .A2(n355), .ZN(n388) );
  FA_X1 U508 ( .A(n353), .B(n356), .CI(n352), .CO(n365), .S(n354) );
  INV_X1 U509 ( .A(n354), .ZN(n392) );
  NOR2_X1 U510 ( .A1(\intadd_2/n1 ), .A2(n355), .ZN(n390) );
  AOI21_X1 U511 ( .B1(n388), .B2(n392), .A(n390), .ZN(n364) );
  AND2_X1 U512 ( .A1(n365), .A2(n364), .ZN(n383) );
  INV_X1 U513 ( .A(n356), .ZN(n369) );
  OAI21_X1 U514 ( .B1(n359), .B2(n358), .A(n357), .ZN(n368) );
  OAI22_X1 U515 ( .A1(n363), .A2(n362), .B1(n361), .B2(n360), .ZN(n367) );
  NOR2_X1 U516 ( .A1(n365), .A2(n364), .ZN(n384) );
  INV_X1 U517 ( .A(n384), .ZN(n366) );
  OAI21_X1 U518 ( .B1(n383), .B2(n386), .A(n366), .ZN(n381) );
  FA_X1 U519 ( .A(n369), .B(n368), .CI(n367), .CO(n370), .S(n386) );
  INV_X1 U520 ( .A(n370), .ZN(n379) );
  AOI222_X1 U521 ( .A1(n381), .A2(n379), .B1(n381), .B2(n378), .C1(n379), .C2(
        n378), .ZN(n376) );
  OAI21_X1 U522 ( .B1(n373), .B2(n372), .A(n371), .ZN(n375) );
  AOI22_X1 U523 ( .A1(valid_in), .A2(n374), .B1(n447), .B2(n433), .ZN(n21) );
  FA_X1 U524 ( .A(n378), .B(n376), .CI(n375), .CO(n374), .S(n377) );
  MUX2_X1 U525 ( .A(n377), .B(prod_ext[14]), .S(n433), .Z(n20) );
  XOR2_X1 U526 ( .A(n379), .B(n378), .Z(n380) );
  XOR2_X1 U527 ( .A(n381), .B(n380), .Z(n382) );
  AOI22_X1 U528 ( .A1(valid_in), .A2(n382), .B1(n448), .B2(n433), .ZN(n19) );
  NOR2_X1 U529 ( .A1(n384), .A2(n383), .ZN(n385) );
  XNOR2_X1 U530 ( .A(n386), .B(n385), .ZN(n387) );
  AOI22_X1 U531 ( .A1(valid_in), .A2(n387), .B1(n449), .B2(n433), .ZN(n18) );
  INV_X1 U532 ( .A(n388), .ZN(n389) );
  NOR2_X1 U533 ( .A1(n390), .A2(n389), .ZN(n391) );
  XOR2_X1 U534 ( .A(n392), .B(n391), .Z(n393) );
  AOI22_X1 U535 ( .A1(valid_in), .A2(n393), .B1(n450), .B2(n433), .ZN(n17) );
  MUX2_X1 U536 ( .A(\intadd_2/SUM[2] ), .B(prod_ext[10]), .S(n433), .Z(n16) );
  AOI22_X1 U537 ( .A1(valid_in), .A2(\intadd_1/SUM[2] ), .B1(n451), .B2(n433), 
        .ZN(n15) );
  XOR2_X1 U538 ( .A(n395), .B(n394), .Z(n396) );
  XOR2_X1 U539 ( .A(n397), .B(n396), .Z(n398) );
  AOI22_X1 U540 ( .A1(valid_in), .A2(n398), .B1(n452), .B2(n433), .ZN(n14) );
  AOI22_X1 U541 ( .A1(n402), .A2(n401), .B1(n400), .B2(n399), .ZN(n403) );
  XNOR2_X1 U542 ( .A(n404), .B(n403), .ZN(n405) );
  AOI22_X1 U543 ( .A1(valid_in), .A2(n405), .B1(n453), .B2(n433), .ZN(n13) );
  NOR2_X1 U544 ( .A1(n407), .A2(n406), .ZN(n408) );
  XNOR2_X1 U545 ( .A(n409), .B(n408), .ZN(n410) );
  AOI22_X1 U546 ( .A1(valid_in), .A2(n410), .B1(n454), .B2(n433), .ZN(n12) );
  INV_X1 U547 ( .A(n413), .ZN(n412) );
  AOI22_X1 U548 ( .A1(n414), .A2(n413), .B1(n412), .B2(n411), .ZN(n415) );
  XNOR2_X1 U549 ( .A(n416), .B(n415), .ZN(n417) );
  AOI22_X1 U550 ( .A1(valid_in), .A2(n417), .B1(n457), .B2(n433), .ZN(n11) );
  INV_X1 U551 ( .A(n418), .ZN(n420) );
  NAND2_X1 U552 ( .A1(n420), .A2(n419), .ZN(n421) );
  XOR2_X1 U553 ( .A(n422), .B(n421), .Z(n423) );
  AOI22_X1 U554 ( .A1(valid_in), .A2(n423), .B1(n455), .B2(n433), .ZN(n10) );
  INV_X1 U555 ( .A(n424), .ZN(n425) );
  NAND2_X1 U556 ( .A1(n426), .A2(n425), .ZN(n427) );
  XNOR2_X1 U557 ( .A(n428), .B(n427), .ZN(n429) );
  AOI22_X1 U558 ( .A1(valid_in), .A2(n429), .B1(n456), .B2(n433), .ZN(n9) );
  AOI21_X1 U559 ( .B1(n432), .B2(n431), .A(n430), .ZN(n434) );
  MUX2_X1 U560 ( .A(n434), .B(prod_ext[2]), .S(n433), .Z(n8) );
  AOI21_X1 U561 ( .B1(w[0]), .B2(x[0]), .A(n435), .ZN(n440) );
  AOI22_X1 U562 ( .A1(n438), .A2(n437), .B1(n436), .B2(w[1]), .ZN(n439) );
  OAI21_X1 U563 ( .B1(n440), .B2(n439), .A(valid_in), .ZN(n441) );
  OAI22_X1 U564 ( .A1(n442), .A2(n441), .B1(valid_in), .B2(n459), .ZN(n7) );
endmodule


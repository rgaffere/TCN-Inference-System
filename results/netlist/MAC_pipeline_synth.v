/////////////////////////////////////////////////////////////
// Created by: Synopsys DC Ultra(TM) in wire load mode
// Version   : X-2025.06-SP4
// Date      : Wed May 13 00:38:20 2026
/////////////////////////////////////////////////////////////


module MAC_pipeline ( clk, rst_n, valid_in, x, w, acc_in, valid_out, acc_out
 );
  input [7:0] x;
  input [7:0] w;
  input [31:0] acc_in;
  output [31:0] acc_out;
  input clk, rst_n, valid_in;
  output valid_out;
  wire   valid_s1, N1, N2, N3, N4, N5, N6, N7, N8, N9, N10, N11, N12, N13, N14,
         N15, N16, N17, N18, N19, N20, N21, N22, N23, N24, N25, N26, N27, N28,
         N29, N30, N31, N32, N33, N34, N35, N36, N37, N38, N39, N40, N41, N42,
         N43, N44, N45, N46, N47, N48, \intadd_0/A[11] , \intadd_0/A[10] ,
         \intadd_0/A[9] , \intadd_0/A[8] , \intadd_0/A[7] , \intadd_0/A[6] ,
         \intadd_0/A[5] , \intadd_0/A[4] , \intadd_0/A[3] , \intadd_0/A[2] ,
         \intadd_0/A[1] , \intadd_0/A[0] , \intadd_0/B[11] , \intadd_0/B[10] ,
         \intadd_0/B[9] , \intadd_0/B[8] , \intadd_0/B[7] , \intadd_0/B[6] ,
         \intadd_0/B[5] , \intadd_0/B[4] , \intadd_0/B[3] , \intadd_0/B[2] ,
         \intadd_0/B[1] , \intadd_0/B[0] , \intadd_0/CI , \intadd_0/SUM[11] ,
         \intadd_0/SUM[10] , \intadd_0/SUM[9] , \intadd_0/SUM[8] ,
         \intadd_0/SUM[7] , \intadd_0/SUM[6] , \intadd_0/SUM[5] ,
         \intadd_0/SUM[4] , \intadd_0/SUM[3] , \intadd_0/SUM[2] ,
         \intadd_0/SUM[1] , \intadd_0/SUM[0] , \intadd_0/n12 , \intadd_0/n11 ,
         \intadd_0/n10 , \intadd_0/n9 , \intadd_0/n8 , \intadd_0/n7 ,
         \intadd_0/n6 , \intadd_0/n5 , \intadd_0/n4 , \intadd_0/n3 ,
         \intadd_0/n2 , n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13,
         n14, n15, n16, n17, n18, n19, n20, n21, n22, n23, n24, n25, n26, n27,
         n28, n29, n30, n31, n32, n33, n34, n35, n36, n37, n38, n39, n40, n41,
         n42, n43, n44, n45, n46, n47, n48, n49, n50, n51, n52, n53, n54, n55,
         n56, n57, n58, n59, n60, n61, n62, n63, n64, n65, n66, n67, n68, n69,
         n70, n71, n72, n73, n74, n75, n76, n77, n78, n79, n80, n81, n82, n83,
         n84, n85, n86, n87, n88, n89, n90, n91, n92, n93, n94, n95, n96, n97,
         n98, n99, n100, n101, n102, n103, n104, n105, n106, n107, n108, n109,
         n110, n111, n112, n113, n114, n115, n116, n117, n118, n119, n120,
         n121, n122, n123, n124, n125, n126, n127, n128, n129, n130, n131,
         n132, n133, n134, n135, n136, n137, n138, n139, n140, n141, n142,
         n143, n144, n145, n146, n147, n148, n149, n150, n151, n152, n153,
         n154, n155, n156, n157, n158, n159, n160, n161, n162, n163, n164,
         n165, n166, n167, n168, n169, n170, n171, n172, n173, n174, n175,
         n176, n177, n178, n179, n180, n181, n182, n183, n184, n185, n186,
         n187, n188, n189, n190, n191, n192, n193, n194, n195, n196, n197,
         n198, n199, n200, n201, n202, n203, n204, n205, n206, n207, n208,
         n209, n210, n211, n212, n213, n214, n215, n216, n217, n218, n219,
         n220, n221, n222, n223, n224, n225, n226, n227, n228, n229, n230,
         n231, n232, n233, n234, n235, n236, n237, n238, n239, n240, n241,
         n242, n243, n244, n245, n246, n247, n248, n249, n250, n251, n252,
         n253, n254, n255, n256, n257, n258, n259, n260, n261, n262, n263,
         n264, n265, n266, n267, n268, n269, n270, n271, n272, n273, n274,
         n275, n276, n277, n278, n279, n280, n281, n282, n283, n284, n285,
         n286, n287, n288, n289, n290, n291, n292, n293, n294, n295, n296,
         n297, n298, n299, n300, n301, n302, n303, n304, n305, n306, n307,
         n308, n309, n310, n311, n312, n313, n314, n315, n316, n317, n318,
         n319, n320, n321, n322, n323, n324, n325, n326, n327, n328, n329,
         n330, n331, n332, n333, n334, n335, n336, n337, n338, n339, n340,
         n341, n342, n343, n344, n345, n346, n347, n348, n349, n350, n351,
         n352, n353, n354, n355, n356, n357, n358, n359, n360, n361, n362,
         n363, n364, n365, n366, n367, n368, n369, n370, n371, n372, n373,
         n374, n375, n376, n377, n378, n379, n380, n381, n382, n383, n384,
         n385, n386, n387, n388, n389, n390, n391, n392, n393, n394, n395,
         n396, n397, n398, n399, n400, n401;
  wire   [15:0] mul_reg;
  wire   [31:0] acc_in_reg;

  FD2 valid_s1_reg ( .D(valid_in), .CP(clk), .CD(rst_n), .Q(valid_s1) );
  FD2 \mul_reg_reg[14]  ( .D(N15), .CP(clk), .CD(rst_n), .Q(mul_reg[14]) );
  FD2 \mul_reg_reg[13]  ( .D(N14), .CP(clk), .CD(rst_n), .Q(mul_reg[13]) );
  FD2 \mul_reg_reg[12]  ( .D(N13), .CP(clk), .CD(rst_n), .Q(mul_reg[12]) );
  FD2 \mul_reg_reg[11]  ( .D(N12), .CP(clk), .CD(rst_n), .Q(mul_reg[11]) );
  FD2 \mul_reg_reg[10]  ( .D(N11), .CP(clk), .CD(rst_n), .Q(mul_reg[10]) );
  FD2 \mul_reg_reg[9]  ( .D(N10), .CP(clk), .CD(rst_n), .Q(mul_reg[9]) );
  FD2 \mul_reg_reg[8]  ( .D(N9), .CP(clk), .CD(rst_n), .Q(mul_reg[8]) );
  FD2 \mul_reg_reg[7]  ( .D(N8), .CP(clk), .CD(rst_n), .Q(mul_reg[7]) );
  FD2 \mul_reg_reg[6]  ( .D(N7), .CP(clk), .CD(rst_n), .Q(mul_reg[6]) );
  FD2 \mul_reg_reg[5]  ( .D(N6), .CP(clk), .CD(rst_n), .Q(mul_reg[5]) );
  FD2 \mul_reg_reg[4]  ( .D(N5), .CP(clk), .CD(rst_n), .Q(mul_reg[4]) );
  FD2 \mul_reg_reg[3]  ( .D(N4), .CP(clk), .CD(rst_n), .Q(mul_reg[3]) );
  FD2 \mul_reg_reg[2]  ( .D(N3), .CP(clk), .CD(rst_n), .Q(mul_reg[2]) );
  FD2 \mul_reg_reg[1]  ( .D(N2), .CP(clk), .CD(rst_n), .Q(mul_reg[1]) );
  FD2 \acc_in_reg_reg[31]  ( .D(acc_in[31]), .CP(clk), .CD(rst_n), .Q(
        acc_in_reg[31]) );
  FD2 \acc_in_reg_reg[30]  ( .D(acc_in[30]), .CP(clk), .CD(rst_n), .Q(
        acc_in_reg[30]) );
  FD2 \acc_in_reg_reg[29]  ( .D(acc_in[29]), .CP(clk), .CD(rst_n), .Q(
        acc_in_reg[29]) );
  FD2 \acc_in_reg_reg[28]  ( .D(acc_in[28]), .CP(clk), .CD(rst_n), .Q(
        acc_in_reg[28]) );
  FD2 \acc_in_reg_reg[27]  ( .D(acc_in[27]), .CP(clk), .CD(rst_n), .Q(
        acc_in_reg[27]) );
  FD2 \acc_in_reg_reg[26]  ( .D(acc_in[26]), .CP(clk), .CD(rst_n), .Q(
        acc_in_reg[26]) );
  FD2 \acc_in_reg_reg[25]  ( .D(acc_in[25]), .CP(clk), .CD(rst_n), .Q(
        acc_in_reg[25]) );
  FD2 \acc_in_reg_reg[24]  ( .D(acc_in[24]), .CP(clk), .CD(rst_n), .Q(
        acc_in_reg[24]) );
  FD2 \acc_in_reg_reg[23]  ( .D(acc_in[23]), .CP(clk), .CD(rst_n), .Q(
        acc_in_reg[23]) );
  FD2 \acc_in_reg_reg[22]  ( .D(acc_in[22]), .CP(clk), .CD(rst_n), .Q(
        acc_in_reg[22]) );
  FD2 \acc_in_reg_reg[21]  ( .D(acc_in[21]), .CP(clk), .CD(rst_n), .Q(
        acc_in_reg[21]) );
  FD2 \acc_in_reg_reg[20]  ( .D(acc_in[20]), .CP(clk), .CD(rst_n), .Q(
        acc_in_reg[20]) );
  FD2 \acc_in_reg_reg[19]  ( .D(acc_in[19]), .CP(clk), .CD(rst_n), .Q(
        acc_in_reg[19]) );
  FD2 \acc_in_reg_reg[18]  ( .D(acc_in[18]), .CP(clk), .CD(rst_n), .Q(
        acc_in_reg[18]) );
  FD2 \acc_in_reg_reg[17]  ( .D(acc_in[17]), .CP(clk), .CD(rst_n), .Q(
        acc_in_reg[17]) );
  FD2 \acc_in_reg_reg[16]  ( .D(acc_in[16]), .CP(clk), .CD(rst_n), .Q(
        acc_in_reg[16]) );
  FD2 \acc_in_reg_reg[15]  ( .D(acc_in[15]), .CP(clk), .CD(rst_n), .Q(
        acc_in_reg[15]) );
  FD2 \acc_in_reg_reg[14]  ( .D(acc_in[14]), .CP(clk), .CD(rst_n), .Q(
        acc_in_reg[14]) );
  FD2 \acc_in_reg_reg[13]  ( .D(acc_in[13]), .CP(clk), .CD(rst_n), .Q(
        acc_in_reg[13]) );
  FD2 \acc_in_reg_reg[12]  ( .D(acc_in[12]), .CP(clk), .CD(rst_n), .Q(
        acc_in_reg[12]) );
  FD2 \acc_in_reg_reg[11]  ( .D(acc_in[11]), .CP(clk), .CD(rst_n), .Q(
        acc_in_reg[11]) );
  FD2 \acc_in_reg_reg[10]  ( .D(acc_in[10]), .CP(clk), .CD(rst_n), .Q(
        acc_in_reg[10]) );
  FD2 \acc_in_reg_reg[9]  ( .D(acc_in[9]), .CP(clk), .CD(rst_n), .Q(
        acc_in_reg[9]) );
  FD2 \acc_in_reg_reg[8]  ( .D(acc_in[8]), .CP(clk), .CD(rst_n), .Q(
        acc_in_reg[8]) );
  FD2 \acc_in_reg_reg[7]  ( .D(acc_in[7]), .CP(clk), .CD(rst_n), .Q(
        acc_in_reg[7]) );
  FD2 \acc_in_reg_reg[6]  ( .D(acc_in[6]), .CP(clk), .CD(rst_n), .Q(
        acc_in_reg[6]) );
  FD2 \acc_in_reg_reg[5]  ( .D(acc_in[5]), .CP(clk), .CD(rst_n), .Q(
        acc_in_reg[5]) );
  FD2 \acc_in_reg_reg[4]  ( .D(acc_in[4]), .CP(clk), .CD(rst_n), .Q(
        acc_in_reg[4]) );
  FD2 \acc_in_reg_reg[3]  ( .D(acc_in[3]), .CP(clk), .CD(rst_n), .Q(
        acc_in_reg[3]) );
  FD2 \acc_in_reg_reg[2]  ( .D(acc_in[2]), .CP(clk), .CD(rst_n), .Q(
        acc_in_reg[2]) );
  FD2 \acc_in_reg_reg[1]  ( .D(acc_in[1]), .CP(clk), .CD(rst_n), .Q(
        acc_in_reg[1]) );
  FD2 valid_out_reg ( .D(valid_s1), .CP(clk), .CD(rst_n), .Q(valid_out) );
  FD2 \acc_out_reg[31]  ( .D(N48), .CP(clk), .CD(rst_n), .Q(acc_out[31]) );
  FD2 \acc_out_reg[30]  ( .D(N47), .CP(clk), .CD(rst_n), .Q(acc_out[30]) );
  FD2 \acc_out_reg[29]  ( .D(N46), .CP(clk), .CD(rst_n), .Q(acc_out[29]) );
  FD2 \acc_out_reg[28]  ( .D(N45), .CP(clk), .CD(rst_n), .Q(acc_out[28]) );
  FD2 \acc_out_reg[27]  ( .D(N44), .CP(clk), .CD(rst_n), .Q(acc_out[27]) );
  FD2 \acc_out_reg[26]  ( .D(N43), .CP(clk), .CD(rst_n), .Q(acc_out[26]) );
  FD2 \acc_out_reg[25]  ( .D(N42), .CP(clk), .CD(rst_n), .Q(acc_out[25]) );
  FD2 \acc_out_reg[24]  ( .D(N41), .CP(clk), .CD(rst_n), .Q(acc_out[24]) );
  FD2 \acc_out_reg[23]  ( .D(N40), .CP(clk), .CD(rst_n), .Q(acc_out[23]) );
  FD2 \acc_out_reg[22]  ( .D(N39), .CP(clk), .CD(rst_n), .Q(acc_out[22]) );
  FD2 \acc_out_reg[21]  ( .D(N38), .CP(clk), .CD(rst_n), .Q(acc_out[21]) );
  FD2 \acc_out_reg[20]  ( .D(N37), .CP(clk), .CD(rst_n), .Q(acc_out[20]) );
  FD2 \acc_out_reg[19]  ( .D(N36), .CP(clk), .CD(rst_n), .Q(acc_out[19]) );
  FD2 \acc_out_reg[18]  ( .D(N35), .CP(clk), .CD(rst_n), .Q(acc_out[18]) );
  FD2 \acc_out_reg[17]  ( .D(N34), .CP(clk), .CD(rst_n), .Q(acc_out[17]) );
  FD2 \acc_out_reg[16]  ( .D(N33), .CP(clk), .CD(rst_n), .Q(acc_out[16]) );
  FD2 \acc_out_reg[15]  ( .D(N32), .CP(clk), .CD(rst_n), .Q(acc_out[15]) );
  FD2 \acc_out_reg[14]  ( .D(N31), .CP(clk), .CD(rst_n), .Q(acc_out[14]) );
  FD2 \acc_out_reg[13]  ( .D(N30), .CP(clk), .CD(rst_n), .Q(acc_out[13]) );
  FD2 \acc_out_reg[12]  ( .D(N29), .CP(clk), .CD(rst_n), .Q(acc_out[12]) );
  FD2 \acc_out_reg[11]  ( .D(N28), .CP(clk), .CD(rst_n), .Q(acc_out[11]) );
  FD2 \acc_out_reg[10]  ( .D(N27), .CP(clk), .CD(rst_n), .Q(acc_out[10]) );
  FD2 \acc_out_reg[9]  ( .D(N26), .CP(clk), .CD(rst_n), .Q(acc_out[9]) );
  FD2 \acc_out_reg[8]  ( .D(N25), .CP(clk), .CD(rst_n), .Q(acc_out[8]) );
  FD2 \acc_out_reg[7]  ( .D(N24), .CP(clk), .CD(rst_n), .Q(acc_out[7]) );
  FD2 \acc_out_reg[6]  ( .D(N23), .CP(clk), .CD(rst_n), .Q(acc_out[6]) );
  FD2 \acc_out_reg[5]  ( .D(N22), .CP(clk), .CD(rst_n), .Q(acc_out[5]) );
  FD2 \acc_out_reg[4]  ( .D(N21), .CP(clk), .CD(rst_n), .Q(acc_out[4]) );
  FD2 \acc_out_reg[3]  ( .D(N20), .CP(clk), .CD(rst_n), .Q(acc_out[3]) );
  FD2 \acc_out_reg[2]  ( .D(N19), .CP(clk), .CD(rst_n), .Q(acc_out[2]) );
  FD2 \acc_out_reg[1]  ( .D(N18), .CP(clk), .CD(rst_n), .Q(acc_out[1]) );
  FD2 \acc_out_reg[0]  ( .D(N17), .CP(clk), .CD(rst_n), .Q(acc_out[0]) );
  FA1A \intadd_0/U11  ( .A(\intadd_0/A[1] ), .B(\intadd_0/B[1] ), .CI(
        \intadd_0/n12 ), .CO(\intadd_0/n11 ), .S(\intadd_0/SUM[1] ) );
  FA1A \intadd_0/U10  ( .A(\intadd_0/A[2] ), .B(\intadd_0/B[2] ), .CI(
        \intadd_0/n11 ), .CO(\intadd_0/n10 ), .S(\intadd_0/SUM[2] ) );
  FA1A \intadd_0/U9  ( .A(\intadd_0/A[3] ), .B(\intadd_0/B[3] ), .CI(
        \intadd_0/n10 ), .CO(\intadd_0/n9 ), .S(\intadd_0/SUM[3] ) );
  FA1A \intadd_0/U8  ( .A(\intadd_0/A[4] ), .B(\intadd_0/B[4] ), .CI(
        \intadd_0/n9 ), .CO(\intadd_0/n8 ), .S(\intadd_0/SUM[4] ) );
  FA1A \intadd_0/U7  ( .A(\intadd_0/A[5] ), .B(\intadd_0/B[5] ), .CI(
        \intadd_0/n8 ), .CO(\intadd_0/n7 ), .S(\intadd_0/SUM[5] ) );
  FA1A \intadd_0/U6  ( .A(\intadd_0/A[6] ), .B(\intadd_0/B[6] ), .CI(
        \intadd_0/n7 ), .CO(\intadd_0/n6 ), .S(\intadd_0/SUM[6] ) );
  FA1A \intadd_0/U5  ( .A(\intadd_0/A[7] ), .B(\intadd_0/B[7] ), .CI(
        \intadd_0/n6 ), .CO(\intadd_0/n5 ), .S(\intadd_0/SUM[7] ) );
  FA1A \intadd_0/U4  ( .A(\intadd_0/A[8] ), .B(\intadd_0/B[8] ), .CI(
        \intadd_0/n5 ), .CO(\intadd_0/n4 ), .S(\intadd_0/SUM[8] ) );
  FA1A \intadd_0/U3  ( .A(\intadd_0/A[9] ), .B(\intadd_0/B[9] ), .CI(
        \intadd_0/n4 ), .CO(\intadd_0/n3 ), .S(\intadd_0/SUM[9] ) );
  FA1A \intadd_0/U2  ( .A(\intadd_0/A[10] ), .B(\intadd_0/B[10] ), .CI(
        \intadd_0/n3 ), .CO(\intadd_0/n2 ), .S(\intadd_0/SUM[10] ) );
  FA1A \intadd_0/U1  ( .A(\intadd_0/A[11] ), .B(\intadd_0/B[11] ), .CI(
        \intadd_0/n2 ), .CO(N16), .S(\intadd_0/SUM[11] ) );
  FD2 \mul_reg_reg[15]  ( .D(N16), .CP(clk), .CD(rst_n), .Q(mul_reg[15]), .QN(
        n2) );
  FD2 \mul_reg_reg[0]  ( .D(N1), .CP(clk), .CD(rst_n), .Q(mul_reg[0]) );
  FD2 \acc_in_reg_reg[0]  ( .D(acc_in[0]), .CP(clk), .CD(rst_n), .Q(
        acc_in_reg[0]) );
  FA1A \intadd_0/U12  ( .A(\intadd_0/A[0] ), .B(\intadd_0/B[0] ), .CI(
        \intadd_0/CI ), .CO(\intadd_0/n12 ), .S(\intadd_0/SUM[0] ) );
  B4I U4 ( .A(n2), .Z(n210) );
  IVP U5 ( .A(n222), .Z(n207) );
  NR2 U6 ( .A(n210), .B(acc_in_reg[23]), .Z(n50) );
  EN U7 ( .A(n210), .B(acc_in_reg[24]), .Z(n3) );
  EN U8 ( .A(n50), .B(n3), .Z(n49) );
  ND2 U9 ( .A(mul_reg[15]), .B(acc_in_reg[23]), .Z(n53) );
  EN U10 ( .A(n3), .B(n53), .Z(n48) );
  ND2 U11 ( .A(mul_reg[15]), .B(acc_in_reg[17]), .Z(n176) );
  ND2 U12 ( .A(mul_reg[15]), .B(acc_in_reg[18]), .Z(n4) );
  ND2 U13 ( .A(n176), .B(n4), .Z(n6) );
  ND2 U14 ( .A(mul_reg[15]), .B(acc_in_reg[15]), .Z(n184) );
  ND2 U15 ( .A(mul_reg[15]), .B(acc_in_reg[16]), .Z(n5) );
  ND2 U16 ( .A(n184), .B(n5), .Z(n190) );
  NR2 U17 ( .A(n6), .B(n190), .Z(n218) );
  ND2 U18 ( .A(n210), .B(acc_in_reg[21]), .Z(n211) );
  ND2 U19 ( .A(n210), .B(acc_in_reg[22]), .Z(n7) );
  ND2 U20 ( .A(n211), .B(n7), .Z(n9) );
  ND2 U21 ( .A(n210), .B(acc_in_reg[19]), .Z(n200) );
  ND2 U22 ( .A(n210), .B(acc_in_reg[20]), .Z(n8) );
  ND2 U23 ( .A(n200), .B(n8), .Z(n215) );
  NR2 U24 ( .A(n9), .B(n215), .Z(n10) );
  ND2 U25 ( .A(n218), .B(n10), .Z(n47) );
  NR2 U26 ( .A(n210), .B(acc_in_reg[20]), .Z(n11) );
  NR2 U27 ( .A(n210), .B(acc_in_reg[19]), .Z(n202) );
  NR2 U28 ( .A(n11), .B(n202), .Z(n214) );
  NR2 U29 ( .A(n210), .B(acc_in_reg[22]), .Z(n12) );
  NR2 U30 ( .A(n210), .B(acc_in_reg[21]), .Z(n213) );
  NR2 U31 ( .A(n12), .B(n213), .Z(n13) );
  ND2 U32 ( .A(n214), .B(n13), .Z(n17) );
  NR2 U33 ( .A(n210), .B(acc_in_reg[16]), .Z(n14) );
  NR2 U34 ( .A(mul_reg[15]), .B(acc_in_reg[15]), .Z(n183) );
  NR2 U35 ( .A(n14), .B(n183), .Z(n188) );
  NR2 U36 ( .A(n210), .B(acc_in_reg[18]), .Z(n15) );
  NR2 U37 ( .A(n210), .B(acc_in_reg[17]), .Z(n178) );
  NR2 U38 ( .A(n15), .B(n178), .Z(n16) );
  ND2 U39 ( .A(n188), .B(n16), .Z(n219) );
  NR2 U40 ( .A(n17), .B(n219), .Z(n46) );
  NR2 U41 ( .A(mul_reg[14]), .B(acc_in_reg[14]), .Z(n21) );
  ND2 U42 ( .A(mul_reg[14]), .B(acc_in_reg[14]), .Z(n20) );
  AN2P U43 ( .A(mul_reg[13]), .B(acc_in_reg[13]), .Z(n19) );
  OR2P U44 ( .A(mul_reg[13]), .B(acc_in_reg[13]), .Z(n18) );
  AN2P U45 ( .A(mul_reg[12]), .B(acc_in_reg[12]), .Z(n165) );
  MUX21L U46 ( .A(n19), .B(n18), .S(n165), .Z(n130) );
  MUX21L U47 ( .A(n21), .B(n20), .S(n130), .Z(n29) );
  OR2P U48 ( .A(mul_reg[12]), .B(acc_in_reg[12]), .Z(n164) );
  MUX21L U49 ( .A(n19), .B(n18), .S(n164), .Z(n132) );
  MUX21L U50 ( .A(n21), .B(n20), .S(n132), .Z(n28) );
  AN2P U51 ( .A(mul_reg[11]), .B(acc_in_reg[11]), .Z(n23) );
  OR2P U52 ( .A(mul_reg[11]), .B(acc_in_reg[11]), .Z(n22) );
  OR2P U53 ( .A(mul_reg[10]), .B(acc_in_reg[10]), .Z(n141) );
  MUX21L U54 ( .A(n23), .B(n22), .S(n141), .Z(n27) );
  AN2P U55 ( .A(mul_reg[10]), .B(acc_in_reg[10]), .Z(n142) );
  MUX21L U56 ( .A(n23), .B(n22), .S(n142), .Z(n26) );
  AN2P U57 ( .A(mul_reg[9]), .B(acc_in_reg[9]), .Z(n25) );
  OR2P U58 ( .A(mul_reg[9]), .B(acc_in_reg[9]), .Z(n24) );
  OR2P U59 ( .A(mul_reg[8]), .B(acc_in_reg[8]), .Z(n151) );
  MUX21L U60 ( .A(n25), .B(n24), .S(n151), .Z(n156) );
  MUX21L U61 ( .A(n27), .B(n26), .S(n156), .Z(n160) );
  MUX21L U62 ( .A(n29), .B(n28), .S(n160), .Z(n45) );
  AN2P U63 ( .A(mul_reg[8]), .B(acc_in_reg[8]), .Z(n149) );
  MUX21L U64 ( .A(n25), .B(n24), .S(n149), .Z(n154) );
  MUX21L U65 ( .A(n27), .B(n26), .S(n154), .Z(n159) );
  MUX21L U66 ( .A(n29), .B(n28), .S(n159), .Z(n44) );
  AN2P U67 ( .A(mul_reg[7]), .B(acc_in_reg[7]), .Z(n31) );
  OR2P U68 ( .A(mul_reg[7]), .B(acc_in_reg[7]), .Z(n30) );
  OR2P U69 ( .A(mul_reg[6]), .B(acc_in_reg[6]), .Z(n122) );
  MUX21L U70 ( .A(n31), .B(n30), .S(n122), .Z(n35) );
  AN2P U71 ( .A(mul_reg[6]), .B(acc_in_reg[6]), .Z(n123) );
  MUX21L U72 ( .A(n31), .B(n30), .S(n123), .Z(n34) );
  AN2P U73 ( .A(mul_reg[5]), .B(acc_in_reg[5]), .Z(n33) );
  OR2P U74 ( .A(mul_reg[5]), .B(acc_in_reg[5]), .Z(n32) );
  AN2P U75 ( .A(mul_reg[4]), .B(acc_in_reg[4]), .Z(n235) );
  MUX21L U76 ( .A(n33), .B(n32), .S(n235), .Z(n229) );
  MUX21L U77 ( .A(n35), .B(n34), .S(n229), .Z(n43) );
  OR2P U78 ( .A(mul_reg[4]), .B(acc_in_reg[4]), .Z(n234) );
  MUX21L U79 ( .A(n33), .B(n32), .S(n234), .Z(n227) );
  MUX21L U80 ( .A(n35), .B(n34), .S(n227), .Z(n42) );
  AN2P U81 ( .A(mul_reg[3]), .B(acc_in_reg[3]), .Z(n37) );
  OR2P U82 ( .A(mul_reg[3]), .B(acc_in_reg[3]), .Z(n36) );
  OR2P U83 ( .A(mul_reg[2]), .B(acc_in_reg[2]), .Z(n117) );
  MUX21L U84 ( .A(n37), .B(n36), .S(n117), .Z(n41) );
  AN2P U85 ( .A(mul_reg[2]), .B(acc_in_reg[2]), .Z(n118) );
  MUX21L U86 ( .A(n37), .B(n36), .S(n118), .Z(n40) );
  AN2P U87 ( .A(mul_reg[1]), .B(acc_in_reg[1]), .Z(n39) );
  OR2P U88 ( .A(mul_reg[1]), .B(acc_in_reg[1]), .Z(n38) );
  AN2P U89 ( .A(mul_reg[0]), .B(acc_in_reg[0]), .Z(n115) );
  MUX21LP U90 ( .A(n39), .B(n38), .S(n115), .Z(n240) );
  MUX21L U91 ( .A(n41), .B(n40), .S(n240), .Z(n225) );
  MUX21L U92 ( .A(n43), .B(n42), .S(n225), .Z(n146) );
  MUX21LP U93 ( .A(n45), .B(n44), .S(n146), .Z(n222) );
  MUX21LP U94 ( .A(n47), .B(n46), .S(n222), .Z(n139) );
  MUX21L U95 ( .A(n49), .B(n48), .S(n139), .Z(N41) );
  NR2 U96 ( .A(n210), .B(acc_in_reg[24]), .Z(n51) );
  NR2 U97 ( .A(n51), .B(n50), .Z(n65) );
  EO U98 ( .A(n210), .B(acc_in_reg[25]), .Z(n54) );
  EN U99 ( .A(n65), .B(n54), .Z(n56) );
  ND2 U100 ( .A(mul_reg[15]), .B(acc_in_reg[24]), .Z(n52) );
  ND2 U101 ( .A(n53), .B(n52), .Z(n69) );
  EN U102 ( .A(n69), .B(n54), .Z(n55) );
  MUX21L U103 ( .A(n56), .B(n55), .S(n139), .Z(N42) );
  EN U104 ( .A(n210), .B(acc_in_reg[26]), .Z(n57) );
  ND2 U105 ( .A(mul_reg[15]), .B(acc_in_reg[25]), .Z(n68) );
  EO U106 ( .A(n57), .B(n68), .Z(n59) );
  NR2 U107 ( .A(mul_reg[15]), .B(acc_in_reg[25]), .Z(n62) );
  EO U108 ( .A(n62), .B(n57), .Z(n58) );
  MUX21L U109 ( .A(n59), .B(n58), .S(n65), .Z(n61) );
  MUX21L U110 ( .A(n59), .B(n58), .S(n69), .Z(n60) );
  MUX21L U111 ( .A(n61), .B(n60), .S(n139), .Z(N43) );
  NR2 U112 ( .A(mul_reg[15]), .B(acc_in_reg[26]), .Z(n63) );
  NR2 U113 ( .A(n63), .B(n62), .Z(n64) );
  ND2 U114 ( .A(n65), .B(n64), .Z(n102) );
  IVP U115 ( .A(n102), .Z(n66) );
  EO U116 ( .A(n210), .B(acc_in_reg[27]), .Z(n71) );
  EN U117 ( .A(n66), .B(n71), .Z(n74) );
  ND2 U118 ( .A(mul_reg[15]), .B(acc_in_reg[26]), .Z(n67) );
  ND2 U119 ( .A(n68), .B(n67), .Z(n70) );
  NR2 U120 ( .A(n70), .B(n69), .Z(n110) );
  IVP U121 ( .A(n110), .Z(n72) );
  EN U122 ( .A(n72), .B(n71), .Z(n73) );
  MUX21L U123 ( .A(n74), .B(n73), .S(n139), .Z(N44) );
  NR2 U124 ( .A(n210), .B(acc_in_reg[27]), .Z(n80) );
  EN U125 ( .A(n210), .B(acc_in_reg[28]), .Z(n75) );
  EN U126 ( .A(n80), .B(n75), .Z(n77) );
  ND2 U127 ( .A(mul_reg[15]), .B(acc_in_reg[27]), .Z(n85) );
  EN U128 ( .A(n75), .B(n85), .Z(n76) );
  MUX21H U129 ( .A(n77), .B(n76), .S(n102), .Z(n79) );
  MUX21H U130 ( .A(n77), .B(n76), .S(n110), .Z(n78) );
  MUX21L U131 ( .A(n79), .B(n78), .S(n139), .Z(N45) );
  NR2 U132 ( .A(n210), .B(acc_in_reg[28]), .Z(n81) );
  NR2 U133 ( .A(n81), .B(n80), .Z(n101) );
  IVP U134 ( .A(n101), .Z(n82) );
  NR2 U135 ( .A(n82), .B(n102), .Z(n83) );
  EO U136 ( .A(n210), .B(acc_in_reg[29]), .Z(n87) );
  EN U137 ( .A(n83), .B(n87), .Z(n90) );
  ND2 U138 ( .A(n210), .B(acc_in_reg[28]), .Z(n84) );
  ND2 U139 ( .A(n85), .B(n84), .Z(n107) );
  IVP U140 ( .A(n107), .Z(n86) );
  ND2 U141 ( .A(n110), .B(n86), .Z(n88) );
  EN U142 ( .A(n88), .B(n87), .Z(n89) );
  MUX21L U143 ( .A(n90), .B(n89), .S(n139), .Z(N46) );
  EN U144 ( .A(n210), .B(acc_in_reg[30]), .Z(n91) );
  ND2 U145 ( .A(n210), .B(acc_in_reg[29]), .Z(n106) );
  EO U146 ( .A(n91), .B(n106), .Z(n93) );
  NR2 U147 ( .A(n210), .B(acc_in_reg[29]), .Z(n98) );
  EO U148 ( .A(n98), .B(n91), .Z(n92) );
  MUX21L U149 ( .A(n93), .B(n92), .S(n101), .Z(n95) );
  MUX21L U150 ( .A(n93), .B(n92), .S(n107), .Z(n94) );
  MUX21H U151 ( .A(n95), .B(n94), .S(n102), .Z(n97) );
  MUX21H U152 ( .A(n95), .B(n94), .S(n110), .Z(n96) );
  MUX21L U153 ( .A(n97), .B(n96), .S(n139), .Z(N47) );
  NR2 U154 ( .A(n210), .B(acc_in_reg[30]), .Z(n99) );
  NR2 U155 ( .A(n99), .B(n98), .Z(n100) );
  ND2 U156 ( .A(n101), .B(n100), .Z(n103) );
  NR2 U157 ( .A(n103), .B(n102), .Z(n104) );
  EO U158 ( .A(n210), .B(acc_in_reg[31]), .Z(n111) );
  EN U159 ( .A(n104), .B(n111), .Z(n114) );
  ND2 U160 ( .A(mul_reg[15]), .B(acc_in_reg[30]), .Z(n105) );
  ND2 U161 ( .A(n106), .B(n105), .Z(n108) );
  NR2 U162 ( .A(n108), .B(n107), .Z(n109) );
  ND2 U163 ( .A(n110), .B(n109), .Z(n112) );
  EN U164 ( .A(n112), .B(n111), .Z(n113) );
  MUX21L U165 ( .A(n114), .B(n113), .S(n139), .Z(N48) );
  EO U166 ( .A(mul_reg[0]), .B(acc_in_reg[0]), .Z(N17) );
  EO U167 ( .A(mul_reg[1]), .B(acc_in_reg[1]), .Z(n116) );
  EO U168 ( .A(n116), .B(n115), .Z(N18) );
  EN U169 ( .A(mul_reg[3]), .B(acc_in_reg[3]), .Z(n119) );
  EN U170 ( .A(n117), .B(n119), .Z(n121) );
  EN U171 ( .A(n119), .B(n118), .Z(n120) );
  MUX21H U172 ( .A(n121), .B(n120), .S(n240), .Z(N20) );
  EO U173 ( .A(mul_reg[7]), .B(acc_in_reg[7]), .Z(n124) );
  EN U174 ( .A(n122), .B(n124), .Z(n126) );
  EN U175 ( .A(n124), .B(n123), .Z(n125) );
  MUX21L U176 ( .A(n126), .B(n125), .S(n229), .Z(n129) );
  MUX21L U177 ( .A(n126), .B(n125), .S(n227), .Z(n128) );
  MUX21H U178 ( .A(n129), .B(n128), .S(n127), .Z(N24) );
  EO U179 ( .A(mul_reg[14]), .B(acc_in_reg[14]), .Z(n131) );
  EO U180 ( .A(n130), .B(n131), .Z(n134) );
  EO U181 ( .A(n132), .B(n131), .Z(n133) );
  MUX21L U182 ( .A(n134), .B(n133), .S(n160), .Z(n137) );
  MUX21L U183 ( .A(n134), .B(n133), .S(n159), .Z(n136) );
  MUX21H U184 ( .A(n137), .B(n136), .S(n135), .Z(N31) );
  EN U185 ( .A(n210), .B(acc_in_reg[23]), .Z(n138) );
  EO U186 ( .A(n139), .B(n138), .Z(N40) );
  EN U187 ( .A(n210), .B(acc_in_reg[15]), .Z(n140) );
  EO U188 ( .A(n207), .B(n140), .Z(N32) );
  EN U189 ( .A(mul_reg[11]), .B(acc_in_reg[11]), .Z(n143) );
  EN U190 ( .A(n141), .B(n143), .Z(n145) );
  EN U191 ( .A(n143), .B(n142), .Z(n144) );
  MUX21L U192 ( .A(n145), .B(n144), .S(n154), .Z(n148) );
  MUX21L U193 ( .A(n145), .B(n144), .S(n156), .Z(n147) );
  IVDA U194 ( .A(n146), .Y(n244), .Z(n135) );
  MUX21L U195 ( .A(n148), .B(n147), .S(n244), .Z(N28) );
  EN U196 ( .A(mul_reg[9]), .B(acc_in_reg[9]), .Z(n150) );
  EO U197 ( .A(n150), .B(n149), .Z(n153) );
  EO U198 ( .A(n151), .B(n150), .Z(n152) );
  MUX21L U199 ( .A(n153), .B(n152), .S(n244), .Z(N26) );
  EO U200 ( .A(mul_reg[10]), .B(acc_in_reg[10]), .Z(n155) );
  EO U201 ( .A(n154), .B(n155), .Z(n158) );
  EO U202 ( .A(n156), .B(n155), .Z(n157) );
  MUX21L U203 ( .A(n158), .B(n157), .S(n244), .Z(N27) );
  IVP U204 ( .A(n159), .Z(n167) );
  EO U205 ( .A(mul_reg[12]), .B(acc_in_reg[12]), .Z(n161) );
  EO U206 ( .A(n167), .B(n161), .Z(n163) );
  IVP U207 ( .A(n160), .Z(n168) );
  EO U208 ( .A(n168), .B(n161), .Z(n162) );
  MUX21L U209 ( .A(n163), .B(n162), .S(n244), .Z(N29) );
  EN U210 ( .A(mul_reg[13]), .B(acc_in_reg[13]), .Z(n166) );
  EN U211 ( .A(n164), .B(n166), .Z(n170) );
  EN U212 ( .A(n166), .B(n165), .Z(n169) );
  MUX21L U213 ( .A(n170), .B(n169), .S(n167), .Z(n172) );
  MUX21L U214 ( .A(n170), .B(n169), .S(n168), .Z(n171) );
  MUX21L U215 ( .A(n172), .B(n171), .S(n244), .Z(N30) );
  IVP U216 ( .A(n219), .Z(n203) );
  EO U217 ( .A(n210), .B(acc_in_reg[19]), .Z(n173) );
  EN U218 ( .A(n203), .B(n173), .Z(n175) );
  IVP U219 ( .A(n218), .Z(n204) );
  EN U220 ( .A(n204), .B(n173), .Z(n174) );
  MUX21L U221 ( .A(n175), .B(n174), .S(n207), .Z(N36) );
  EN U222 ( .A(n210), .B(acc_in_reg[18]), .Z(n177) );
  EO U223 ( .A(n177), .B(n176), .Z(n180) );
  EO U224 ( .A(n178), .B(n177), .Z(n179) );
  MUX21L U225 ( .A(n180), .B(n179), .S(n188), .Z(n182) );
  MUX21L U226 ( .A(n180), .B(n179), .S(n190), .Z(n181) );
  MUX21L U227 ( .A(n182), .B(n181), .S(n207), .Z(N35) );
  EN U228 ( .A(n210), .B(acc_in_reg[16]), .Z(n185) );
  EN U229 ( .A(n183), .B(n185), .Z(n187) );
  EN U230 ( .A(n185), .B(n184), .Z(n186) );
  MUX21L U231 ( .A(n187), .B(n186), .S(n207), .Z(N33) );
  EO U232 ( .A(n210), .B(acc_in_reg[17]), .Z(n189) );
  EN U233 ( .A(n188), .B(n189), .Z(n192) );
  EN U234 ( .A(n190), .B(n189), .Z(n191) );
  MUX21L U235 ( .A(n192), .B(n191), .S(n207), .Z(N34) );
  IVP U236 ( .A(n215), .Z(n193) );
  EN U237 ( .A(n210), .B(acc_in_reg[21]), .Z(n194) );
  EO U238 ( .A(n193), .B(n194), .Z(n197) );
  IVP U239 ( .A(n214), .Z(n195) );
  EO U240 ( .A(n195), .B(n194), .Z(n196) );
  MUX21L U241 ( .A(n197), .B(n196), .S(n203), .Z(n199) );
  MUX21L U242 ( .A(n197), .B(n196), .S(n204), .Z(n198) );
  MUX21L U243 ( .A(n199), .B(n198), .S(n207), .Z(N38) );
  EN U244 ( .A(n210), .B(acc_in_reg[20]), .Z(n201) );
  EO U245 ( .A(n201), .B(n200), .Z(n206) );
  EO U246 ( .A(n202), .B(n201), .Z(n205) );
  MUX21L U247 ( .A(n206), .B(n205), .S(n203), .Z(n209) );
  MUX21L U248 ( .A(n206), .B(n205), .S(n204), .Z(n208) );
  MUX21L U249 ( .A(n209), .B(n208), .S(n207), .Z(N37) );
  EO U250 ( .A(n210), .B(acc_in_reg[22]), .Z(n212) );
  EO U251 ( .A(n212), .B(n211), .Z(n217) );
  EO U252 ( .A(n213), .B(n212), .Z(n216) );
  MUX21L U253 ( .A(n217), .B(n216), .S(n214), .Z(n221) );
  MUX21L U254 ( .A(n217), .B(n216), .S(n215), .Z(n220) );
  MUX21L U255 ( .A(n221), .B(n220), .S(n218), .Z(n224) );
  MUX21L U256 ( .A(n221), .B(n220), .S(n219), .Z(n223) );
  MUX21L U257 ( .A(n224), .B(n223), .S(n222), .Z(N39) );
  IVDA U258 ( .A(n225), .Y(n237), .Z(n127) );
  EN U259 ( .A(mul_reg[4]), .B(acc_in_reg[4]), .Z(n226) );
  EO U260 ( .A(n237), .B(n226), .Z(N21) );
  IVP U261 ( .A(n227), .Z(n228) );
  EO U262 ( .A(mul_reg[6]), .B(acc_in_reg[6]), .Z(n230) );
  EN U263 ( .A(n228), .B(n230), .Z(n233) );
  IVP U264 ( .A(n229), .Z(n231) );
  EN U265 ( .A(n231), .B(n230), .Z(n232) );
  MUX21L U266 ( .A(n233), .B(n232), .S(n237), .Z(N23) );
  EO U267 ( .A(mul_reg[5]), .B(acc_in_reg[5]), .Z(n236) );
  EN U268 ( .A(n234), .B(n236), .Z(n239) );
  EN U269 ( .A(n236), .B(n235), .Z(n238) );
  MUX21L U270 ( .A(n239), .B(n238), .S(n237), .Z(N22) );
  IVP U271 ( .A(n240), .Z(n242) );
  EN U272 ( .A(mul_reg[2]), .B(acc_in_reg[2]), .Z(n241) );
  EN U273 ( .A(n242), .B(n241), .Z(N19) );
  EN U274 ( .A(mul_reg[8]), .B(acc_in_reg[8]), .Z(n243) );
  EN U275 ( .A(n244), .B(n243), .Z(N25) );
  IVP U276 ( .A(x[7]), .Z(n302) );
  IVP U277 ( .A(w[7]), .Z(n287) );
  AO4 U278 ( .A(n302), .B(w[7]), .C(n287), .D(x[7]), .Z(n252) );
  IVP U279 ( .A(x[6]), .Z(n268) );
  IVP U280 ( .A(x[5]), .Z(n349) );
  AO2 U281 ( .A(x[5]), .B(x[6]), .C(n268), .D(n349), .Z(n316) );
  ND2 U282 ( .A(n252), .B(n316), .Z(n254) );
  NR2 U283 ( .A(x[6]), .B(x[7]), .Z(n245) );
  AO1P U284 ( .A(x[6]), .B(x[7]), .C(n316), .D(n245), .Z(n318) );
  IVP U285 ( .A(w[6]), .Z(n273) );
  AO4 U286 ( .A(n302), .B(w[6]), .C(n273), .D(x[7]), .Z(n255) );
  ND2 U287 ( .A(n318), .B(n255), .Z(n246) );
  ND2 U288 ( .A(n254), .B(n246), .Z(\intadd_0/A[10] ) );
  ND2 U289 ( .A(x[0]), .B(x[1]), .Z(n393) );
  IVP U290 ( .A(x[0]), .Z(n251) );
  NR2 U291 ( .A(x[1]), .B(n251), .Z(n388) );
  ND2 U292 ( .A(n388), .B(w[2]), .Z(n248) );
  ND2 U293 ( .A(n251), .B(x[1]), .Z(n338) );
  IVP U294 ( .A(n338), .Z(n390) );
  IVP U295 ( .A(w[1]), .Z(n341) );
  ND2 U296 ( .A(n390), .B(n341), .Z(n247) );
  AO3 U297 ( .A(w[2]), .B(n393), .C(n248), .D(n247), .Z(n384) );
  IVP U298 ( .A(x[1]), .Z(n249) );
  IVP U299 ( .A(x[2]), .Z(n260) );
  AO4 U300 ( .A(n249), .B(x[2]), .C(n260), .D(x[1]), .Z(n363) );
  IVP U301 ( .A(n363), .Z(n398) );
  IVP U302 ( .A(w[0]), .Z(n396) );
  AO3 U303 ( .A(n251), .B(n341), .C(x[1]), .D(n396), .Z(n250) );
  AO7 U304 ( .A(n398), .B(n396), .C(n250), .Z(n383) );
  EO U305 ( .A(n384), .B(n383), .Z(N3) );
  NR2 U306 ( .A(n251), .B(n396), .Z(N1) );
  IVP U307 ( .A(\intadd_0/SUM[8] ), .Z(N12) );
  IVP U308 ( .A(\intadd_0/SUM[7] ), .Z(N11) );
  IVP U309 ( .A(\intadd_0/SUM[6] ), .Z(N10) );
  IVP U310 ( .A(\intadd_0/SUM[5] ), .Z(N9) );
  IVP U311 ( .A(\intadd_0/SUM[11] ), .Z(N15) );
  IVP U312 ( .A(\intadd_0/A[10] ), .Z(\intadd_0/B[11] ) );
  ND2 U313 ( .A(n252), .B(n318), .Z(n253) );
  ND2 U314 ( .A(n254), .B(n253), .Z(\intadd_0/A[11] ) );
  IVP U315 ( .A(\intadd_0/SUM[3] ), .Z(N7) );
  IVP U316 ( .A(\intadd_0/SUM[2] ), .Z(N6) );
  IVP U317 ( .A(\intadd_0/SUM[1] ), .Z(N5) );
  IVP U318 ( .A(\intadd_0/SUM[10] ), .Z(N14) );
  IVP U319 ( .A(w[5]), .Z(n295) );
  AO2 U320 ( .A(x[7]), .B(w[5]), .C(n295), .D(n302), .Z(n317) );
  AO2 U321 ( .A(n317), .B(n318), .C(n255), .D(n316), .Z(n331) );
  AO2 U322 ( .A(x[5]), .B(w[6]), .C(n273), .D(n349), .Z(n306) );
  EON1 U323 ( .A(x[4]), .B(x[3]), .C(x[3]), .D(x[4]), .Z(n353) );
  ND2 U324 ( .A(x[4]), .B(x[5]), .Z(n256) );
  AO3 U325 ( .A(x[4]), .B(x[5]), .C(n353), .D(n256), .Z(n350) );
  IVP U326 ( .A(n350), .Z(n307) );
  AO2 U327 ( .A(x[5]), .B(n287), .C(w[7]), .D(n349), .Z(n257) );
  EO1 U328 ( .A(n306), .B(n307), .C(n353), .D(n257), .Z(n335) );
  AO6 U329 ( .A(n353), .B(n350), .C(n257), .Z(n329) );
  AO5 U330 ( .A(n331), .B(n335), .C(n329), .Z(n258) );
  IVP U331 ( .A(n258), .Z(\intadd_0/B[10] ) );
  IVP U332 ( .A(\intadd_0/SUM[9] ), .Z(N13) );
  IVP U333 ( .A(w[2]), .Z(n389) );
  AO4 U334 ( .A(n389), .B(x[7]), .C(n302), .D(w[2]), .Z(n294) );
  AO2 U335 ( .A(w[1]), .B(x[7]), .C(n302), .D(n341), .Z(n266) );
  AO2 U336 ( .A(n316), .B(n294), .C(n266), .D(n318), .Z(n259) );
  IVP U337 ( .A(n259), .Z(n264) );
  ND2 U338 ( .A(x[1]), .B(n260), .Z(n262) );
  ND2 U339 ( .A(x[2]), .B(x[3]), .Z(n261) );
  AO3 U340 ( .A(x[3]), .B(x[1]), .C(n262), .D(n261), .Z(n387) );
  IVP U341 ( .A(x[3]), .Z(n399) );
  AO2 U342 ( .A(x[3]), .B(n295), .C(w[5]), .D(n399), .Z(n270) );
  AO2 U343 ( .A(x[3]), .B(n273), .C(w[6]), .D(n399), .Z(n288) );
  AO4 U344 ( .A(n387), .B(n270), .C(n398), .D(n288), .Z(n263) );
  NR2 U345 ( .A(n264), .B(n263), .Z(n310) );
  AO6 U346 ( .A(n264), .B(n263), .C(n310), .Z(n298) );
  IVP U347 ( .A(n353), .Z(n305) );
  IVP U348 ( .A(w[4]), .Z(n344) );
  AO4 U349 ( .A(n349), .B(w[4]), .C(n344), .D(x[5]), .Z(n296) );
  IVP U350 ( .A(w[3]), .Z(n347) );
  AO2 U351 ( .A(w[3]), .B(x[5]), .C(n349), .D(n347), .Z(n269) );
  AO2 U352 ( .A(n305), .B(n296), .C(n269), .D(n307), .Z(n265) );
  IVP U353 ( .A(n265), .Z(n293) );
  AO2 U354 ( .A(w[0]), .B(x[7]), .C(n302), .D(n396), .Z(n267) );
  AO2 U355 ( .A(n267), .B(n318), .C(n266), .D(n316), .Z(n280) );
  ND2 U356 ( .A(w[0]), .B(n316), .Z(n369) );
  AO3 U357 ( .A(n268), .B(n349), .C(x[7]), .D(n369), .Z(n279) );
  NR2 U358 ( .A(n280), .B(n279), .Z(n292) );
  OR2P U359 ( .A(n287), .B(n388), .Z(n272) );
  AO7 U360 ( .A(x[1]), .B(w[7]), .C(n272), .Z(n291) );
  AO2 U361 ( .A(w[2]), .B(x[5]), .C(n349), .D(n389), .Z(n278) );
  AO2 U362 ( .A(n278), .B(n307), .C(n269), .D(n305), .Z(n285) );
  AO2 U363 ( .A(x[3]), .B(w[4]), .C(n344), .D(n399), .Z(n364) );
  IVP U364 ( .A(n387), .Z(n365) );
  EO1 U365 ( .A(n364), .B(n365), .C(n270), .D(n398), .Z(n284) );
  ND2 U366 ( .A(n393), .B(n287), .Z(n271) );
  AO2 U367 ( .A(n390), .B(n273), .C(n272), .D(n271), .Z(n283) );
  IVP U368 ( .A(n274), .Z(n299) );
  EN U369 ( .A(n301), .B(n299), .Z(n275) );
  EN U370 ( .A(n298), .B(n275), .Z(\intadd_0/B[5] ) );
  NR2 U371 ( .A(w[5]), .B(n338), .Z(n277) );
  NR2 U372 ( .A(n393), .B(w[6]), .Z(n276) );
  AO1P U373 ( .A(w[6]), .B(n388), .C(n277), .D(n276), .Z(n368) );
  AO2 U374 ( .A(w[1]), .B(n349), .C(x[5]), .D(n341), .Z(n352) );
  EO1 U375 ( .A(n278), .B(n305), .C(n352), .D(n350), .Z(n367) );
  IVP U376 ( .A(n292), .Z(n282) );
  ND2 U377 ( .A(n280), .B(n279), .Z(n281) );
  ND2 U378 ( .A(n282), .B(n281), .Z(n376) );
  FA1A U379 ( .A(n285), .B(n284), .CI(n283), .CO(n274), .S(n375) );
  AO5 U380 ( .A(n378), .B(n376), .C(n375), .Z(n286) );
  IVP U381 ( .A(n286), .Z(\intadd_0/A[5] ) );
  AO2 U382 ( .A(x[3]), .B(n287), .C(w[7]), .D(n399), .Z(n304) );
  NR2 U383 ( .A(n304), .B(n398), .Z(n290) );
  NR2 U384 ( .A(n288), .B(n387), .Z(n289) );
  NR2 U385 ( .A(n290), .B(n289), .Z(n322) );
  FA1A U386 ( .A(n293), .B(n292), .CI(n291), .CO(n315), .S(n301) );
  AO2 U387 ( .A(w[3]), .B(x[7]), .C(n302), .D(n347), .Z(n303) );
  AO2 U388 ( .A(n294), .B(n318), .C(n303), .D(n316), .Z(n311) );
  AO2 U389 ( .A(x[5]), .B(w[5]), .C(n295), .D(n349), .Z(n308) );
  AO2 U390 ( .A(n296), .B(n307), .C(n308), .D(n305), .Z(n309) );
  EN U391 ( .A(n315), .B(n313), .Z(n297) );
  EN U392 ( .A(n322), .B(n297), .Z(\intadd_0/B[6] ) );
  IVP U393 ( .A(n298), .Z(n300) );
  AO5 U394 ( .A(n301), .B(n300), .C(n299), .Z(\intadd_0/A[6] ) );
  AO2 U395 ( .A(x[7]), .B(w[4]), .C(n344), .D(n302), .Z(n319) );
  AO2 U396 ( .A(n303), .B(n318), .C(n319), .D(n316), .Z(n327) );
  AO6 U397 ( .A(n398), .B(n387), .C(n304), .Z(n321) );
  AO2 U398 ( .A(n308), .B(n307), .C(n306), .D(n305), .Z(n320) );
  FA1A U399 ( .A(n311), .B(n310), .CI(n309), .CO(n325), .S(n313) );
  EN U400 ( .A(n326), .B(n325), .Z(n312) );
  EN U401 ( .A(n327), .B(n312), .Z(\intadd_0/B[7] ) );
  IVP U402 ( .A(n313), .Z(n314) );
  AO5 U403 ( .A(n315), .B(n322), .C(n314), .Z(\intadd_0/A[7] ) );
  AO2 U404 ( .A(n319), .B(n318), .C(n317), .D(n316), .Z(n332) );
  FA1A U405 ( .A(n322), .B(n321), .CI(n320), .CO(n323), .S(n326) );
  IVP U406 ( .A(n323), .Z(n333) );
  EN U407 ( .A(n335), .B(n333), .Z(n324) );
  EN U408 ( .A(n332), .B(n324), .Z(\intadd_0/B[8] ) );
  AO5 U409 ( .A(n327), .B(n326), .C(n325), .Z(n328) );
  IVP U410 ( .A(n328), .Z(\intadd_0/A[8] ) );
  EN U411 ( .A(n335), .B(n329), .Z(n330) );
  EN U412 ( .A(n331), .B(n330), .Z(\intadd_0/B[9] ) );
  IVP U413 ( .A(n332), .Z(n334) );
  AO5 U414 ( .A(n335), .B(n334), .C(n333), .Z(\intadd_0/A[9] ) );
  ND2 U415 ( .A(x[0]), .B(w[1]), .Z(n337) );
  ND2 U416 ( .A(w[0]), .B(x[1]), .Z(n336) );
  EO U417 ( .A(n337), .B(n336), .Z(N2) );
  IVP U418 ( .A(\intadd_0/SUM[4] ), .Z(N8) );
  NR2 U419 ( .A(n396), .B(n353), .Z(n362) );
  NR2 U420 ( .A(w[3]), .B(n338), .Z(n340) );
  NR2 U421 ( .A(n393), .B(w[4]), .Z(n339) );
  AO1P U422 ( .A(w[4]), .B(n388), .C(n340), .D(n339), .Z(n359) );
  AO2 U423 ( .A(w[2]), .B(x[3]), .C(n399), .D(n389), .Z(n348) );
  AO2 U424 ( .A(w[1]), .B(n399), .C(x[3]), .D(n341), .Z(n385) );
  EO1 U425 ( .A(n348), .B(n363), .C(n385), .D(n387), .Z(n342) );
  IVP U426 ( .A(n342), .Z(n360) );
  EN U427 ( .A(n359), .B(n360), .Z(n343) );
  EN U428 ( .A(n362), .B(n343), .Z(\intadd_0/B[1] ) );
  ND2 U429 ( .A(n388), .B(w[5]), .Z(n346) );
  ND2 U430 ( .A(n390), .B(n344), .Z(n345) );
  AO3 U431 ( .A(w[5]), .B(n393), .C(n346), .D(n345), .Z(n374) );
  AO4 U432 ( .A(n399), .B(w[3]), .C(n347), .D(x[3]), .Z(n366) );
  AO2 U433 ( .A(n365), .B(n348), .C(n363), .D(n366), .Z(n371) );
  AO1P U434 ( .A(x[4]), .B(x[3]), .C(n362), .D(n349), .Z(n355) );
  AO2 U435 ( .A(w[0]), .B(n349), .C(x[5]), .D(n396), .Z(n351) );
  AO4 U436 ( .A(n353), .B(n352), .C(n351), .D(n350), .Z(n354) );
  ND2 U437 ( .A(n355), .B(n354), .Z(n379) );
  IVP U438 ( .A(n379), .Z(n357) );
  NR2 U439 ( .A(n355), .B(n354), .Z(n356) );
  NR2 U440 ( .A(n357), .B(n356), .Z(n373) );
  EN U441 ( .A(n371), .B(n373), .Z(n358) );
  EN U442 ( .A(n374), .B(n358), .Z(\intadd_0/B[2] ) );
  IVP U443 ( .A(n359), .Z(n361) );
  AO5 U444 ( .A(n362), .B(n361), .C(n360), .Z(\intadd_0/A[2] ) );
  AO2 U445 ( .A(n366), .B(n365), .C(n364), .D(n363), .Z(n381) );
  FA1A U446 ( .A(n369), .B(n368), .CI(n367), .CO(n378), .S(n380) );
  EN U447 ( .A(n380), .B(n379), .Z(n370) );
  EN U448 ( .A(n381), .B(n370), .Z(\intadd_0/B[3] ) );
  IVP U449 ( .A(n371), .Z(n372) );
  AO5 U450 ( .A(n374), .B(n373), .C(n372), .Z(\intadd_0/A[3] ) );
  EN U451 ( .A(n376), .B(n375), .Z(n377) );
  EN U452 ( .A(n378), .B(n377), .Z(\intadd_0/B[4] ) );
  AO5 U453 ( .A(n381), .B(n380), .C(n379), .Z(n382) );
  IVP U454 ( .A(n382), .Z(\intadd_0/A[4] ) );
  IVP U455 ( .A(\intadd_0/SUM[0] ), .Z(N4) );
  ND2 U456 ( .A(n384), .B(n383), .Z(\intadd_0/CI ) );
  AO2 U457 ( .A(w[0]), .B(n399), .C(x[3]), .D(n396), .Z(n386) );
  AO4 U458 ( .A(n387), .B(n386), .C(n398), .D(n385), .Z(n395) );
  ND2 U459 ( .A(n388), .B(w[3]), .Z(n392) );
  ND2 U460 ( .A(n390), .B(n389), .Z(n391) );
  AO3 U461 ( .A(w[3]), .B(n393), .C(n392), .D(n391), .Z(n394) );
  ND2 U462 ( .A(n395), .B(n394), .Z(\intadd_0/A[1] ) );
  AO7 U463 ( .A(n395), .B(n394), .C(\intadd_0/A[1] ), .Z(\intadd_0/B[0] ) );
  ND2 U464 ( .A(x[3]), .B(n396), .Z(n397) );
  NR2 U465 ( .A(n398), .B(n397), .Z(n401) );
  NR3 U466 ( .A(n399), .B(x[1]), .C(x[2]), .Z(n400) );
  NR2 U467 ( .A(n401), .B(n400), .Z(\intadd_0/A[0] ) );
endmodule


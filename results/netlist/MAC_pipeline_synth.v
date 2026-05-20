/////////////////////////////////////////////////////////////
// Created by: Synopsys DC Ultra(TM) in wire load mode
// Version   : X-2025.06-SP4
// Date      : Tue May 19 21:12:03 2026
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
         N43, N44, N45, N46, N47, N48, \intadd_1/A[11] , \intadd_1/A[10] ,
         \intadd_1/A[9] , \intadd_1/A[8] , \intadd_1/A[7] , \intadd_1/A[6] ,
         \intadd_1/A[5] , \intadd_1/A[4] , \intadd_1/A[3] , \intadd_1/A[2] ,
         \intadd_1/A[1] , \intadd_1/A[0] , \intadd_1/B[11] , \intadd_1/B[10] ,
         \intadd_1/B[9] , \intadd_1/B[8] , \intadd_1/B[7] , \intadd_1/B[6] ,
         \intadd_1/B[5] , \intadd_1/B[4] , \intadd_1/B[3] , \intadd_1/B[2] ,
         \intadd_1/B[1] , \intadd_1/B[0] , \intadd_1/CI , \intadd_1/n12 ,
         \intadd_1/n11 , \intadd_1/n10 , \intadd_1/n9 , \intadd_1/n8 ,
         \intadd_1/n7 , \intadd_1/n6 , \intadd_1/n5 , \intadd_1/n4 ,
         \intadd_1/n3 , \intadd_1/n2 , \intadd_1/n1 , \intadd_0/CI ,
         \intadd_0/n29 , \intadd_0/n28 , \intadd_0/n9 , \intadd_0/n8 , n2, n3,
         n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, n14, n15, n16, n17, n18,
         n19, n20, n21, n22, n23, n24, n25, n26, n27, n28, n29, n30, n31, n32,
         n33, n34, n35, n36, n37, n38, n39, n40, n41, n42, n43, n44, n45, n46,
         n47, n48, n49, n50, n51, n52, n53, n54, n55, n56, n57, n58, n59, n60,
         n61, n62, n63, n64, n65, n66, n67, n68, n69, n70, n71, n72, n73, n74,
         n75, n76, n77, n78, n79, n80, n81, n82, n83, n84, n85, n86, n87, n88,
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
         n256, n257, n258, n259, n260, n261, n262, n263, n264, n265, n266;
  wire   [15:0] mul_reg;
  wire   [31:0] acc_in_reg;

  DFFR_X1 valid_s1_reg ( .D(valid_in), .CK(clk), .RN(rst_n), .Q(valid_s1) );
  DFFR_X1 \mul_reg_reg[14]  ( .D(N15), .CK(clk), .RN(rst_n), .Q(mul_reg[14])
         );
  DFFR_X1 \mul_reg_reg[13]  ( .D(N14), .CK(clk), .RN(rst_n), .Q(mul_reg[13])
         );
  DFFR_X1 \mul_reg_reg[12]  ( .D(N13), .CK(clk), .RN(rst_n), .Q(mul_reg[12])
         );
  DFFR_X1 \mul_reg_reg[11]  ( .D(N12), .CK(clk), .RN(rst_n), .Q(mul_reg[11])
         );
  DFFR_X1 \mul_reg_reg[10]  ( .D(N11), .CK(clk), .RN(rst_n), .Q(mul_reg[10])
         );
  DFFR_X1 \mul_reg_reg[9]  ( .D(N10), .CK(clk), .RN(rst_n), .Q(mul_reg[9]) );
  DFFR_X1 \mul_reg_reg[8]  ( .D(N9), .CK(clk), .RN(rst_n), .Q(mul_reg[8]) );
  DFFR_X1 \mul_reg_reg[7]  ( .D(N8), .CK(clk), .RN(rst_n), .Q(mul_reg[7]) );
  DFFR_X1 \mul_reg_reg[6]  ( .D(N7), .CK(clk), .RN(rst_n), .Q(mul_reg[6]) );
  DFFR_X1 \mul_reg_reg[5]  ( .D(N6), .CK(clk), .RN(rst_n), .Q(mul_reg[5]) );
  DFFR_X1 \mul_reg_reg[4]  ( .D(N5), .CK(clk), .RN(rst_n), .Q(mul_reg[4]) );
  DFFR_X1 \mul_reg_reg[3]  ( .D(N4), .CK(clk), .RN(rst_n), .Q(mul_reg[3]) );
  DFFR_X1 \mul_reg_reg[2]  ( .D(N3), .CK(clk), .RN(rst_n), .Q(mul_reg[2]) );
  DFFR_X1 \mul_reg_reg[1]  ( .D(N2), .CK(clk), .RN(rst_n), .Q(mul_reg[1]) );
  DFFR_X1 \mul_reg_reg[0]  ( .D(N1), .CK(clk), .RN(rst_n), .Q(mul_reg[0]), 
        .QN(n265) );
  DFFR_X1 \acc_in_reg_reg[31]  ( .D(acc_in[31]), .CK(clk), .RN(rst_n), .Q(
        acc_in_reg[31]) );
  DFFR_X1 \acc_in_reg_reg[30]  ( .D(acc_in[30]), .CK(clk), .RN(rst_n), .Q(
        acc_in_reg[30]) );
  DFFR_X1 \acc_in_reg_reg[29]  ( .D(acc_in[29]), .CK(clk), .RN(rst_n), .Q(
        acc_in_reg[29]) );
  DFFR_X1 \acc_in_reg_reg[28]  ( .D(acc_in[28]), .CK(clk), .RN(rst_n), .Q(
        acc_in_reg[28]) );
  DFFR_X1 \acc_in_reg_reg[27]  ( .D(acc_in[27]), .CK(clk), .RN(rst_n), .Q(
        acc_in_reg[27]) );
  DFFR_X1 \acc_in_reg_reg[26]  ( .D(acc_in[26]), .CK(clk), .RN(rst_n), .Q(
        acc_in_reg[26]) );
  DFFR_X1 \acc_in_reg_reg[25]  ( .D(acc_in[25]), .CK(clk), .RN(rst_n), .Q(
        acc_in_reg[25]) );
  DFFR_X1 \acc_in_reg_reg[24]  ( .D(acc_in[24]), .CK(clk), .RN(rst_n), .Q(
        acc_in_reg[24]) );
  DFFR_X1 \acc_in_reg_reg[23]  ( .D(acc_in[23]), .CK(clk), .RN(rst_n), .Q(
        acc_in_reg[23]) );
  DFFR_X1 \acc_in_reg_reg[22]  ( .D(acc_in[22]), .CK(clk), .RN(rst_n), .Q(
        acc_in_reg[22]) );
  DFFR_X1 \acc_in_reg_reg[21]  ( .D(acc_in[21]), .CK(clk), .RN(rst_n), .Q(
        acc_in_reg[21]) );
  DFFR_X1 \acc_in_reg_reg[20]  ( .D(acc_in[20]), .CK(clk), .RN(rst_n), .Q(
        acc_in_reg[20]) );
  DFFR_X1 \acc_in_reg_reg[19]  ( .D(acc_in[19]), .CK(clk), .RN(rst_n), .Q(
        acc_in_reg[19]) );
  DFFR_X1 \acc_in_reg_reg[18]  ( .D(acc_in[18]), .CK(clk), .RN(rst_n), .Q(
        acc_in_reg[18]) );
  DFFR_X1 \acc_in_reg_reg[17]  ( .D(acc_in[17]), .CK(clk), .RN(rst_n), .Q(
        acc_in_reg[17]) );
  DFFR_X1 \acc_in_reg_reg[16]  ( .D(acc_in[16]), .CK(clk), .RN(rst_n), .Q(
        acc_in_reg[16]) );
  DFFR_X1 \acc_in_reg_reg[15]  ( .D(acc_in[15]), .CK(clk), .RN(rst_n), .Q(
        acc_in_reg[15]) );
  DFFR_X1 \acc_in_reg_reg[14]  ( .D(acc_in[14]), .CK(clk), .RN(rst_n), .Q(
        acc_in_reg[14]) );
  DFFR_X1 \acc_in_reg_reg[13]  ( .D(acc_in[13]), .CK(clk), .RN(rst_n), .Q(
        acc_in_reg[13]) );
  DFFR_X1 \acc_in_reg_reg[12]  ( .D(acc_in[12]), .CK(clk), .RN(rst_n), .Q(
        acc_in_reg[12]) );
  DFFR_X1 \acc_in_reg_reg[11]  ( .D(acc_in[11]), .CK(clk), .RN(rst_n), .Q(
        acc_in_reg[11]) );
  DFFR_X1 \acc_in_reg_reg[10]  ( .D(acc_in[10]), .CK(clk), .RN(rst_n), .Q(
        acc_in_reg[10]) );
  DFFR_X1 \acc_in_reg_reg[9]  ( .D(acc_in[9]), .CK(clk), .RN(rst_n), .Q(
        acc_in_reg[9]) );
  DFFR_X1 \acc_in_reg_reg[8]  ( .D(acc_in[8]), .CK(clk), .RN(rst_n), .Q(
        acc_in_reg[8]) );
  DFFR_X1 \acc_in_reg_reg[7]  ( .D(acc_in[7]), .CK(clk), .RN(rst_n), .Q(
        acc_in_reg[7]) );
  DFFR_X1 \acc_in_reg_reg[6]  ( .D(acc_in[6]), .CK(clk), .RN(rst_n), .Q(
        acc_in_reg[6]) );
  DFFR_X1 \acc_in_reg_reg[5]  ( .D(acc_in[5]), .CK(clk), .RN(rst_n), .Q(
        acc_in_reg[5]) );
  DFFR_X1 \acc_in_reg_reg[4]  ( .D(acc_in[4]), .CK(clk), .RN(rst_n), .Q(
        acc_in_reg[4]) );
  DFFR_X1 \acc_in_reg_reg[3]  ( .D(acc_in[3]), .CK(clk), .RN(rst_n), .Q(
        acc_in_reg[3]) );
  DFFR_X1 \acc_in_reg_reg[2]  ( .D(acc_in[2]), .CK(clk), .RN(rst_n), .Q(
        acc_in_reg[2]) );
  DFFR_X1 \acc_in_reg_reg[1]  ( .D(acc_in[1]), .CK(clk), .RN(rst_n), .Q(
        acc_in_reg[1]) );
  DFFR_X1 \acc_in_reg_reg[0]  ( .D(acc_in[0]), .CK(clk), .RN(rst_n), .Q(
        acc_in_reg[0]), .QN(n266) );
  DFFR_X1 valid_out_reg ( .D(valid_s1), .CK(clk), .RN(rst_n), .Q(valid_out) );
  DFFR_X1 \acc_out_reg[31]  ( .D(N48), .CK(clk), .RN(rst_n), .Q(acc_out[31])
         );
  DFFR_X1 \acc_out_reg[30]  ( .D(N47), .CK(clk), .RN(rst_n), .Q(acc_out[30])
         );
  DFFR_X1 \acc_out_reg[29]  ( .D(N46), .CK(clk), .RN(rst_n), .Q(acc_out[29])
         );
  DFFR_X1 \acc_out_reg[28]  ( .D(N45), .CK(clk), .RN(rst_n), .Q(acc_out[28])
         );
  DFFR_X1 \acc_out_reg[27]  ( .D(N44), .CK(clk), .RN(rst_n), .Q(acc_out[27])
         );
  DFFR_X1 \acc_out_reg[26]  ( .D(N43), .CK(clk), .RN(rst_n), .Q(acc_out[26])
         );
  DFFR_X1 \acc_out_reg[25]  ( .D(N42), .CK(clk), .RN(rst_n), .Q(acc_out[25])
         );
  DFFR_X1 \acc_out_reg[24]  ( .D(N41), .CK(clk), .RN(rst_n), .Q(acc_out[24])
         );
  DFFR_X1 \acc_out_reg[23]  ( .D(N40), .CK(clk), .RN(rst_n), .Q(acc_out[23])
         );
  DFFR_X1 \acc_out_reg[22]  ( .D(N39), .CK(clk), .RN(rst_n), .Q(acc_out[22])
         );
  DFFR_X1 \acc_out_reg[21]  ( .D(N38), .CK(clk), .RN(rst_n), .Q(acc_out[21])
         );
  DFFR_X1 \acc_out_reg[20]  ( .D(N37), .CK(clk), .RN(rst_n), .Q(acc_out[20])
         );
  DFFR_X1 \acc_out_reg[19]  ( .D(N36), .CK(clk), .RN(rst_n), .Q(acc_out[19])
         );
  DFFR_X1 \acc_out_reg[18]  ( .D(N35), .CK(clk), .RN(rst_n), .Q(acc_out[18])
         );
  DFFR_X1 \acc_out_reg[17]  ( .D(N34), .CK(clk), .RN(rst_n), .Q(acc_out[17])
         );
  DFFR_X1 \acc_out_reg[16]  ( .D(N33), .CK(clk), .RN(rst_n), .Q(acc_out[16])
         );
  DFFR_X1 \acc_out_reg[15]  ( .D(N32), .CK(clk), .RN(rst_n), .Q(acc_out[15])
         );
  DFFR_X1 \acc_out_reg[14]  ( .D(N31), .CK(clk), .RN(rst_n), .Q(acc_out[14])
         );
  DFFR_X1 \acc_out_reg[13]  ( .D(N30), .CK(clk), .RN(rst_n), .Q(acc_out[13])
         );
  DFFR_X1 \acc_out_reg[12]  ( .D(N29), .CK(clk), .RN(rst_n), .Q(acc_out[12])
         );
  DFFR_X1 \acc_out_reg[11]  ( .D(N28), .CK(clk), .RN(rst_n), .Q(acc_out[11])
         );
  DFFR_X1 \acc_out_reg[10]  ( .D(N27), .CK(clk), .RN(rst_n), .Q(acc_out[10])
         );
  DFFR_X1 \acc_out_reg[9]  ( .D(N26), .CK(clk), .RN(rst_n), .Q(acc_out[9]) );
  DFFR_X1 \acc_out_reg[8]  ( .D(N25), .CK(clk), .RN(rst_n), .Q(acc_out[8]) );
  DFFR_X1 \acc_out_reg[7]  ( .D(N24), .CK(clk), .RN(rst_n), .Q(acc_out[7]) );
  DFFR_X1 \acc_out_reg[6]  ( .D(N23), .CK(clk), .RN(rst_n), .Q(acc_out[6]) );
  DFFR_X1 \acc_out_reg[5]  ( .D(N22), .CK(clk), .RN(rst_n), .Q(acc_out[5]) );
  DFFR_X1 \acc_out_reg[4]  ( .D(N21), .CK(clk), .RN(rst_n), .Q(acc_out[4]) );
  DFFR_X1 \acc_out_reg[3]  ( .D(N20), .CK(clk), .RN(rst_n), .Q(acc_out[3]) );
  DFFR_X1 \acc_out_reg[2]  ( .D(N19), .CK(clk), .RN(rst_n), .Q(acc_out[2]) );
  DFFR_X1 \acc_out_reg[1]  ( .D(N18), .CK(clk), .RN(rst_n), .Q(acc_out[1]) );
  DFFR_X1 \acc_out_reg[0]  ( .D(N17), .CK(clk), .RN(rst_n), .Q(acc_out[0]) );
  FA_X1 \intadd_1/U13  ( .A(\intadd_1/A[0] ), .B(\intadd_1/B[0] ), .CI(
        \intadd_1/CI ), .CO(\intadd_1/n12 ), .S(N4) );
  FA_X1 \intadd_1/U12  ( .A(\intadd_1/A[1] ), .B(\intadd_1/B[1] ), .CI(
        \intadd_1/n12 ), .CO(\intadd_1/n11 ), .S(N5) );
  FA_X1 \intadd_1/U11  ( .A(\intadd_1/A[2] ), .B(\intadd_1/B[2] ), .CI(
        \intadd_1/n11 ), .CO(\intadd_1/n10 ), .S(N6) );
  FA_X1 \intadd_1/U10  ( .A(\intadd_1/A[3] ), .B(\intadd_1/B[3] ), .CI(
        \intadd_1/n10 ), .CO(\intadd_1/n9 ), .S(N7) );
  FA_X1 \intadd_1/U9  ( .A(\intadd_1/A[4] ), .B(\intadd_1/B[4] ), .CI(
        \intadd_1/n9 ), .CO(\intadd_1/n8 ), .S(N8) );
  FA_X1 \intadd_1/U8  ( .A(\intadd_1/A[5] ), .B(\intadd_1/B[5] ), .CI(
        \intadd_1/n8 ), .CO(\intadd_1/n7 ), .S(N9) );
  FA_X1 \intadd_1/U7  ( .A(\intadd_1/A[6] ), .B(\intadd_1/B[6] ), .CI(
        \intadd_1/n7 ), .CO(\intadd_1/n6 ), .S(N10) );
  FA_X1 \intadd_1/U6  ( .A(\intadd_1/A[7] ), .B(\intadd_1/B[7] ), .CI(
        \intadd_1/n6 ), .CO(\intadd_1/n5 ), .S(N11) );
  FA_X1 \intadd_1/U5  ( .A(\intadd_1/A[8] ), .B(\intadd_1/B[8] ), .CI(
        \intadd_1/n5 ), .CO(\intadd_1/n4 ), .S(N12) );
  FA_X1 \intadd_1/U4  ( .A(\intadd_1/A[9] ), .B(\intadd_1/B[9] ), .CI(
        \intadd_1/n4 ), .CO(\intadd_1/n3 ), .S(N13) );
  FA_X1 \intadd_1/U3  ( .A(\intadd_1/A[10] ), .B(\intadd_1/B[10] ), .CI(
        \intadd_1/n3 ), .CO(\intadd_1/n2 ), .S(N14) );
  FA_X1 \intadd_1/U2  ( .A(\intadd_1/A[11] ), .B(\intadd_1/B[11] ), .CI(
        \intadd_1/n2 ), .CO(\intadd_1/n1 ), .S(N15) );
  FA_X1 \intadd_0/U30  ( .A(acc_in_reg[1]), .B(mul_reg[1]), .CI(\intadd_0/CI ), 
        .CO(\intadd_0/n29 ), .S(N18) );
  FA_X1 \intadd_0/U29  ( .A(acc_in_reg[2]), .B(mul_reg[2]), .CI(\intadd_0/n29 ), .CO(\intadd_0/n28 ), .S(N19) );
  FA_X1 \intadd_0/U9  ( .A(mul_reg[15]), .B(acc_in_reg[22]), .CI(\intadd_0/n9 ), .CO(\intadd_0/n8 ), .S(N39) );
  DFFR_X2 \mul_reg_reg[15]  ( .D(N16), .CK(clk), .RN(rst_n), .Q(mul_reg[15])
         );
  AND3_X1 U4 ( .A1(n37), .A2(n36), .A3(n35), .ZN(n83) );
  AND3_X1 U5 ( .A1(n51), .A2(n50), .A3(n49), .ZN(n78) );
  OAI21_X1 U6 ( .B1(n3), .B2(n83), .A(n38), .ZN(n59) );
  AND3_X1 U7 ( .A1(n47), .A2(n46), .A3(n45), .ZN(n80) );
  AND3_X1 U8 ( .A1(n55), .A2(n54), .A3(n53), .ZN(n58) );
  AND3_X1 U9 ( .A1(n62), .A2(n61), .A3(n60), .ZN(n76) );
  AND3_X1 U10 ( .A1(n9), .A2(n73), .A3(n72), .ZN(n259) );
  OAI21_X1 U11 ( .B1(n4), .B2(n81), .A(n28), .ZN(n168) );
  NAND2_X1 U12 ( .A1(mul_reg[14]), .A2(acc_in_reg[14]), .ZN(n40) );
  NAND2_X1 U13 ( .A1(n59), .A2(n39), .ZN(n41) );
  OR2_X1 U14 ( .A1(mul_reg[14]), .A2(acc_in_reg[14]), .ZN(n39) );
  NAND2_X1 U15 ( .A1(mul_reg[15]), .A2(acc_in_reg[17]), .ZN(n48) );
  NAND2_X1 U16 ( .A1(mul_reg[15]), .A2(acc_in_reg[19]), .ZN(n52) );
  NAND2_X1 U17 ( .A1(acc_in_reg[24]), .A2(mul_reg[15]), .ZN(n63) );
  OAI21_X1 U18 ( .B1(n2), .B2(n259), .A(n258), .ZN(n263) );
  NAND2_X1 U19 ( .A1(acc_in_reg[29]), .A2(mul_reg[15]), .ZN(n258) );
  XNOR2_X1 U20 ( .A(n58), .B(n56), .ZN(N38) );
  OAI21_X1 U21 ( .B1(n7), .B2(n58), .A(n57), .ZN(\intadd_0/n9 ) );
  XNOR2_X1 U22 ( .A(n259), .B(n74), .ZN(N46) );
  NOR2_X1 U23 ( .A1(mul_reg[15]), .A2(acc_in_reg[29]), .ZN(n2) );
  AOI221_X2 U24 ( .B1(n100), .B2(n240), .C1(x[6]), .C2(x[7]), .A(n195), .ZN(
        n194) );
  AOI221_X2 U25 ( .B1(x[2]), .B2(x[3]), .C1(n85), .C2(n197), .A(n211), .ZN(
        n209) );
  NAND2_X1 U26 ( .A1(acc_in_reg[21]), .A2(mul_reg[15]), .ZN(n57) );
  NOR2_X1 U27 ( .A1(mul_reg[13]), .A2(acc_in_reg[13]), .ZN(n3) );
  NOR2_X1 U28 ( .A1(mul_reg[9]), .A2(acc_in_reg[9]), .ZN(n4) );
  NOR2_X1 U29 ( .A1(acc_in_reg[17]), .A2(mul_reg[15]), .ZN(n5) );
  NOR2_X1 U30 ( .A1(acc_in_reg[19]), .A2(mul_reg[15]), .ZN(n6) );
  NOR2_X1 U31 ( .A1(mul_reg[15]), .A2(acc_in_reg[21]), .ZN(n7) );
  NOR2_X1 U32 ( .A1(mul_reg[15]), .A2(acc_in_reg[24]), .ZN(n8) );
  AND3_X1 U33 ( .A1(n27), .A2(n26), .A3(n25), .ZN(n81) );
  AND2_X1 U34 ( .A1(n187), .A2(n188), .ZN(n9) );
  OAI21_X1 U35 ( .B1(n5), .B2(n80), .A(n48), .ZN(n180) );
  OAI21_X1 U36 ( .B1(n8), .B2(n76), .A(n63), .ZN(n174) );
  NAND2_X1 U37 ( .A1(mul_reg[9]), .A2(acc_in_reg[9]), .ZN(n28) );
  NAND2_X1 U38 ( .A1(mul_reg[13]), .A2(acc_in_reg[13]), .ZN(n38) );
  OAI21_X1 U39 ( .B1(n6), .B2(n78), .A(n52), .ZN(n191) );
  NAND2_X1 U40 ( .A1(n41), .A2(n40), .ZN(n183) );
  XNOR2_X1 U41 ( .A(n81), .B(n82), .ZN(N26) );
  XNOR2_X1 U42 ( .A(n76), .B(n75), .ZN(N41) );
  NAND2_X1 U43 ( .A1(\intadd_0/n28 ), .A2(acc_in_reg[3]), .ZN(n12) );
  NAND2_X1 U44 ( .A1(\intadd_0/n28 ), .A2(mul_reg[3]), .ZN(n11) );
  NAND2_X1 U45 ( .A1(acc_in_reg[3]), .A2(mul_reg[3]), .ZN(n10) );
  NAND3_X1 U46 ( .A1(n12), .A2(n11), .A3(n10), .ZN(n173) );
  NAND2_X1 U47 ( .A1(n173), .A2(mul_reg[4]), .ZN(n15) );
  NAND2_X1 U48 ( .A1(n173), .A2(acc_in_reg[4]), .ZN(n14) );
  NAND2_X1 U49 ( .A1(acc_in_reg[4]), .A2(mul_reg[4]), .ZN(n13) );
  NAND3_X1 U50 ( .A1(n15), .A2(n14), .A3(n13), .ZN(n169) );
  NAND2_X1 U51 ( .A1(n169), .A2(mul_reg[5]), .ZN(n18) );
  NAND2_X1 U52 ( .A1(n169), .A2(acc_in_reg[5]), .ZN(n17) );
  NAND2_X1 U53 ( .A1(acc_in_reg[5]), .A2(mul_reg[5]), .ZN(n16) );
  NAND3_X1 U54 ( .A1(n18), .A2(n17), .A3(n16), .ZN(n162) );
  NAND2_X1 U55 ( .A1(n162), .A2(mul_reg[6]), .ZN(n21) );
  NAND2_X1 U56 ( .A1(n162), .A2(acc_in_reg[6]), .ZN(n20) );
  NAND2_X1 U57 ( .A1(mul_reg[6]), .A2(acc_in_reg[6]), .ZN(n19) );
  NAND3_X1 U58 ( .A1(n21), .A2(n20), .A3(n19), .ZN(n164) );
  NAND2_X1 U59 ( .A1(n164), .A2(acc_in_reg[7]), .ZN(n24) );
  NAND2_X1 U60 ( .A1(n164), .A2(mul_reg[7]), .ZN(n23) );
  NAND2_X1 U61 ( .A1(acc_in_reg[7]), .A2(mul_reg[7]), .ZN(n22) );
  NAND3_X1 U62 ( .A1(n24), .A2(n23), .A3(n22), .ZN(n156) );
  NAND2_X1 U63 ( .A1(n156), .A2(mul_reg[8]), .ZN(n27) );
  NAND2_X1 U64 ( .A1(n156), .A2(acc_in_reg[8]), .ZN(n26) );
  NAND2_X1 U65 ( .A1(mul_reg[8]), .A2(acc_in_reg[8]), .ZN(n25) );
  NAND2_X1 U66 ( .A1(n168), .A2(mul_reg[10]), .ZN(n31) );
  NAND2_X1 U67 ( .A1(n168), .A2(acc_in_reg[10]), .ZN(n30) );
  NAND2_X1 U68 ( .A1(mul_reg[10]), .A2(acc_in_reg[10]), .ZN(n29) );
  NAND3_X1 U69 ( .A1(n31), .A2(n30), .A3(n29), .ZN(n158) );
  NAND2_X1 U70 ( .A1(n158), .A2(acc_in_reg[11]), .ZN(n34) );
  NAND2_X1 U71 ( .A1(n158), .A2(mul_reg[11]), .ZN(n33) );
  NAND2_X1 U72 ( .A1(acc_in_reg[11]), .A2(mul_reg[11]), .ZN(n32) );
  NAND3_X1 U73 ( .A1(n34), .A2(n33), .A3(n32), .ZN(n161) );
  NAND2_X1 U74 ( .A1(n161), .A2(mul_reg[12]), .ZN(n37) );
  NAND2_X1 U75 ( .A1(n161), .A2(acc_in_reg[12]), .ZN(n36) );
  NAND2_X1 U76 ( .A1(acc_in_reg[12]), .A2(mul_reg[12]), .ZN(n35) );
  NAND2_X1 U77 ( .A1(n183), .A2(mul_reg[15]), .ZN(n44) );
  NAND2_X1 U78 ( .A1(n183), .A2(acc_in_reg[15]), .ZN(n43) );
  NAND2_X1 U79 ( .A1(mul_reg[15]), .A2(acc_in_reg[15]), .ZN(n42) );
  NAND3_X1 U80 ( .A1(n44), .A2(n43), .A3(n42), .ZN(n176) );
  NAND2_X1 U81 ( .A1(n176), .A2(mul_reg[15]), .ZN(n47) );
  NAND2_X1 U82 ( .A1(n176), .A2(acc_in_reg[16]), .ZN(n46) );
  NAND2_X1 U83 ( .A1(mul_reg[15]), .A2(acc_in_reg[16]), .ZN(n45) );
  NAND2_X1 U84 ( .A1(n180), .A2(mul_reg[15]), .ZN(n51) );
  NAND2_X1 U85 ( .A1(n180), .A2(acc_in_reg[18]), .ZN(n50) );
  NAND2_X1 U86 ( .A1(mul_reg[15]), .A2(acc_in_reg[18]), .ZN(n49) );
  NAND2_X1 U87 ( .A1(n191), .A2(acc_in_reg[20]), .ZN(n55) );
  NAND2_X1 U88 ( .A1(n191), .A2(mul_reg[15]), .ZN(n54) );
  NAND2_X1 U89 ( .A1(mul_reg[15]), .A2(acc_in_reg[20]), .ZN(n53) );
  XOR2_X1 U90 ( .A(mul_reg[15]), .B(acc_in_reg[21]), .Z(n56) );
  FA_X1 U91 ( .A(mul_reg[14]), .B(n59), .CI(acc_in_reg[14]), .S(N31) );
  NAND2_X1 U92 ( .A1(mul_reg[15]), .A2(acc_in_reg[27]), .ZN(n187) );
  NAND2_X1 U93 ( .A1(\intadd_0/n8 ), .A2(mul_reg[15]), .ZN(n62) );
  NAND2_X1 U94 ( .A1(\intadd_0/n8 ), .A2(acc_in_reg[23]), .ZN(n61) );
  NAND2_X1 U95 ( .A1(mul_reg[15]), .A2(acc_in_reg[23]), .ZN(n60) );
  NAND2_X1 U96 ( .A1(n174), .A2(mul_reg[15]), .ZN(n65) );
  NAND2_X1 U97 ( .A1(n174), .A2(acc_in_reg[25]), .ZN(n66) );
  NAND2_X1 U98 ( .A1(mul_reg[15]), .A2(acc_in_reg[25]), .ZN(n64) );
  NAND3_X1 U99 ( .A1(n65), .A2(n66), .A3(n64), .ZN(n178) );
  NAND2_X1 U100 ( .A1(n178), .A2(mul_reg[15]), .ZN(n70) );
  INV_X1 U101 ( .A(n66), .ZN(n67) );
  NAND2_X1 U102 ( .A1(n67), .A2(acc_in_reg[26]), .ZN(n69) );
  NAND2_X1 U103 ( .A1(mul_reg[15]), .A2(acc_in_reg[26]), .ZN(n68) );
  NAND3_X1 U104 ( .A1(n70), .A2(n69), .A3(n68), .ZN(n185) );
  NAND2_X1 U105 ( .A1(n185), .A2(mul_reg[15]), .ZN(n188) );
  NAND2_X1 U106 ( .A1(n185), .A2(acc_in_reg[27]), .ZN(n186) );
  INV_X1 U107 ( .A(n186), .ZN(n71) );
  NAND2_X1 U108 ( .A1(n71), .A2(acc_in_reg[28]), .ZN(n73) );
  NAND2_X1 U109 ( .A1(mul_reg[15]), .A2(acc_in_reg[28]), .ZN(n72) );
  XOR2_X1 U110 ( .A(mul_reg[15]), .B(acc_in_reg[29]), .Z(n74) );
  XOR2_X1 U111 ( .A(mul_reg[15]), .B(acc_in_reg[24]), .Z(n75) );
  XOR2_X1 U112 ( .A(mul_reg[15]), .B(acc_in_reg[19]), .Z(n77) );
  XNOR2_X1 U113 ( .A(n78), .B(n77), .ZN(N36) );
  XOR2_X1 U114 ( .A(mul_reg[15]), .B(acc_in_reg[17]), .Z(n79) );
  XNOR2_X1 U115 ( .A(n80), .B(n79), .ZN(N34) );
  XOR2_X1 U116 ( .A(mul_reg[9]), .B(acc_in_reg[9]), .Z(n82) );
  XOR2_X1 U117 ( .A(mul_reg[13]), .B(acc_in_reg[13]), .Z(n84) );
  XNOR2_X1 U118 ( .A(n83), .B(n84), .ZN(N30) );
  AND2_X1 U119 ( .A1(x[0]), .A2(w[0]), .ZN(N1) );
  INV_X1 U120 ( .A(\intadd_1/n1 ), .ZN(N16) );
  INV_X1 U121 ( .A(w[0]), .ZN(n196) );
  INV_X1 U122 ( .A(x[2]), .ZN(n85) );
  INV_X1 U123 ( .A(x[3]), .ZN(n197) );
  XOR2_X1 U124 ( .A(x[1]), .B(x[2]), .Z(n211) );
  OAI221_X1 U125 ( .B1(n196), .B2(n209), .C1(n211), .C2(n209), .A(x[3]), .ZN(
        n86) );
  INV_X1 U126 ( .A(n86), .ZN(\intadd_1/A[0] ) );
  INV_X1 U127 ( .A(x[4]), .ZN(n93) );
  OAI22_X1 U128 ( .A1(n197), .A2(n93), .B1(x[4]), .B2(x[3]), .ZN(n144) );
  INV_X1 U129 ( .A(n144), .ZN(n252) );
  NAND2_X1 U130 ( .A1(w[0]), .A2(n252), .ZN(n91) );
  INV_X1 U131 ( .A(x[1]), .ZN(n154) );
  NAND2_X1 U132 ( .A1(n154), .A2(x[0]), .ZN(n108) );
  INV_X1 U133 ( .A(n108), .ZN(n215) );
  NOR2_X1 U134 ( .A1(n154), .A2(x[0]), .ZN(n214) );
  INV_X1 U135 ( .A(n214), .ZN(n203) );
  NAND2_X1 U136 ( .A1(x[0]), .A2(x[1]), .ZN(n217) );
  OAI22_X1 U137 ( .A1(w[3]), .A2(n203), .B1(w[4]), .B2(n217), .ZN(n87) );
  AOI21_X1 U138 ( .B1(n215), .B2(w[4]), .A(n87), .ZN(n90) );
  INV_X1 U139 ( .A(w[2]), .ZN(n115) );
  OAI22_X1 U140 ( .A1(n115), .A2(x[3]), .B1(n197), .B2(w[2]), .ZN(n208) );
  INV_X1 U141 ( .A(w[1]), .ZN(n99) );
  AOI22_X1 U142 ( .A1(w[1]), .A2(x[3]), .B1(n197), .B2(n99), .ZN(n199) );
  AOI22_X1 U143 ( .A1(n211), .A2(n208), .B1(n209), .B2(n199), .ZN(n89) );
  INV_X1 U144 ( .A(n88), .ZN(\intadd_1/B[1] ) );
  FA_X1 U145 ( .A(n91), .B(n90), .CI(n89), .CO(n92), .S(n88) );
  INV_X1 U146 ( .A(n92), .ZN(\intadd_1/A[2] ) );
  INV_X1 U147 ( .A(x[5]), .ZN(n232) );
  AOI22_X1 U148 ( .A1(w[2]), .A2(x[5]), .B1(n232), .B2(n115), .ZN(n105) );
  OAI221_X1 U149 ( .B1(n93), .B2(n232), .C1(x[4]), .C2(x[5]), .A(n144), .ZN(
        n235) );
  INV_X1 U150 ( .A(n235), .ZN(n251) );
  OAI22_X1 U151 ( .A1(n99), .A2(x[5]), .B1(n232), .B2(w[1]), .ZN(n95) );
  AOI22_X1 U152 ( .A1(n252), .A2(n105), .B1(n251), .B2(n95), .ZN(n113) );
  INV_X1 U153 ( .A(x[6]), .ZN(n100) );
  OAI22_X1 U154 ( .A1(n232), .A2(n100), .B1(x[6]), .B2(x[5]), .ZN(n249) );
  INV_X1 U155 ( .A(n249), .ZN(n195) );
  NAND2_X1 U156 ( .A1(w[0]), .A2(n195), .ZN(n104) );
  OAI22_X1 U157 ( .A1(w[5]), .A2(n203), .B1(w[6]), .B2(n217), .ZN(n94) );
  AOI21_X1 U158 ( .B1(n215), .B2(w[6]), .A(n94), .ZN(n103) );
  INV_X1 U159 ( .A(w[4]), .ZN(n213) );
  AOI22_X1 U160 ( .A1(x[3]), .A2(w[4]), .B1(n213), .B2(n197), .ZN(n107) );
  INV_X1 U161 ( .A(w[3]), .ZN(n134) );
  OAI22_X1 U162 ( .A1(n197), .A2(w[3]), .B1(n134), .B2(x[3]), .ZN(n210) );
  AOI22_X1 U163 ( .A1(n211), .A2(n107), .B1(n209), .B2(n210), .ZN(n102) );
  AOI221_X1 U164 ( .B1(w[0]), .B2(n252), .C1(x[4]), .C2(n144), .A(n232), .ZN(
        n207) );
  INV_X1 U165 ( .A(n95), .ZN(n97) );
  OAI221_X1 U166 ( .B1(w[0]), .B2(x[5]), .C1(n196), .C2(n232), .A(n251), .ZN(
        n96) );
  OAI21_X1 U167 ( .B1(n144), .B2(n97), .A(n96), .ZN(n206) );
  NAND2_X1 U168 ( .A1(n207), .A2(n206), .ZN(n111) );
  INV_X1 U169 ( .A(n98), .ZN(\intadd_1/A[3] ) );
  INV_X1 U170 ( .A(x[7]), .ZN(n240) );
  AOI22_X1 U171 ( .A1(w[1]), .A2(x[7]), .B1(n240), .B2(n99), .ZN(n116) );
  INV_X1 U172 ( .A(n194), .ZN(n247) );
  AOI221_X1 U173 ( .B1(w[0]), .B2(x[7]), .C1(n196), .C2(n240), .A(n247), .ZN(
        n101) );
  AOI21_X1 U174 ( .B1(n195), .B2(n116), .A(n101), .ZN(n123) );
  OAI221_X1 U175 ( .B1(n194), .B2(n195), .C1(n194), .C2(n196), .A(x[7]), .ZN(
        n122) );
  XNOR2_X1 U176 ( .A(n123), .B(n122), .ZN(n128) );
  FA_X1 U177 ( .A(n104), .B(n103), .CI(n102), .CO(n127), .S(n112) );
  OAI22_X1 U178 ( .A1(n232), .A2(n134), .B1(w[3]), .B2(x[5]), .ZN(n121) );
  INV_X1 U179 ( .A(n121), .ZN(n106) );
  AOI22_X1 U180 ( .A1(n252), .A2(n106), .B1(n251), .B2(n105), .ZN(n120) );
  INV_X1 U181 ( .A(w[5]), .ZN(n241) );
  AOI22_X1 U182 ( .A1(x[3]), .A2(w[5]), .B1(n241), .B2(n197), .ZN(n117) );
  AOI22_X1 U183 ( .A1(n211), .A2(n117), .B1(n209), .B2(n107), .ZN(n119) );
  INV_X1 U184 ( .A(w[6]), .ZN(n151) );
  INV_X1 U185 ( .A(w[7]), .ZN(n233) );
  AOI22_X1 U186 ( .A1(w[7]), .A2(n108), .B1(n217), .B2(n233), .ZN(n109) );
  AOI21_X1 U187 ( .B1(n214), .B2(n151), .A(n109), .ZN(n118) );
  INV_X1 U188 ( .A(n110), .ZN(\intadd_1/B[4] ) );
  FA_X1 U189 ( .A(n113), .B(n112), .CI(n111), .CO(n114), .S(n98) );
  INV_X1 U190 ( .A(n114), .ZN(\intadd_1/A[4] ) );
  AOI22_X1 U191 ( .A1(w[2]), .A2(x[7]), .B1(n240), .B2(n115), .ZN(n138) );
  AOI22_X1 U192 ( .A1(n195), .A2(n138), .B1(n194), .B2(n116), .ZN(n222) );
  AOI22_X1 U193 ( .A1(x[3]), .A2(w[6]), .B1(n151), .B2(n197), .ZN(n136) );
  AOI22_X1 U194 ( .A1(n211), .A2(n136), .B1(n209), .B2(n117), .ZN(n221) );
  XOR2_X1 U195 ( .A(n222), .B(n221), .Z(n132) );
  FA_X1 U196 ( .A(n120), .B(n119), .CI(n118), .CO(n131), .S(n126) );
  AOI22_X1 U197 ( .A1(x[5]), .A2(n213), .B1(w[4]), .B2(n232), .ZN(n140) );
  OAI22_X1 U198 ( .A1(n144), .A2(n140), .B1(n235), .B2(n121), .ZN(n225) );
  NOR2_X1 U199 ( .A1(n123), .A2(n122), .ZN(n224) );
  AOI22_X1 U200 ( .A1(w[7]), .A2(n215), .B1(x[1]), .B2(n233), .ZN(n223) );
  INV_X1 U201 ( .A(n124), .ZN(n130) );
  INV_X1 U202 ( .A(n125), .ZN(\intadd_1/B[5] ) );
  FA_X1 U203 ( .A(n128), .B(n127), .CI(n126), .CO(n129), .S(n110) );
  INV_X1 U204 ( .A(n129), .ZN(\intadd_1/A[5] ) );
  FA_X1 U205 ( .A(n132), .B(n131), .CI(n130), .CO(n133), .S(n125) );
  INV_X1 U206 ( .A(n133), .ZN(\intadd_1/A[6] ) );
  OAI22_X1 U207 ( .A1(n240), .A2(n213), .B1(w[4]), .B2(x[7]), .ZN(n242) );
  INV_X1 U208 ( .A(n242), .ZN(n135) );
  AOI22_X1 U209 ( .A1(w[3]), .A2(x[7]), .B1(n240), .B2(n134), .ZN(n139) );
  AOI22_X1 U210 ( .A1(n195), .A2(n135), .B1(n194), .B2(n139), .ZN(n149) );
  AOI22_X1 U211 ( .A1(x[3]), .A2(w[7]), .B1(n233), .B2(n197), .ZN(n142) );
  AOI22_X1 U212 ( .A1(n211), .A2(n142), .B1(n209), .B2(n136), .ZN(n137) );
  INV_X1 U213 ( .A(n137), .ZN(n239) );
  AOI22_X1 U214 ( .A1(n195), .A2(n139), .B1(n194), .B2(n138), .ZN(n227) );
  OAI22_X1 U215 ( .A1(n232), .A2(n241), .B1(w[5]), .B2(x[5]), .ZN(n143) );
  OAI22_X1 U216 ( .A1(n144), .A2(n143), .B1(n235), .B2(n140), .ZN(n141) );
  INV_X1 U217 ( .A(n141), .ZN(n226) );
  OAI21_X1 U218 ( .B1(n211), .B2(n209), .A(n142), .ZN(n238) );
  AOI22_X1 U219 ( .A1(x[5]), .A2(n151), .B1(w[6]), .B2(n232), .ZN(n236) );
  OAI22_X1 U220 ( .A1(n144), .A2(n236), .B1(n235), .B2(n143), .ZN(n237) );
  INV_X1 U221 ( .A(n145), .ZN(n147) );
  INV_X1 U222 ( .A(n146), .ZN(\intadd_1/B[7] ) );
  FA_X1 U223 ( .A(n149), .B(n148), .CI(n147), .CO(n150), .S(n146) );
  INV_X1 U224 ( .A(n150), .ZN(\intadd_1/A[8] ) );
  AOI22_X1 U225 ( .A1(x[7]), .A2(w[7]), .B1(n233), .B2(n240), .ZN(n193) );
  INV_X1 U226 ( .A(n193), .ZN(n152) );
  AOI22_X1 U227 ( .A1(x[7]), .A2(n151), .B1(w[6]), .B2(n240), .ZN(n248) );
  OAI22_X1 U228 ( .A1(n249), .A2(n152), .B1(n247), .B2(n248), .ZN(
        \intadd_1/B[11] ) );
  INV_X1 U229 ( .A(\intadd_1/B[11] ), .ZN(\intadd_1/A[10] ) );
  NOR2_X1 U230 ( .A1(w[1]), .A2(n217), .ZN(n153) );
  AOI21_X1 U231 ( .B1(n215), .B2(w[1]), .A(n153), .ZN(n155) );
  AOI211_X1 U232 ( .C1(x[0]), .C2(w[1]), .A(w[0]), .B(n154), .ZN(n205) );
  AOI221_X1 U233 ( .B1(N1), .B2(n155), .C1(n154), .C2(n155), .A(n205), .ZN(N2)
         );
  NOR2_X1 U234 ( .A1(n265), .A2(n266), .ZN(\intadd_0/CI ) );
  XOR2_X1 U235 ( .A(mul_reg[8]), .B(acc_in_reg[8]), .Z(n157) );
  XOR2_X1 U236 ( .A(n156), .B(n157), .Z(N25) );
  XOR2_X1 U237 ( .A(acc_in_reg[11]), .B(mul_reg[11]), .Z(n159) );
  XOR2_X1 U238 ( .A(n158), .B(n159), .Z(N28) );
  XOR2_X1 U239 ( .A(acc_in_reg[12]), .B(mul_reg[12]), .Z(n160) );
  XOR2_X1 U240 ( .A(n161), .B(n160), .Z(N29) );
  XOR2_X1 U241 ( .A(mul_reg[6]), .B(acc_in_reg[6]), .Z(n163) );
  XOR2_X1 U242 ( .A(n162), .B(n163), .Z(N23) );
  XOR2_X1 U243 ( .A(acc_in_reg[7]), .B(mul_reg[7]), .Z(n165) );
  XOR2_X1 U244 ( .A(n164), .B(n165), .Z(N24) );
  XOR2_X1 U245 ( .A(mul_reg[15]), .B(acc_in_reg[23]), .Z(n166) );
  XOR2_X1 U246 ( .A(\intadd_0/n8 ), .B(n166), .Z(N40) );
  XOR2_X1 U247 ( .A(mul_reg[10]), .B(acc_in_reg[10]), .Z(n167) );
  XOR2_X1 U248 ( .A(n168), .B(n167), .Z(N27) );
  XOR2_X1 U249 ( .A(acc_in_reg[5]), .B(mul_reg[5]), .Z(n170) );
  XOR2_X1 U250 ( .A(n169), .B(n170), .Z(N22) );
  XOR2_X1 U251 ( .A(acc_in_reg[3]), .B(mul_reg[3]), .Z(n171) );
  XOR2_X1 U252 ( .A(\intadd_0/n28 ), .B(n171), .Z(N20) );
  XOR2_X1 U253 ( .A(acc_in_reg[4]), .B(mul_reg[4]), .Z(n172) );
  XOR2_X1 U254 ( .A(n173), .B(n172), .Z(N21) );
  XOR2_X1 U255 ( .A(mul_reg[15]), .B(acc_in_reg[25]), .Z(n175) );
  XOR2_X1 U256 ( .A(n174), .B(n175), .Z(N42) );
  XOR2_X1 U257 ( .A(mul_reg[15]), .B(acc_in_reg[16]), .Z(n177) );
  XOR2_X1 U258 ( .A(n176), .B(n177), .Z(N33) );
  XOR2_X1 U259 ( .A(mul_reg[15]), .B(acc_in_reg[26]), .Z(n179) );
  XOR2_X1 U260 ( .A(n178), .B(n179), .Z(N43) );
  XOR2_X1 U261 ( .A(mul_reg[15]), .B(acc_in_reg[18]), .Z(n181) );
  XOR2_X1 U262 ( .A(n180), .B(n181), .Z(N35) );
  XOR2_X1 U263 ( .A(mul_reg[15]), .B(acc_in_reg[15]), .Z(n182) );
  XOR2_X1 U264 ( .A(n183), .B(n182), .Z(N32) );
  XOR2_X1 U265 ( .A(mul_reg[15]), .B(acc_in_reg[27]), .Z(n184) );
  XOR2_X1 U266 ( .A(n185), .B(n184), .Z(N44) );
  NAND3_X1 U267 ( .A1(n188), .A2(n186), .A3(n187), .ZN(n190) );
  XOR2_X1 U268 ( .A(mul_reg[15]), .B(acc_in_reg[28]), .Z(n189) );
  XOR2_X1 U269 ( .A(n190), .B(n189), .Z(N45) );
  XOR2_X1 U270 ( .A(mul_reg[15]), .B(acc_in_reg[20]), .Z(n192) );
  XOR2_X1 U271 ( .A(n191), .B(n192), .Z(N37) );
  OAI21_X1 U272 ( .B1(n195), .B2(n194), .A(n193), .ZN(\intadd_1/A[11] ) );
  AOI22_X1 U273 ( .A1(w[0]), .A2(x[3]), .B1(n197), .B2(n196), .ZN(n198) );
  AOI22_X1 U274 ( .A1(n211), .A2(n199), .B1(n209), .B2(n198), .ZN(n201) );
  OAI22_X1 U275 ( .A1(w[2]), .A2(n203), .B1(w[3]), .B2(n217), .ZN(n200) );
  AOI21_X1 U276 ( .B1(n215), .B2(w[3]), .A(n200), .ZN(n202) );
  NOR2_X1 U277 ( .A1(n201), .A2(n202), .ZN(\intadd_1/A[1] ) );
  AOI21_X1 U278 ( .B1(n202), .B2(n201), .A(\intadd_1/A[1] ), .ZN(
        \intadd_1/B[0] ) );
  OAI22_X1 U279 ( .A1(w[1]), .A2(n203), .B1(w[2]), .B2(n217), .ZN(n204) );
  AOI21_X1 U280 ( .B1(n215), .B2(w[2]), .A(n204), .ZN(n256) );
  AOI21_X1 U281 ( .B1(w[0]), .B2(n211), .A(n205), .ZN(n257) );
  NOR2_X1 U282 ( .A1(n256), .A2(n257), .ZN(\intadd_1/CI ) );
  XOR2_X1 U283 ( .A(n207), .B(n206), .Z(n220) );
  AOI22_X1 U284 ( .A1(n211), .A2(n210), .B1(n209), .B2(n208), .ZN(n212) );
  INV_X1 U285 ( .A(n212), .ZN(n219) );
  AOI22_X1 U286 ( .A1(n215), .A2(w[5]), .B1(n214), .B2(n213), .ZN(n216) );
  OAI21_X1 U287 ( .B1(w[5]), .B2(n217), .A(n216), .ZN(n218) );
  FA_X1 U288 ( .A(n220), .B(n219), .CI(n218), .CO(\intadd_1/B[3] ), .S(
        \intadd_1/B[2] ) );
  NAND2_X1 U289 ( .A1(n222), .A2(n221), .ZN(n231) );
  FA_X1 U290 ( .A(n225), .B(n224), .CI(n223), .CO(n230), .S(n124) );
  FA_X1 U291 ( .A(n239), .B(n227), .CI(n226), .CO(n148), .S(n228) );
  INV_X1 U292 ( .A(n228), .ZN(n229) );
  FA_X1 U293 ( .A(n231), .B(n230), .CI(n229), .CO(\intadd_1/A[7] ), .S(
        \intadd_1/B[6] ) );
  AOI22_X1 U294 ( .A1(x[5]), .A2(w[7]), .B1(n233), .B2(n232), .ZN(n250) );
  NAND2_X1 U295 ( .A1(n252), .A2(n250), .ZN(n234) );
  OAI21_X1 U296 ( .B1(n236), .B2(n235), .A(n234), .ZN(n253) );
  INV_X1 U297 ( .A(n253), .ZN(n245) );
  FA_X1 U298 ( .A(n239), .B(n238), .CI(n237), .CO(n244), .S(n145) );
  AOI22_X1 U299 ( .A1(x[7]), .A2(n241), .B1(w[5]), .B2(n240), .ZN(n246) );
  OAI22_X1 U300 ( .A1(n249), .A2(n246), .B1(n247), .B2(n242), .ZN(n243) );
  FA_X1 U301 ( .A(n245), .B(n244), .CI(n243), .CO(\intadd_1/A[9] ), .S(
        \intadd_1/B[8] ) );
  OAI22_X1 U302 ( .A1(n249), .A2(n248), .B1(n247), .B2(n246), .ZN(n255) );
  OAI21_X1 U303 ( .B1(n252), .B2(n251), .A(n250), .ZN(n254) );
  FA_X1 U304 ( .A(n255), .B(n254), .CI(n253), .CO(\intadd_1/B[10] ), .S(
        \intadd_1/B[9] ) );
  AOI21_X1 U305 ( .B1(n257), .B2(n256), .A(\intadd_1/CI ), .ZN(N3) );
  NOR2_X1 U306 ( .A1(mul_reg[15]), .A2(acc_in_reg[30]), .ZN(n261) );
  AOI21_X1 U307 ( .B1(acc_in_reg[30]), .B2(mul_reg[15]), .A(n261), .ZN(n260)
         );
  XOR2_X1 U308 ( .A(n260), .B(n263), .Z(N47) );
  XOR2_X1 U309 ( .A(mul_reg[0]), .B(acc_in_reg[0]), .Z(N17) );
  INV_X1 U310 ( .A(n263), .ZN(n262) );
  AOI221_X1 U311 ( .B1(mul_reg[15]), .B2(n263), .C1(acc_in_reg[30]), .C2(n262), 
        .A(n261), .ZN(n264) );
  XOR2_X1 U312 ( .A(n264), .B(acc_in_reg[31]), .Z(N48) );
endmodule


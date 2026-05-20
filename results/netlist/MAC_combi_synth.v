/////////////////////////////////////////////////////////////
// Created by: Synopsys DC Ultra(TM) in wire load mode
// Version   : X-2025.06-SP4
// Date      : Tue May 19 21:11:59 2026
/////////////////////////////////////////////////////////////


module MAC_combi ( x, weight, acc_in, acc_out );
  input [7:0] x;
  input [7:0] weight;
  input [31:0] acc_in;
  output [31:0] acc_out;
  wire   n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, n14, n15, n16,
         n17, n18, n19, n20, n21, n22, n23, n24, n25, n26, n27, n28, n29, n30,
         n31, n32, n33, n34, n35, n36, n37, n38, n39, n40, n41, n42, n43, n44,
         n45, n46, n47, n48, n49, n50, n51, n52, n53, n54, n55, n56, n57, n58,
         n59, n60, n61, n62, n63, n64, n65, n66, n67, n68, n69, n70, n71, n72,
         n73, n74, n75, n76, n77, n78, n79, n80, n81, n82, n83, n84, n85, n86,
         n87, n88, n89, n90, n91, n92, n93, n94, n95, n96, n97, n98, n99, n100,
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
         n255, n256;

  AOI221_X2 U1 ( .B1(x[6]), .B2(x[7]), .C1(n90), .C2(n99), .A(n89), .ZN(n98)
         );
  XNOR2_X1 U2 ( .A(x[2]), .B(x[1]), .ZN(n143) );
  INV_X1 U3 ( .A(n143), .ZN(n31) );
  INV_X1 U4 ( .A(weight[2]), .ZN(n91) );
  INV_X1 U5 ( .A(x[3]), .ZN(n94) );
  AOI22_X1 U6 ( .A1(x[3]), .A2(weight[2]), .B1(n91), .B2(n94), .ZN(n29) );
  INV_X1 U7 ( .A(x[2]), .ZN(n4) );
  OAI221_X1 U8 ( .B1(x[2]), .B2(x[3]), .C1(n4), .C2(n94), .A(n143), .ZN(n141)
         );
  INV_X1 U9 ( .A(n141), .ZN(n30) );
  INV_X1 U10 ( .A(weight[1]), .ZN(n77) );
  AOI22_X1 U11 ( .A1(x[3]), .A2(weight[1]), .B1(n77), .B2(n94), .ZN(n6) );
  AOI22_X1 U12 ( .A1(n31), .A2(n29), .B1(n30), .B2(n6), .ZN(n19) );
  NAND2_X1 U13 ( .A1(x[1]), .A2(x[0]), .ZN(n124) );
  INV_X1 U14 ( .A(x[1]), .ZN(n79) );
  NAND2_X1 U15 ( .A1(n79), .A2(x[0]), .ZN(n114) );
  INV_X1 U16 ( .A(n114), .ZN(n122) );
  NOR2_X1 U17 ( .A1(x[0]), .A2(n79), .ZN(n121) );
  AOI22_X1 U18 ( .A1(weight[3]), .A2(n122), .B1(n121), .B2(n91), .ZN(n1) );
  OAI21_X1 U19 ( .B1(weight[3]), .B2(n124), .A(n1), .ZN(n7) );
  NAND2_X1 U20 ( .A1(acc_in[3]), .A2(n7), .ZN(n18) );
  INV_X1 U21 ( .A(x[4]), .ZN(n26) );
  AOI22_X1 U22 ( .A1(x[3]), .A2(x[4]), .B1(n26), .B2(n94), .ZN(n137) );
  NAND2_X1 U23 ( .A1(n137), .A2(weight[0]), .ZN(n25) );
  INV_X1 U24 ( .A(n25), .ZN(n24) );
  INV_X1 U25 ( .A(weight[3]), .ZN(n75) );
  AOI22_X1 U26 ( .A1(weight[4]), .A2(n122), .B1(n121), .B2(n75), .ZN(n2) );
  OAI21_X1 U27 ( .B1(weight[4]), .B2(n124), .A(n2), .ZN(n23) );
  INV_X1 U28 ( .A(n3), .ZN(n17) );
  INV_X1 U29 ( .A(weight[0]), .ZN(n100) );
  OAI221_X1 U30 ( .B1(n31), .B2(n4), .C1(n143), .C2(n100), .A(x[3]), .ZN(n16)
         );
  AOI221_X1 U31 ( .B1(x[3]), .B2(weight[0]), .C1(n94), .C2(n100), .A(n141), 
        .ZN(n5) );
  AOI21_X1 U32 ( .B1(n31), .B2(n6), .A(n5), .ZN(n15) );
  OAI21_X1 U33 ( .B1(acc_in[3]), .B2(n7), .A(n18), .ZN(n14) );
  INV_X1 U34 ( .A(acc_in[1]), .ZN(n45) );
  INV_X1 U35 ( .A(n121), .ZN(n27) );
  OAI22_X1 U36 ( .A1(weight[1]), .A2(n124), .B1(weight[0]), .B2(n27), .ZN(n8)
         );
  AOI21_X1 U37 ( .B1(weight[1]), .B2(n122), .A(n8), .ZN(n44) );
  NAND2_X1 U38 ( .A1(weight[0]), .A2(x[0]), .ZN(n251) );
  INV_X1 U39 ( .A(n251), .ZN(n9) );
  AOI22_X1 U40 ( .A1(n9), .A2(acc_in[0]), .B1(x[1]), .B2(n251), .ZN(n43) );
  INV_X1 U41 ( .A(acc_in[2]), .ZN(n13) );
  OAI22_X1 U42 ( .A1(weight[1]), .A2(n27), .B1(weight[2]), .B2(n124), .ZN(n10)
         );
  AOI21_X1 U43 ( .B1(n122), .B2(weight[2]), .A(n10), .ZN(n12) );
  NAND2_X1 U44 ( .A1(weight[0]), .A2(n31), .ZN(n11) );
  NOR2_X1 U45 ( .A1(n249), .A2(n250), .ZN(n248) );
  INV_X1 U46 ( .A(n248), .ZN(n41) );
  FA_X1 U47 ( .A(n13), .B(n12), .CI(n11), .CO(n40), .S(n250) );
  FA_X1 U48 ( .A(n16), .B(n15), .CI(n14), .CO(n37), .S(n39) );
  FA_X1 U49 ( .A(n19), .B(n18), .CI(n17), .CO(n171), .S(n35) );
  INV_X1 U50 ( .A(x[5]), .ZN(n92) );
  AOI22_X1 U51 ( .A1(x[5]), .A2(weight[1]), .B1(n77), .B2(n92), .ZN(n135) );
  INV_X1 U52 ( .A(n137), .ZN(n64) );
  OAI221_X1 U53 ( .B1(x[4]), .B2(x[5]), .C1(n26), .C2(n92), .A(n64), .ZN(n65)
         );
  AOI221_X1 U54 ( .B1(x[5]), .B2(weight[0]), .C1(n92), .C2(n100), .A(n65), 
        .ZN(n20) );
  AOI21_X1 U55 ( .B1(n135), .B2(n137), .A(n20), .ZN(n22) );
  INV_X1 U56 ( .A(acc_in[5]), .ZN(n21) );
  NOR2_X1 U57 ( .A1(n22), .A2(n21), .ZN(n158) );
  AOI21_X1 U58 ( .B1(n22), .B2(n21), .A(n158), .ZN(n168) );
  FA_X1 U59 ( .A(acc_in[4]), .B(n24), .CI(n23), .CO(n167), .S(n3) );
  OAI211_X1 U60 ( .C1(n26), .C2(n94), .A(x[5]), .B(n25), .ZN(n153) );
  OAI22_X1 U61 ( .A1(weight[5]), .A2(n124), .B1(weight[4]), .B2(n27), .ZN(n28)
         );
  AOI21_X1 U62 ( .B1(weight[5]), .B2(n122), .A(n28), .ZN(n152) );
  AOI22_X1 U63 ( .A1(x[3]), .A2(weight[3]), .B1(n75), .B2(n94), .ZN(n139) );
  AOI22_X1 U64 ( .A1(n31), .A2(n139), .B1(n30), .B2(n29), .ZN(n151) );
  INV_X1 U65 ( .A(n32), .ZN(n166) );
  INV_X1 U66 ( .A(n33), .ZN(n169) );
  INV_X1 U67 ( .A(n34), .ZN(acc_out[5]) );
  FA_X1 U68 ( .A(n37), .B(n36), .CI(n35), .CO(n170), .S(n38) );
  INV_X1 U69 ( .A(n38), .ZN(acc_out[4]) );
  FA_X1 U70 ( .A(n41), .B(n40), .CI(n39), .CO(n36), .S(n42) );
  INV_X1 U71 ( .A(n42), .ZN(acc_out[3]) );
  FA_X1 U72 ( .A(n45), .B(n44), .CI(n43), .CO(n249), .S(n46) );
  INV_X1 U73 ( .A(n46), .ZN(acc_out[1]) );
  INV_X1 U74 ( .A(acc_in[29]), .ZN(n254) );
  INV_X1 U75 ( .A(acc_in[28]), .ZN(n193) );
  INV_X1 U76 ( .A(acc_in[27]), .ZN(n195) );
  INV_X1 U77 ( .A(acc_in[26]), .ZN(n197) );
  INV_X1 U78 ( .A(acc_in[25]), .ZN(n199) );
  INV_X1 U79 ( .A(acc_in[24]), .ZN(n201) );
  INV_X1 U80 ( .A(acc_in[23]), .ZN(n203) );
  INV_X1 U81 ( .A(acc_in[22]), .ZN(n205) );
  INV_X1 U82 ( .A(acc_in[21]), .ZN(n207) );
  INV_X1 U83 ( .A(acc_in[20]), .ZN(n209) );
  INV_X1 U84 ( .A(acc_in[19]), .ZN(n211) );
  INV_X1 U85 ( .A(acc_in[18]), .ZN(n213) );
  INV_X1 U86 ( .A(acc_in[17]), .ZN(n215) );
  INV_X1 U87 ( .A(acc_in[16]), .ZN(n217) );
  INV_X1 U88 ( .A(acc_in[15]), .ZN(n220) );
  INV_X1 U89 ( .A(x[6]), .ZN(n90) );
  INV_X1 U90 ( .A(x[7]), .ZN(n99) );
  AOI22_X1 U91 ( .A1(x[5]), .A2(n90), .B1(x[6]), .B2(n92), .ZN(n103) );
  INV_X1 U92 ( .A(n103), .ZN(n89) );
  INV_X1 U93 ( .A(weight[7]), .ZN(n115) );
  AOI22_X1 U94 ( .A1(x[7]), .A2(weight[7]), .B1(n115), .B2(n99), .ZN(n47) );
  OAI21_X1 U95 ( .B1(n98), .B2(n89), .A(n47), .ZN(n189) );
  INV_X1 U96 ( .A(acc_in[13]), .ZN(n188) );
  INV_X1 U97 ( .A(n98), .ZN(n49) );
  INV_X1 U98 ( .A(weight[6]), .ZN(n117) );
  AOI22_X1 U99 ( .A1(x[7]), .A2(n117), .B1(weight[6]), .B2(n99), .ZN(n51) );
  NAND2_X1 U100 ( .A1(n47), .A2(n89), .ZN(n48) );
  OAI21_X1 U101 ( .B1(n49), .B2(n51), .A(n48), .ZN(n187) );
  INV_X1 U102 ( .A(weight[5]), .ZN(n120) );
  AOI22_X1 U103 ( .A1(x[7]), .A2(weight[5]), .B1(n120), .B2(n99), .ZN(n52) );
  INV_X1 U104 ( .A(n52), .ZN(n50) );
  OAI22_X1 U105 ( .A1(n103), .A2(n51), .B1(n50), .B2(n49), .ZN(n55) );
  AOI22_X1 U106 ( .A1(x[5]), .A2(n115), .B1(weight[7]), .B2(n92), .ZN(n53) );
  AOI21_X1 U107 ( .B1(n64), .B2(n65), .A(n53), .ZN(n60) );
  INV_X1 U108 ( .A(weight[4]), .ZN(n95) );
  AOI22_X1 U109 ( .A1(x[7]), .A2(weight[4]), .B1(n95), .B2(n99), .ZN(n69) );
  AOI22_X1 U110 ( .A1(n69), .A2(n98), .B1(n52), .B2(n89), .ZN(n72) );
  AOI22_X1 U111 ( .A1(x[5]), .A2(n117), .B1(weight[6]), .B2(n92), .ZN(n63) );
  OAI22_X1 U112 ( .A1(n64), .A2(n53), .B1(n63), .B2(n65), .ZN(n54) );
  INV_X1 U113 ( .A(n54), .ZN(n71) );
  FA_X1 U114 ( .A(acc_in[12]), .B(acc_in[11]), .CI(n55), .CO(n186), .S(n56) );
  INV_X1 U115 ( .A(n56), .ZN(n58) );
  INV_X1 U116 ( .A(n57), .ZN(n226) );
  FA_X1 U117 ( .A(n60), .B(n59), .CI(n58), .CO(n57), .S(n61) );
  INV_X1 U118 ( .A(n61), .ZN(n229) );
  AOI22_X1 U119 ( .A1(x[5]), .A2(weight[5]), .B1(n120), .B2(n92), .ZN(n66) );
  INV_X1 U120 ( .A(n66), .ZN(n62) );
  OAI22_X1 U121 ( .A1(n64), .A2(n63), .B1(n62), .B2(n65), .ZN(n82) );
  AOI22_X1 U122 ( .A1(x[3]), .A2(n115), .B1(weight[7]), .B2(n94), .ZN(n67) );
  AOI21_X1 U123 ( .B1(n143), .B2(n141), .A(n67), .ZN(n86) );
  AOI22_X1 U124 ( .A1(x[5]), .A2(weight[4]), .B1(n95), .B2(n92), .ZN(n76) );
  INV_X1 U125 ( .A(n65), .ZN(n134) );
  AOI22_X1 U126 ( .A1(n76), .A2(n134), .B1(n66), .B2(n137), .ZN(n105) );
  AOI22_X1 U127 ( .A1(x[3]), .A2(n117), .B1(weight[6]), .B2(n94), .ZN(n74) );
  OAI22_X1 U128 ( .A1(n143), .A2(n67), .B1(n141), .B2(n74), .ZN(n68) );
  INV_X1 U129 ( .A(n68), .ZN(n104) );
  AOI22_X1 U130 ( .A1(x[7]), .A2(weight[3]), .B1(n75), .B2(n99), .ZN(n80) );
  AOI22_X1 U131 ( .A1(n80), .A2(n98), .B1(n69), .B2(n89), .ZN(n84) );
  INV_X1 U132 ( .A(n70), .ZN(n184) );
  FA_X1 U133 ( .A(acc_in[11]), .B(n72), .CI(n71), .CO(n59), .S(n73) );
  INV_X1 U134 ( .A(n73), .ZN(n183) );
  AOI22_X1 U135 ( .A1(x[3]), .A2(n120), .B1(weight[5]), .B2(n94), .ZN(n96) );
  OAI22_X1 U136 ( .A1(n143), .A2(n74), .B1(n141), .B2(n96), .ZN(n88) );
  NOR2_X1 U137 ( .A1(acc_in[8]), .A2(n88), .ZN(n108) );
  AOI22_X1 U138 ( .A1(x[5]), .A2(weight[3]), .B1(n75), .B2(n92), .ZN(n93) );
  AOI22_X1 U139 ( .A1(n93), .A2(n134), .B1(n76), .B2(n137), .ZN(n128) );
  AOI22_X1 U140 ( .A1(weight[1]), .A2(n99), .B1(x[7]), .B2(n77), .ZN(n102) );
  INV_X1 U141 ( .A(n102), .ZN(n78) );
  AOI22_X1 U142 ( .A1(x[7]), .A2(weight[2]), .B1(n91), .B2(n99), .ZN(n81) );
  AOI22_X1 U143 ( .A1(n78), .A2(n98), .B1(n81), .B2(n89), .ZN(n127) );
  AOI22_X1 U144 ( .A1(weight[7]), .A2(n114), .B1(n79), .B2(n115), .ZN(n126) );
  AOI22_X1 U145 ( .A1(n81), .A2(n98), .B1(n80), .B2(n89), .ZN(n106) );
  FA_X1 U146 ( .A(acc_in[9]), .B(acc_in[10]), .CI(n82), .CO(n185), .S(n83) );
  INV_X1 U147 ( .A(n83), .ZN(n180) );
  FA_X1 U148 ( .A(n86), .B(n85), .CI(n84), .CO(n70), .S(n179) );
  INV_X1 U149 ( .A(n87), .ZN(n232) );
  AOI21_X1 U150 ( .B1(acc_in[8]), .B2(n88), .A(n108), .ZN(n132) );
  NAND2_X1 U151 ( .A1(n89), .A2(weight[0]), .ZN(n119) );
  OAI211_X1 U152 ( .C1(n90), .C2(n92), .A(x[7]), .B(n119), .ZN(n149) );
  OAI22_X1 U153 ( .A1(n92), .A2(weight[2]), .B1(n91), .B2(x[5]), .ZN(n136) );
  AOI22_X1 U154 ( .A1(n136), .A2(n134), .B1(n93), .B2(n137), .ZN(n148) );
  AOI22_X1 U155 ( .A1(x[3]), .A2(n95), .B1(weight[4]), .B2(n94), .ZN(n142) );
  OAI22_X1 U156 ( .A1(n143), .A2(n96), .B1(n141), .B2(n142), .ZN(n97) );
  INV_X1 U157 ( .A(n97), .ZN(n147) );
  OAI221_X1 U158 ( .B1(weight[0]), .B2(x[7]), .C1(n100), .C2(n99), .A(n98), 
        .ZN(n101) );
  OAI21_X1 U159 ( .B1(n103), .B2(n102), .A(n101), .ZN(n125) );
  NAND2_X1 U160 ( .A1(acc_in[7]), .A2(n125), .ZN(n130) );
  FA_X1 U161 ( .A(acc_in[9]), .B(n105), .CI(n104), .CO(n85), .S(n111) );
  FA_X1 U162 ( .A(n108), .B(n107), .CI(n106), .CO(n181), .S(n110) );
  INV_X1 U163 ( .A(n109), .ZN(n235) );
  FA_X1 U164 ( .A(n112), .B(n111), .CI(n110), .CO(n109), .S(n113) );
  INV_X1 U165 ( .A(n113), .ZN(n238) );
  OAI22_X1 U166 ( .A1(n115), .A2(n114), .B1(n124), .B2(weight[7]), .ZN(n116)
         );
  AOI21_X1 U167 ( .B1(n117), .B2(n121), .A(n116), .ZN(n118) );
  INV_X1 U168 ( .A(n118), .ZN(n146) );
  INV_X1 U169 ( .A(n119), .ZN(n155) );
  AOI22_X1 U170 ( .A1(n122), .A2(weight[6]), .B1(n121), .B2(n120), .ZN(n123)
         );
  OAI21_X1 U171 ( .B1(weight[6]), .B2(n124), .A(n123), .ZN(n154) );
  XOR2_X1 U172 ( .A(acc_in[7]), .B(n125), .Z(n144) );
  FA_X1 U173 ( .A(n128), .B(n127), .CI(n126), .CO(n107), .S(n129) );
  INV_X1 U174 ( .A(n129), .ZN(n177) );
  FA_X1 U175 ( .A(n132), .B(n131), .CI(n130), .CO(n112), .S(n133) );
  INV_X1 U176 ( .A(n133), .ZN(n176) );
  AOI22_X1 U177 ( .A1(n137), .A2(n136), .B1(n135), .B2(n134), .ZN(n138) );
  INV_X1 U178 ( .A(n138), .ZN(n159) );
  INV_X1 U179 ( .A(n139), .ZN(n140) );
  OAI22_X1 U180 ( .A1(n143), .A2(n142), .B1(n141), .B2(n140), .ZN(n157) );
  FA_X1 U181 ( .A(n146), .B(n145), .CI(n144), .CO(n178), .S(n174) );
  FA_X1 U182 ( .A(n149), .B(n148), .CI(n147), .CO(n131), .S(n150) );
  INV_X1 U183 ( .A(n150), .ZN(n173) );
  FA_X1 U184 ( .A(n153), .B(n152), .CI(n151), .CO(n164), .S(n32) );
  FA_X1 U185 ( .A(acc_in[6]), .B(n155), .CI(n154), .CO(n145), .S(n156) );
  INV_X1 U186 ( .A(n156), .ZN(n163) );
  FA_X1 U187 ( .A(n159), .B(n158), .CI(n157), .CO(n175), .S(n160) );
  INV_X1 U188 ( .A(n160), .ZN(n162) );
  INV_X1 U189 ( .A(n161), .ZN(n244) );
  FA_X1 U190 ( .A(n164), .B(n163), .CI(n162), .CO(n161), .S(n165) );
  INV_X1 U191 ( .A(n165), .ZN(n247) );
  FA_X1 U192 ( .A(n168), .B(n167), .CI(n166), .CO(n246), .S(n33) );
  FA_X1 U193 ( .A(n171), .B(n170), .CI(n169), .CO(n172), .S(n34) );
  INV_X1 U194 ( .A(n172), .ZN(n245) );
  FA_X1 U195 ( .A(n175), .B(n174), .CI(n173), .CO(n241), .S(n242) );
  FA_X1 U196 ( .A(n178), .B(n177), .CI(n176), .CO(n237), .S(n239) );
  FA_X1 U197 ( .A(n181), .B(n180), .CI(n179), .CO(n87), .S(n182) );
  INV_X1 U198 ( .A(n182), .ZN(n233) );
  FA_X1 U199 ( .A(n185), .B(n184), .CI(n183), .CO(n228), .S(n230) );
  FA_X1 U200 ( .A(n188), .B(n187), .CI(n186), .CO(n223), .S(n224) );
  FA_X1 U201 ( .A(acc_in[14]), .B(acc_in[13]), .CI(n189), .CO(n219), .S(n221)
         );
  NAND2_X1 U202 ( .A1(acc_in[30]), .A2(n254), .ZN(n252) );
  OAI21_X1 U203 ( .B1(n254), .B2(acc_in[30]), .A(n252), .ZN(n190) );
  XNOR2_X1 U204 ( .A(n255), .B(n190), .ZN(acc_out[30]) );
  FA_X1 U205 ( .A(acc_in[28]), .B(n254), .CI(n191), .CO(n255), .S(acc_out[29])
         );
  FA_X1 U206 ( .A(acc_in[27]), .B(n193), .CI(n192), .CO(n191), .S(acc_out[28])
         );
  FA_X1 U207 ( .A(acc_in[26]), .B(n195), .CI(n194), .CO(n192), .S(acc_out[27])
         );
  FA_X1 U208 ( .A(acc_in[25]), .B(n197), .CI(n196), .CO(n194), .S(acc_out[26])
         );
  FA_X1 U209 ( .A(acc_in[24]), .B(n199), .CI(n198), .CO(n196), .S(acc_out[25])
         );
  FA_X1 U210 ( .A(acc_in[23]), .B(n201), .CI(n200), .CO(n198), .S(acc_out[24])
         );
  FA_X1 U211 ( .A(acc_in[22]), .B(n203), .CI(n202), .CO(n200), .S(acc_out[23])
         );
  FA_X1 U212 ( .A(acc_in[21]), .B(n205), .CI(n204), .CO(n202), .S(acc_out[22])
         );
  FA_X1 U213 ( .A(acc_in[20]), .B(n207), .CI(n206), .CO(n204), .S(acc_out[21])
         );
  FA_X1 U214 ( .A(acc_in[19]), .B(n209), .CI(n208), .CO(n206), .S(acc_out[20])
         );
  FA_X1 U215 ( .A(acc_in[18]), .B(n211), .CI(n210), .CO(n208), .S(acc_out[19])
         );
  FA_X1 U216 ( .A(acc_in[17]), .B(n213), .CI(n212), .CO(n210), .S(acc_out[18])
         );
  FA_X1 U217 ( .A(acc_in[16]), .B(n215), .CI(n214), .CO(n212), .S(acc_out[17])
         );
  FA_X1 U218 ( .A(acc_in[15]), .B(n217), .CI(n216), .CO(n214), .S(acc_out[16])
         );
  FA_X1 U219 ( .A(n220), .B(n219), .CI(n218), .CO(n216), .S(acc_out[15]) );
  FA_X1 U220 ( .A(n223), .B(n222), .CI(n221), .CO(n218), .S(acc_out[14]) );
  FA_X1 U221 ( .A(n226), .B(n225), .CI(n224), .CO(n222), .S(acc_out[13]) );
  FA_X1 U222 ( .A(n229), .B(n228), .CI(n227), .CO(n225), .S(acc_out[12]) );
  FA_X1 U223 ( .A(n232), .B(n231), .CI(n230), .CO(n227), .S(acc_out[11]) );
  FA_X1 U224 ( .A(n235), .B(n234), .CI(n233), .CO(n231), .S(acc_out[10]) );
  FA_X1 U225 ( .A(n238), .B(n237), .CI(n236), .CO(n234), .S(acc_out[9]) );
  FA_X1 U226 ( .A(n241), .B(n240), .CI(n239), .CO(n236), .S(acc_out[8]) );
  FA_X1 U227 ( .A(n244), .B(n243), .CI(n242), .CO(n240), .S(acc_out[7]) );
  FA_X1 U228 ( .A(n247), .B(n246), .CI(n245), .CO(n243), .S(acc_out[6]) );
  AOI21_X1 U229 ( .B1(n250), .B2(n249), .A(n248), .ZN(acc_out[2]) );
  XNOR2_X1 U230 ( .A(acc_in[0]), .B(n251), .ZN(acc_out[0]) );
  INV_X1 U231 ( .A(n255), .ZN(n253) );
  OAI221_X1 U232 ( .B1(n255), .B2(n254), .C1(n253), .C2(acc_in[30]), .A(n252), 
        .ZN(n256) );
  XNOR2_X1 U233 ( .A(acc_in[31]), .B(n256), .ZN(acc_out[31]) );
endmodule


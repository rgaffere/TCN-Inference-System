/////////////////////////////////////////////////////////////
// Created by: Synopsys DC Ultra(TM) in wire load mode
// Version   : X-2025.06-SP4
// Date      : Wed May 13 00:38:17 2026
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
         n255, n256, n257, n258, n259, n260;

  EO1 U1 ( .A(x[4]), .B(x[3]), .C(x[3]), .D(x[4]), .Z(n2) );
  IVP U2 ( .A(n2), .Z(n130) );
  IVP U3 ( .A(weight[2]), .Z(n91) );
  IVP U4 ( .A(x[5]), .Z(n119) );
  AO2 U5 ( .A(x[5]), .B(n91), .C(weight[2]), .D(n119), .Z(n15) );
  IVP U6 ( .A(weight[1]), .Z(n65) );
  AO2 U7 ( .A(x[5]), .B(weight[1]), .C(n65), .D(n119), .Z(n3) );
  NR2 U8 ( .A(x[4]), .B(x[5]), .Z(n1) );
  AO1P U9 ( .A(x[4]), .B(x[5]), .C(n2), .D(n1), .Z(n14) );
  EON1 U10 ( .A(n130), .B(n15), .C(n3), .D(n14), .Z(n29) );
  IVP U11 ( .A(weight[0]), .Z(n53) );
  AO2 U12 ( .A(x[5]), .B(weight[0]), .C(n53), .D(n119), .Z(n4) );
  AO2 U13 ( .A(n4), .B(n14), .C(n3), .D(n2), .Z(n38) );
  IVP U14 ( .A(acc_in[5]), .Z(n37) );
  NR2 U15 ( .A(n38), .B(n37), .Z(n36) );
  IVP U16 ( .A(x[1]), .Z(n100) );
  EO1 U17 ( .A(x[2]), .B(n100), .C(n100), .D(x[2]), .Z(n135) );
  IVP U18 ( .A(weight[4]), .Z(n105) );
  IVP U19 ( .A(x[3]), .Z(n121) );
  AO2 U20 ( .A(x[3]), .B(n105), .C(weight[4]), .D(n121), .Z(n17) );
  IVP U21 ( .A(weight[3]), .Z(n124) );
  AO2 U22 ( .A(x[3]), .B(weight[3]), .C(n124), .D(n121), .Z(n24) );
  ND2 U23 ( .A(x[3]), .B(x[2]), .Z(n5) );
  AO3 U24 ( .A(x[3]), .B(x[2]), .C(n135), .D(n5), .Z(n137) );
  IVP U25 ( .A(n137), .Z(n56) );
  EON1 U26 ( .A(n135), .B(n17), .C(n24), .D(n56), .Z(n28) );
  AN2P U27 ( .A(n100), .B(x[0]), .Z(n64) );
  IVP U28 ( .A(weight[7]), .Z(n175) );
  NR2 U29 ( .A(n64), .B(n175), .Z(n99) );
  ND2 U30 ( .A(x[1]), .B(x[0]), .Z(n69) );
  AN2P U31 ( .A(n69), .B(n175), .Z(n6) );
  NR2 U32 ( .A(x[0]), .B(n100), .Z(n66) );
  IVP U33 ( .A(n66), .Z(n60) );
  AO4 U34 ( .A(n99), .B(n6), .C(weight[6]), .D(n60), .Z(n89) );
  EO1 U35 ( .A(x[5]), .B(x[6]), .C(x[6]), .D(x[5]), .Z(n185) );
  IVP U36 ( .A(n185), .Z(n111) );
  NR2 U37 ( .A(n111), .B(n53), .Z(n26) );
  ND2 U38 ( .A(n64), .B(weight[6]), .Z(n8) );
  IVP U39 ( .A(weight[5]), .Z(n118) );
  ND2 U40 ( .A(n66), .B(n118), .Z(n7) );
  AO3 U41 ( .A(weight[6]), .B(n69), .C(n8), .D(n7), .Z(n25) );
  IVP U42 ( .A(x[7]), .Z(n176) );
  AO2 U43 ( .A(weight[0]), .B(x[7]), .C(n176), .D(n53), .Z(n10) );
  NR2 U44 ( .A(x[6]), .B(x[7]), .Z(n9) );
  AO1P U45 ( .A(x[6]), .B(x[7]), .C(n185), .D(n9), .Z(n186) );
  AO2 U46 ( .A(weight[1]), .B(x[7]), .C(n176), .D(n65), .Z(n92) );
  AO2 U47 ( .A(n10), .B(n186), .C(n92), .D(n185), .Z(n12) );
  IVP U48 ( .A(acc_in[7]), .Z(n11) );
  NR2 U49 ( .A(n12), .B(n11), .Z(n149) );
  AO6 U50 ( .A(n12), .B(n11), .C(n149), .Z(n87) );
  AO1P U51 ( .A(x[6]), .B(x[5]), .C(n176), .D(n26), .Z(n13) );
  IVP U52 ( .A(n13), .Z(n98) );
  IVP U53 ( .A(n14), .Z(n132) );
  AO4 U54 ( .A(n119), .B(n124), .C(weight[3]), .D(x[5]), .Z(n90) );
  AO4 U55 ( .A(n15), .B(n132), .C(n90), .D(n130), .Z(n16) );
  IVP U56 ( .A(n16), .Z(n97) );
  AO2 U57 ( .A(x[3]), .B(n118), .C(weight[5]), .D(n121), .Z(n94) );
  NR2 U58 ( .A(n94), .B(n135), .Z(n19) );
  NR2 U59 ( .A(n17), .B(n137), .Z(n18) );
  NR2 U60 ( .A(n19), .B(n18), .Z(n96) );
  IVP U61 ( .A(n20), .Z(n84) );
  NR2 U62 ( .A(n130), .B(n53), .Z(n49) );
  AO1P U63 ( .A(x[3]), .B(x[4]), .C(n119), .D(n49), .Z(n21) );
  IVP U64 ( .A(n21), .Z(n43) );
  NR2 U65 ( .A(weight[4]), .B(n60), .Z(n23) );
  NR2 U66 ( .A(n69), .B(weight[5]), .Z(n22) );
  AO1P U67 ( .A(weight[5]), .B(n64), .C(n23), .D(n22), .Z(n42) );
  AO2 U68 ( .A(x[3]), .B(weight[2]), .C(n91), .D(n121), .Z(n45) );
  IVP U69 ( .A(n135), .Z(n54) );
  AO2 U70 ( .A(n45), .B(n56), .C(n24), .D(n54), .Z(n41) );
  FA1A U71 ( .A(acc_in[6]), .B(n26), .CI(n25), .CO(n88), .S(n27) );
  IVP U72 ( .A(n27), .Z(n33) );
  FA1A U73 ( .A(n29), .B(n36), .CI(n28), .CO(n86), .S(n30) );
  IVP U74 ( .A(n30), .Z(n32) );
  IVP U75 ( .A(n31), .Z(n104) );
  FA1A U76 ( .A(n34), .B(n33), .CI(n32), .CO(n31), .S(n35) );
  IVP U77 ( .A(n35), .Z(n242) );
  AO6 U78 ( .A(n38), .B(n37), .C(n36), .Z(n83) );
  ND2 U79 ( .A(n64), .B(weight[4]), .Z(n40) );
  ND2 U80 ( .A(n66), .B(n124), .Z(n39) );
  AO3 U81 ( .A(weight[4]), .B(n69), .C(n40), .D(n39), .Z(n48) );
  FA1A U82 ( .A(n43), .B(n42), .CI(n41), .CO(n34), .S(n44) );
  IVP U83 ( .A(n44), .Z(n81) );
  AO2 U84 ( .A(x[3]), .B(weight[1]), .C(n65), .D(n121), .Z(n55) );
  AO2 U85 ( .A(n55), .B(n56), .C(n45), .D(n54), .Z(n79) );
  ND2 U86 ( .A(n64), .B(weight[3]), .Z(n47) );
  ND2 U87 ( .A(n66), .B(n91), .Z(n46) );
  AO3 U88 ( .A(weight[3]), .B(n69), .C(n47), .D(n46), .Z(n58) );
  ND2 U89 ( .A(acc_in[3]), .B(n58), .Z(n78) );
  FA1A U90 ( .A(acc_in[4]), .B(n49), .CI(n48), .CO(n82), .S(n50) );
  IVP U91 ( .A(n50), .Z(n77) );
  IVP U92 ( .A(n51), .Z(n245) );
  NR2 U93 ( .A(n135), .B(n53), .Z(n72) );
  AO1P U94 ( .A(x[2]), .B(x[1]), .C(n121), .D(n72), .Z(n52) );
  IVP U95 ( .A(n52), .Z(n75) );
  AO2 U96 ( .A(x[3]), .B(weight[0]), .C(n53), .D(n121), .Z(n57) );
  AO2 U97 ( .A(n57), .B(n56), .C(n55), .D(n54), .Z(n74) );
  AO7 U98 ( .A(acc_in[3]), .B(n58), .C(n78), .Z(n73) );
  IVP U99 ( .A(n59), .Z(n248) );
  IVP U100 ( .A(acc_in[1]), .Z(n182) );
  NR2 U101 ( .A(weight[0]), .B(n60), .Z(n62) );
  NR2 U102 ( .A(n69), .B(weight[1]), .Z(n61) );
  AO1P U103 ( .A(weight[1]), .B(n64), .C(n62), .D(n61), .Z(n181) );
  ND2 U104 ( .A(weight[0]), .B(x[0]), .Z(n63) );
  IVP U105 ( .A(n63), .Z(n254) );
  AO2 U106 ( .A(n254), .B(acc_in[0]), .C(x[1]), .D(n63), .Z(n180) );
  ND2 U107 ( .A(n64), .B(weight[2]), .Z(n68) );
  ND2 U108 ( .A(n66), .B(n65), .Z(n67) );
  AO3 U109 ( .A(weight[2]), .B(n69), .C(n68), .D(n67), .Z(n71) );
  IVP U110 ( .A(n70), .Z(n252) );
  NR2 U111 ( .A(n253), .B(n252), .Z(n251) );
  FA1A U112 ( .A(acc_in[2]), .B(n72), .CI(n71), .CO(n250), .S(n70) );
  FA1A U113 ( .A(n75), .B(n74), .CI(n73), .CO(n59), .S(n76) );
  IVP U114 ( .A(n76), .Z(n249) );
  FA1A U115 ( .A(n79), .B(n78), .CI(n77), .CO(n51), .S(n80) );
  IVP U116 ( .A(n80), .Z(n246) );
  FA1A U117 ( .A(n83), .B(n82), .CI(n81), .CO(n241), .S(n243) );
  FA1A U118 ( .A(n86), .B(n85), .CI(n84), .CO(n167), .S(n102) );
  FA1A U119 ( .A(n89), .B(n88), .CI(n87), .CO(n164), .S(n85) );
  AO2 U120 ( .A(x[5]), .B(n105), .C(weight[4]), .D(n119), .Z(n133) );
  AO4 U121 ( .A(n130), .B(n133), .C(n90), .D(n132), .Z(n150) );
  AO4 U122 ( .A(n176), .B(weight[2]), .C(n91), .D(x[7]), .Z(n143) );
  AO2 U123 ( .A(n185), .B(n143), .C(n92), .D(n186), .Z(n93) );
  IVP U124 ( .A(n93), .Z(n148) );
  IVP U125 ( .A(weight[6]), .Z(n109) );
  AO2 U126 ( .A(x[3]), .B(n109), .C(weight[6]), .D(n121), .Z(n138) );
  AO4 U127 ( .A(n135), .B(n138), .C(n94), .D(n137), .Z(n95) );
  NR2 U128 ( .A(acc_in[8]), .B(n95), .Z(n157) );
  AO6 U129 ( .A(acc_in[8]), .B(n95), .C(n157), .Z(n153) );
  FA1A U130 ( .A(n98), .B(n97), .CI(n96), .CO(n152), .S(n20) );
  AO6 U131 ( .A(n100), .B(n175), .C(n99), .Z(n151) );
  IVP U132 ( .A(n101), .Z(n162) );
  FA1A U133 ( .A(n104), .B(n103), .CI(n102), .CO(n166), .S(acc_out[7]) );
  AO2 U134 ( .A(x[5]), .B(n175), .C(weight[7]), .D(n119), .Z(n106) );
  AO6 U135 ( .A(n130), .B(n132), .C(n106), .Z(n116) );
  AO2 U136 ( .A(x[7]), .B(weight[4]), .C(n105), .D(n176), .Z(n125) );
  AO2 U137 ( .A(x[7]), .B(weight[5]), .C(n118), .D(n176), .Z(n110) );
  AO2 U138 ( .A(n125), .B(n186), .C(n110), .D(n185), .Z(n128) );
  NR2 U139 ( .A(n106), .B(n130), .Z(n108) );
  AO2 U140 ( .A(x[5]), .B(n109), .C(weight[6]), .D(n119), .Z(n120) );
  NR2 U141 ( .A(n120), .B(n132), .Z(n107) );
  NR2 U142 ( .A(n108), .B(n107), .Z(n127) );
  AO2 U143 ( .A(x[7]), .B(n109), .C(weight[6]), .D(n176), .Z(n178) );
  EON1 U144 ( .A(n111), .B(n178), .C(n110), .D(n186), .Z(n179) );
  IVP U145 ( .A(n112), .Z(n114) );
  IVP U146 ( .A(n113), .Z(n193) );
  FA1A U147 ( .A(n116), .B(n115), .CI(n114), .CO(n113), .S(n117) );
  IVP U148 ( .A(n117), .Z(n230) );
  AO4 U149 ( .A(n119), .B(n118), .C(weight[5]), .D(x[5]), .Z(n131) );
  AO4 U150 ( .A(n130), .B(n120), .C(n131), .D(n132), .Z(n122) );
  AO4 U151 ( .A(n121), .B(n175), .C(weight[7]), .D(x[3]), .Z(n136) );
  AO6 U152 ( .A(n135), .B(n137), .C(n136), .Z(n146) );
  FA1A U153 ( .A(acc_in[9]), .B(acc_in[10]), .CI(n122), .CO(n174), .S(n123) );
  IVP U154 ( .A(n123), .Z(n145) );
  AO2 U155 ( .A(x[7]), .B(weight[3]), .C(n124), .D(n176), .Z(n142) );
  AO2 U156 ( .A(n142), .B(n186), .C(n125), .D(n185), .Z(n144) );
  IVP U157 ( .A(n126), .Z(n173) );
  FA1A U158 ( .A(acc_in[11]), .B(n128), .CI(n127), .CO(n115), .S(n129) );
  IVP U159 ( .A(n129), .Z(n172) );
  AO4 U160 ( .A(n133), .B(n132), .C(n131), .D(n130), .Z(n134) );
  IVP U161 ( .A(n134), .Z(n141) );
  AO4 U162 ( .A(n138), .B(n137), .C(n136), .D(n135), .Z(n139) );
  IVP U163 ( .A(n139), .Z(n140) );
  FA1A U164 ( .A(acc_in[9]), .B(n141), .CI(n140), .CO(n170), .S(n156) );
  AO2 U165 ( .A(n143), .B(n186), .C(n142), .D(n185), .Z(n155) );
  FA1A U166 ( .A(n146), .B(n145), .CI(n144), .CO(n126), .S(n168) );
  IVP U167 ( .A(n147), .Z(n233) );
  FA1A U168 ( .A(n150), .B(n149), .CI(n148), .CO(n161), .S(n163) );
  FA1A U169 ( .A(n153), .B(n152), .CI(n151), .CO(n154), .S(n101) );
  IVP U170 ( .A(n154), .Z(n160) );
  FA1A U171 ( .A(n157), .B(n156), .CI(n155), .CO(n169), .S(n158) );
  IVP U172 ( .A(n158), .Z(n159) );
  FA1A U173 ( .A(n161), .B(n160), .CI(n159), .CO(n236), .S(n239) );
  FA1A U174 ( .A(n164), .B(n163), .CI(n162), .CO(n238), .S(n165) );
  FA1A U175 ( .A(n167), .B(n166), .CI(n165), .CO(n237), .S(acc_out[8]) );
  FA1A U176 ( .A(n170), .B(n169), .CI(n168), .CO(n147), .S(n171) );
  IVP U177 ( .A(n171), .Z(n234) );
  FA1A U178 ( .A(n174), .B(n173), .CI(n172), .CO(n229), .S(n231) );
  IVP U179 ( .A(acc_in[13]), .Z(n190) );
  IVP U180 ( .A(n186), .Z(n177) );
  AO4 U181 ( .A(n176), .B(weight[7]), .C(n175), .D(x[7]), .Z(n184) );
  EON1 U182 ( .A(n178), .B(n177), .C(n185), .D(n184), .Z(n189) );
  FA1A U183 ( .A(acc_in[12]), .B(acc_in[11]), .CI(n179), .CO(n188), .S(n112)
         );
  FA1A U184 ( .A(n182), .B(n181), .CI(n180), .CO(n253), .S(n183) );
  IVP U185 ( .A(n183), .Z(acc_out[1]) );
  IVP U186 ( .A(acc_in[29]), .Z(n258) );
  IVP U187 ( .A(acc_in[28]), .Z(n197) );
  IVP U188 ( .A(acc_in[27]), .Z(n199) );
  IVP U189 ( .A(acc_in[26]), .Z(n201) );
  IVP U190 ( .A(acc_in[25]), .Z(n203) );
  IVP U191 ( .A(acc_in[24]), .Z(n205) );
  IVP U192 ( .A(acc_in[23]), .Z(n207) );
  IVP U193 ( .A(acc_in[22]), .Z(n209) );
  IVP U194 ( .A(acc_in[21]), .Z(n211) );
  IVP U195 ( .A(acc_in[20]), .Z(n213) );
  IVP U196 ( .A(acc_in[19]), .Z(n215) );
  IVP U197 ( .A(acc_in[18]), .Z(n217) );
  IVP U198 ( .A(acc_in[17]), .Z(n219) );
  IVP U199 ( .A(acc_in[16]), .Z(n221) );
  IVP U200 ( .A(acc_in[15]), .Z(n224) );
  AO7 U201 ( .A(n186), .B(n185), .C(n184), .Z(n187) );
  FA1A U202 ( .A(acc_in[14]), .B(acc_in[13]), .CI(n187), .CO(n223), .S(n227)
         );
  FA1A U203 ( .A(n190), .B(n189), .CI(n188), .CO(n226), .S(n191) );
  FA1A U204 ( .A(n193), .B(n192), .CI(n191), .CO(n225), .S(acc_out[13]) );
  ND2 U205 ( .A(acc_in[30]), .B(n258), .Z(n256) );
  AO7 U206 ( .A(n258), .B(acc_in[30]), .C(n256), .Z(n195) );
  FA1A U207 ( .A(acc_in[28]), .B(n258), .CI(n194), .CO(n259), .S(acc_out[29])
         );
  EN U208 ( .A(n195), .B(n259), .Z(acc_out[30]) );
  FA1A U209 ( .A(acc_in[27]), .B(n197), .CI(n196), .CO(n194), .S(acc_out[28])
         );
  FA1A U210 ( .A(acc_in[26]), .B(n199), .CI(n198), .CO(n196), .S(acc_out[27])
         );
  FA1A U211 ( .A(acc_in[25]), .B(n201), .CI(n200), .CO(n198), .S(acc_out[26])
         );
  FA1A U212 ( .A(acc_in[24]), .B(n203), .CI(n202), .CO(n200), .S(acc_out[25])
         );
  FA1A U213 ( .A(acc_in[23]), .B(n205), .CI(n204), .CO(n202), .S(acc_out[24])
         );
  FA1A U214 ( .A(acc_in[22]), .B(n207), .CI(n206), .CO(n204), .S(acc_out[23])
         );
  FA1A U215 ( .A(acc_in[21]), .B(n209), .CI(n208), .CO(n206), .S(acc_out[22])
         );
  FA1A U216 ( .A(acc_in[20]), .B(n211), .CI(n210), .CO(n208), .S(acc_out[21])
         );
  FA1A U217 ( .A(acc_in[19]), .B(n213), .CI(n212), .CO(n210), .S(acc_out[20])
         );
  FA1A U218 ( .A(acc_in[18]), .B(n215), .CI(n214), .CO(n212), .S(acc_out[19])
         );
  FA1A U219 ( .A(acc_in[17]), .B(n217), .CI(n216), .CO(n214), .S(acc_out[18])
         );
  FA1A U220 ( .A(acc_in[16]), .B(n219), .CI(n218), .CO(n216), .S(acc_out[17])
         );
  FA1A U221 ( .A(acc_in[15]), .B(n221), .CI(n220), .CO(n218), .S(acc_out[16])
         );
  FA1A U222 ( .A(n224), .B(n223), .CI(n222), .CO(n220), .S(acc_out[15]) );
  FA1A U223 ( .A(n227), .B(n226), .CI(n225), .CO(n222), .S(acc_out[14]) );
  FA1A U224 ( .A(n230), .B(n229), .CI(n228), .CO(n192), .S(acc_out[12]) );
  FA1A U225 ( .A(n233), .B(n232), .CI(n231), .CO(n228), .S(acc_out[11]) );
  FA1A U226 ( .A(n236), .B(n235), .CI(n234), .CO(n232), .S(acc_out[10]) );
  FA1A U227 ( .A(n239), .B(n238), .CI(n237), .CO(n235), .S(acc_out[9]) );
  FA1A U228 ( .A(n242), .B(n241), .CI(n240), .CO(n103), .S(acc_out[6]) );
  FA1A U229 ( .A(n245), .B(n244), .CI(n243), .CO(n240), .S(acc_out[5]) );
  FA1A U230 ( .A(n248), .B(n247), .CI(n246), .CO(n244), .S(acc_out[4]) );
  FA1A U231 ( .A(n251), .B(n250), .CI(n249), .CO(n247), .S(acc_out[3]) );
  AO6 U232 ( .A(n253), .B(n252), .C(n251), .Z(acc_out[2]) );
  EO1 U233 ( .A(acc_in[0]), .B(n254), .C(n254), .D(acc_in[0]), .Z(acc_out[0])
         );
  IVP U234 ( .A(acc_in[30]), .Z(n255) );
  ND2 U235 ( .A(n255), .B(n259), .Z(n257) );
  AO3 U236 ( .A(n259), .B(n258), .C(n257), .D(n256), .Z(n260) );
  EN U237 ( .A(acc_in[31]), .B(n260), .Z(acc_out[31]) );
endmodule


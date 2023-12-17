\ Model Shortest Path Problem
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  15 x_1,2 + 25 x_1,4 + 45 x_1,3 + 30 x_2,5 + 2 x_2,4 + 50 x_4,7 + 2 x_4,3
   + 25 x_3,6 + 2 x_5,7 + x_6,7
Subject To
 start_flow: x_1,2 + x_1,4 + x_1,3 = 1
 end_flow: x_4,7 + x_5,7 + x_6,7 = 1
 限制节点进出平衡: = 0
 限制节点进出平衡: x_1,2 - x_2,5 - x_2,4 = 0
 限制节点进出平衡: x_1,3 + x_4,3 - x_3,6 = 0
 限制节点进出平衡: x_1,4 + x_2,4 - x_4,7 - x_4,3 = 0
 限制节点进出平衡: x_2,5 - x_5,7 = 0
 限制节点进出平衡: x_3,6 - x_6,7 = 0
 限制节点进出平衡: = 0
Bounds
Binaries
 x_1,2 x_1,4 x_1,3 x_2,5 x_2,4 x_4,7 x_4,3 x_3,6 x_5,7 x_6,7
End

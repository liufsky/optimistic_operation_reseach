"""
同其他文件的使用逻辑，只需要更改少部分的文件
明细就可以重新使用该文件的功能
该文件去除01变量强约束改为下0上1连续变量
"""
from gurobipy import *
import pandas as pd
import numpy as np

""" 
第一步创立节点的名称和序号
"""
Nodes = ['1', '2', '3', '4', '5', '6', '7']

"""
创建弧的字典
"""
Arcs = {('1', '2'): 15
        , ('1', '4'): 25
        , ('1', '3'): 45
        , ('2', '5'): 30
        , ('2', '4'): 2
        , ('4', '7'): 50
        , ('4', '3'): 2
        , ('3', '6'): 25
        , ('5', '7'): 2
        , ('6', '7'): 1
}
#命名你的模型
model = Model('Shortest Path Problem none integer var')

#加入对应的变量，表示是否从i走到j节点
X = {}
for key in Arcs.keys():
    index = 'x_'+key[0]+','+key[1]
    X[key] = model.addVar(lb=0,ub=1,vtype=GRB.CONTINUOUS, name=index)

#添加目标函数的功能，根据弧长*01变量以表示走最短路的代价
obj = LinExpr(0)
for key in Arcs.keys():
    obj.addTerms(Arcs[key], X[key])
#添加目标函数的目标方向，是求路线最短最小
model.setObjective(obj, GRB.MINIMIZE)

"""
以下的限制1和限制2用以约束从头部走出和走入尾部的只能各有一条
即确定出发点和到达点
"""
lhs_1 = LinExpr(0)
lhs_2 = LinExpr(0)
for key in Arcs.keys():
    if (key[0] == '1'):
        lhs_1.addTerms(1,X[key])
    elif (key[1] == '7'):
        lhs_2.addTerms(1,X[key])
model.addConstr(lhs_1 == 1, name='start flow')
model.addConstr(lhs_2 == 1, name='end flow')

#约束3对于其中的每一个点都要保证进出平衡
#虽然说可能存在多进多出的情况，但是由于初试节点和结束节点只能有一个的约束，保证了单进单出
for node in Nodes:
    lhs = LinExpr(0)
    if(node != '1' and node != '7'):
        for key in Arcs.keys():
            if(key[1] == node):
                lhs.addTerms(1, X[key])
            elif(key[0] == node):
                lhs.addTerms(-1, X[key])
    model.addConstr(lhs == 0, name = '限制节点进出平衡')

model.write('model_shortest_path_problem_none_integer_var.lp')
model.optimize()

print('Optimal solution:', model.objVal)
for var in model.getVars():
    if(var.x > 0 ):
        print(var.varName, '\t', var.x)

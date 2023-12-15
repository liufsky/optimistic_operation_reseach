"""
本文只需要改动矩阵的大小
价值矩阵，就可以重新用于解决新的问题
"""
from gurobipy import *
import pandas as pd
import numpy as np
import random
"""
第一步我们需要生成一个可以用来计算的指派价值矩阵用以计算
使用np函数先生成一个全0的函数，然后我们对其中的每一个元
素进行遍历并且对其进行使用随机种子生成随机数
"""
#生成指派价值矩阵的行和列的大小，此处我们选择5*5
employee_num = job_num = 5
#生成5*5矩阵
cost_matrix = np.zeros((employee_num, job_num))
#遍历矩阵中的每一个元素，并且通过random对其进行赋值
for i in range(employee_num):
    for j in range(job_num):
        random.seed(i * employee_num + j)
        cost_matrix[i][j] = round(10 * random.random() + 5, 0)

"""
接下来我们需要构建一个模型选择，称其为model object
可以直接选用Model中的"assignment problem"
"""
model = Model("assignment problem")

"""
我们通过循环的方法构建一个新的决策变量矩阵，详细的生成过程
在下方都详细的表明了
"""
#通过循环往列表中添加决策变量，此处使用列表自动生成的方法
x = [[[] for i in range(employee_num)] for j in range(job_num)]
#对内部进行遍历，并且赋予每一个决策变量对应名称
for i in range(employee_num):
    for j in range(job_num):
        x[i][j] = model.addVar(vtype= GRB.BINARY, name="X_"+str(i)+"_"+str(j))
#上述含义为添加01变量，并且为他们各自取了个名字

"""
构建真正的目标函数
"""
obj = LinExpr(0)
#先生成一个空的线性方程，用以描述总的指派时间

for i in range(employee_num):
    for j in range(job_num):
        obj.addTerms(cost_matrix[i][j], x[i][j])

#其中cost_matrix是价值系数，而x[i][j]是01变量，通过塞入函数，得到目标函数

model.setObjective(obj, GRB.MINIMIZE)
#设定函数的目标为追求总指派时间最小化

"""
接下来我们需要构建两个约束方程，
以表示每一项工作只有一个人做
以及表示每一个人只做一项工作
"""
#约束1，对每一列做求和为1的限制
for j in range(employee_num):
    expr = LinExpr(0)
    for i in range(job_num):
        expr.addTerms(1, x[i][j])
    model.addConstr(expr == 1, name="D_"+str(j))

#约束2
for i in range(job_num):
    expr = LinExpr(0)
    for j in range(employee_num):
        expr.addTerms(1, x[i][j])
    model.addConstr(expr == 1, name="R_"+str(i))
#求解该函数
model.write("model.lp")
model.optimize()

#打印对应答案表
for var in model.getVars():
    if(var.x > 0):
        print(var.varName, '\t', var.x)
print('obejctive:', model.ObjVal)
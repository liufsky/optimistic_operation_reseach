from gurobipy import *
import copy
import pandas as pd
import numpy as np
"""
两个部分:
1.初始化模型(直接复制pure_linar_version的代码)
2.使用次梯度算法进行拉格朗日松弛
"""
#设置log记录上下界，步长，和theta
LBlog = []
UBlog = []
stepSizelog = []
thetalog = []
timer = 0
#第一部分初始化模型：
#构建汽车座椅需求矩阵
#矩阵行为方向，矩阵列为车型，有点奇怪，但是别搞混了
car_seat_Matrix = np.array([[10,11,26,18,18,25],[15,17,22,9,14,19],[12,17,15,11,28,6],[20,22,21,24,16,18]])
car_type = 6
car_seattype = 4

#并且设置一个需要被松弛的约束合集
relaxedCons = []

#接下来我们需要构建一个模型
#可以为其取名"linar_parallel_problem"
model = Model("linar_parallel_problem")

#接下来通过列表自动生成的办法生成四种机器的变量矩阵
#老样子为其设定1-4的名称:X_1,X_2,X_3,X_4/Y_1,Y_2,Y_3,Y_4
#其中X为整数变量,Y为01变量
X_1 = [[[] for i in range(car_type)] for j in range(car_seattype)]
X_2 = [[[] for i in range(car_type)] for j in range(car_seattype)]
X_3 = [[] for i in range(car_type)]
X_4 = [[] for i in range(car_type)]
Y_1 = [[[] for i in range(car_type)] for j in range(car_seattype)]
Y_2 = [[[] for i in range(car_type)] for j in range(car_seattype)]
Y_3 = [[] for i in range(car_type)]
Y_4 = [[] for i in range(car_type)]
C_group = [[],[],[],[]]
Z_group = [[],[],[],[]]
#接下来把这些变量设置为决策变量
#以下设置single类的机器
for i in range(car_seattype):
    for j in range(car_type):
        X_1[i][j] = model.addVar(lb= 0, ub= car_seat_Matrix[i][j], vtype=GRB.INTEGER, name="X1[" + str(i+1) + "][" + str(j+1)+"]")
for i in range(car_seattype):
    for j in range(car_type):
        X_2[i][j] = model.addVar(lb= 0, ub= car_seat_Matrix[i][j], vtype=GRB.INTEGER, name="X2[" + str(i+1) + "][" + str(j+1)+"]")
#以下设置multiple类的机器
for j in range(car_type):
    max_jcar_seat = max(car_seat_Matrix[0][j], car_seat_Matrix[1][j], car_seat_Matrix[2][j], car_seat_Matrix[3][j])
    X_3[j]= model.addVar(lb= 0, ub= max_jcar_seat, vtype=GRB.INTEGER, name="X3[" + str(j+1)+"]")
for j in range(car_type):
    max_jcar_seat = max(car_seat_Matrix[0][j], car_seat_Matrix[1][j], car_seat_Matrix[2][j], car_seat_Matrix[3][j])
    X_4[j]= model.addVar(lb= 0, ub= max_jcar_seat, vtype=GRB.INTEGER, name="X4[" + str(j+1)+"]")
#以下设置single类的机器的Y类型变量
for i in range(car_seattype):
    for j in range(car_type):
        Y_1[i][j] = model.addVar(vtype=GRB.BINARY, name="Y1[" + str(i+1) + "][" + str(j+1)+"]")
for i in range(car_seattype):
    for j in range(car_type):
        Y_2[i][j] = model.addVar(vtype=GRB.BINARY, name="Y2[" + str(i+1) + "][" + str(j+1)+"]")
#以下设置multiple类的机器的Y类型变量
for j in range(car_type):
    Y_3[j]= model.addVar(vtype=GRB.BINARY, name="Y3[" + str(j+1)+"]")
for j in range(car_type):
    Y_4[j]= model.addVar(vtype=GRB.BINARY, name="Y4[" + str(j+1)+"]")
#设置时间长度的变量C
for i in range(4):
    C_group[i] = model.addVar(vtype=GRB.CONTINUOUS, name="C["+str(i+1)+"]")
C_total = model.addVar(vtype=GRB.CONTINUOUS, name="C_TOTAL")
#设置控制求极大值的Z变量
for i in range(4):
    Z_group[i] = model.addVar(vtype=GRB.BINARY, name="Z["+str(i+1)+"]")
"""
以上设置完了变量，和变量的上下界限，接下来需要建立四个约束和目标函数
"""
#目标函数，目标函数为最小化C
obj = LinExpr(0)
obj.addTerms(1, C_total)
model.setObjective(obj, GRB.MINIMIZE)

#约束1,为了找到每个机器的C-最大makespan
#对于每一台机器，我们把时间区分为生产时间和换机时间。
T_1 = LinExpr(0)
T_2 = LinExpr(0)
T_3 = LinExpr(0)
T_4 = LinExpr(0)
#首先，先把每个机器的生产时间计算出来，这部分是single类的
for i in range(car_seattype):
    for j in range(car_type):
        T_1.addTerms(5, X_1[i][j])
        T_2.addTerms(5, X_2[i][j])
for j in range(car_type):
    T_3.addTerms(5, X_3[j])
    T_4.addTerms(5, X_4[j])
#添加换模型的时间的需求
for i in range(car_seattype):
    for j in range(car_type):
        if (j%2 == 1):
            T_1.addTerms(1.5, Y_1[i][j])
            T_2.addTerms(1.5, Y_2[i][j])
        else:
            T_1.addTerms(1, Y_1[i][j])
            T_2.addTerms(1, Y_2[i][j])
for j in range(car_type):
    if (j%2 == 1):
        T_3.addTerms(1.5, Y_3[j])
        T_4.addTerms(1.5, Y_4[j])
    else:
        T_3.addTerms(1, Y_3[j])
        T_4.addTerms(1, Y_4[j])
model.addConstr(C_group[0] == T_1 + 2.5, name="机器1的时间")
model.addConstr(C_group[1] == T_2 + 2.5, name="机器2的时间")
model.addConstr(C_group[2] == T_3 + 2.5, name="机器3的时间")
model.addConstr(C_group[3] == T_4 + 2.5, name="机器4的时间")

#约束2，体现每种座椅生产需求都要被满足
#在拉格朗日松弛中，我们将约束2设置为被松弛的变量
for i in range(car_seattype):
    for j in range(car_type):
        expr = LinExpr(0)
        expr.addTerms((1,1,1,1),(X_1[i][j],X_2[i][j],X_3[j],X_4[j]))
        relaxedCons.append(model.addConstr(expr >= car_seat_Matrix[i][j],name=f"第{j+1}种车的{i+1}方向需满足："))

#约束3,体现没有Y就不能被生产
#此处需要设定一个M变量罚因子
M = -10000
for i in range(car_seattype):
    for j in range(car_type):
        expr = LinExpr(0)
        expr.addTerms((1, M), (X_1[i][j], Y_1[i][j]))
        model.addConstr(expr <= 0,name=f'机器1需要换{j+1}车{i+1}模方可生产：')
for i in range(car_seattype):
    for j in range(car_type):
        expr = LinExpr(0)
        expr.addTerms((1, M), (X_2[i][j], Y_2[i][j]))
        model.addConstr(expr <= 0,name=f'机器2需要换{j+1}车{i+1}模方可生产：')
for j in range(car_type):
    expr = LinExpr(0)
    expr.addTerms((1, M), (X_3[j], Y_3[j]))
    model.addConstr(expr <= 0, name=f'机器3需要换{j + 1}车的模具方可生产：')
for j in range(car_type):
    expr = LinExpr(0)
    expr.addTerms((1, M), (X_4[j], Y_4[j]))
    model.addConstr(expr <= 0, name=f'机器4需要换{j + 1}车的模具方可生产：')

#约束4,此处为求最大
M = 10000
for i in range(4):
    expr = LinExpr(0)
    expr.addTerms((1, -1), (C_group[i], C_total))
    model.addConstr(expr <= 0, name=f"此处为求极大值的第{i+1}个函数")
for i in range(4):
    expr = LinExpr(0)
    expr.addTerms((1, -1, -M), (C_group[i], C_total, Z_group[i]))
    model.addConstr(expr >= -M, name=f"此处为求极大值的第{i+5}个函数")
control_lhs = LinExpr(0)
control_lhs.addTerms((1, 1, 1, 1), (Z_group[0], Z_group[1], Z_group[2], Z_group[3]))
model.addConstr(control_lhs >= 1, name=f"此处为求极大值的第9个函数")

#拉格朗日松弛：次梯度算法
#首先我们需要设置以下这些拉格朗日松弛的参数
noChangeCut = 0
noChangeCutLimit = 5
squareSum = 0.0     #这是个浮点数
stepSize = 0.0      #这是步长
theta = 2.0         #这是theta参数
LB = 0.0
UB = 0.0            #设定一个上界和下界
#这地方还需要加入两个东西
Lag_multiplier = np.zeros((4,6))      #拉格朗日乘子列表，因为有24个需要被松弛的式子
slack = np.zeros((4,6))              #松弛

#拉格朗日上界LB = relaxUB通过松弛原有的01和整数变量约束得到一个更小的下界
model.write("test1.lp")
model_copy = model.copy()
model_copy.write('test2.lp')
relaxed_model = model_copy.relax()
relaxed_model.write('test3.lp')
relaxed_model.optimize()
LB = relaxed_model.ObjVal
print('下界:', LB)
LB = 0
#拉格朗日上界UB = 通过启发式算法得到一个次优解
# 设置启发式参数
model_copy.setParam('Heuristics', 0.1)  # 降低启发式强度
model_copy.setParam('TimeLimit', 10)    # 设置时间限制
model_copy.setParam('MIPGap', 0.1)      # 增加MIP间隙
model_copy.setParam('Cuts', 0)          # 减少割的生成
# 求解模型
model_copy.optimize()
UB = model_copy.ObjVal
print('上界:', UB)

#用以指示当前模型是否为拉格朗日松弛
isModelLagrangianRelaxed = False

#取出松弛的约束
relaxedConsNum = len(relaxedCons)
for i in range(relaxedConsNum):
    model.remove(relaxedCons[i])
relaxedCons = []

#用以指示循环次数的指标
MaxIter = 60

#接下来我们直接开始'lagrangian relaxtion'主循环！！
for iter in range(MaxIter):

    #生成拉格朗日的后面的子项
    obj_lagrangian_relaxed_term = quicksum(
        [Lag_multiplier[i][j] * (car_seat_Matrix[i][j] - X_1[i][j] - X_2[i][j] - X_3[j] - X_4[j]) for i in range(4) for
         j in range(6)])

    #拉格朗日的目标函数
    model.setObjective(obj + obj_lagrangian_relaxed_term, GRB.MINIMIZE)

    #解决并且计算松弛公式并且得到lower bound
    model.update()
    model.optimize()
    print('下界最优解为:', model.ObjVal)
    if timer == MaxIter - 1:
        # 打印对应答案表
        for var in model.getVars():
            if (var.x > 0):
                print(var.varName, '\t', var.x)
        print('obejctive:', model.ObjVal)
    timer += 1
    #基于解对每一个被松弛约束计算slack
    for i in range(4):
        for j in range(6):
            slack[i][j] = sum([X_1[i][j].x, X_2[i][j].x, X_3[j].x, X_4[j].x]) - car_seat_Matrix[i][j]
    print(slack)

    #如果存在可优化，升级lower bound
    if(model.ObjVal > LB + 1e-6):
        #1e-6更新太慢了
        LB = model.ObjVal
        noChangeCut = 0
    else:
        noChangeCut += 1

    #如果noChangecut在一定的limit以内没有优化，theta减半：
    if(noChangeCut == noChangeCutLimit):
        theta = theta / 2.0
        noChangeCut = 0

    #计算squaresum
    squareSum = sum(slack[i][j]**2.0 for i in range(4) for j in range(6))
    print(squareSum)

    #更新步长
    stepSize = theta * (UB - model.ObjVal) / squareSum

    #更新拉格朗日乘子
    for i in range(4):
        for j in range(6):
            if(Lag_multiplier[i][j] > stepSize * slack[i][j]):
                Lag_multiplier[i][j] = Lag_multiplier[i][j] - stepSize * slack[i][j]
            else:
                Lag_multiplier[i][j] = 0.0


    LBlog.append(LB)
    UBlog.append(UB)
    stepSizelog.append(stepSize)
    thetalog.append(theta)

#最后报告
print("\n  ------------ Iteration log information ------------  \n")
print("  Iter        LB          UB(固定)          theta          stepSize")

for i in range(len(LBlog)):
    print("  %3d  %12.6f  %12.6f  %8.6f  %8.6f"\
          %(i,LBlog[i],UBlog[i],thetalog[i],stepSizelog[i]))




from gurobipy import *
import numpy as np
"""
区别于原有求解器建模，我们将求极大值部分更改为线性模型。
首先我们需要建立
1.汽车座椅需求矩阵，此矩阵表示全部的汽车座椅需求清单
2.对应四种机器，每种机器生成两个矩阵X和Y。
X_i矩阵为第i种机器生产第几行几列的产品
Y_i矩阵为第i中机器是否生产第几行几列的产品
附加以下几种约束
(1)满足生产数量
(2)约束生产的转换和启动条件
(3)小于最大产品生产上限
*(4)此处求C_total = max(C_1,C_2,C_3,C_4)

例外，我们添加一个约束，T>=max(f(x)),然后目标函数是min T
以此实现最小化最大问题的目标条件
"""
#构建汽车座椅需求矩阵
#矩阵行为方向，矩阵列为车型，有点奇怪，但是别搞混了
car_seat_Matrix = np.array([[10,11,26,18,18,25],[15,17,22,9,14,19],[12,17,15,11,28,6],[20,22,21,24,16,18]])
car_type = 6
car_seattype = 4

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
for i in range(car_seattype):
    for j in range(car_type):
        expr = LinExpr(0)
        expr.addTerms((1,1,1,1),(X_1[i][j],X_2[i][j],X_3[j],X_4[j]))
        model.addConstr(expr >= car_seat_Matrix[i][j],name=f"第{j+1}种车的{i+1}方向需满足：")

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

#最优化运行，并且给出约束性表格
model.write("pure_linar_version.lp")
model.optimize()

#打印对应答案表
for var in model.getVars():
    if(var.x > 0):
        print(var.varName, '\t', var.x)
print('obejctive:', model.ObjVal)

#以表的形式输出

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from IVMD import MSSA

file1 = pd.read_excel('ces.xlsx')
y = file1.iloc[:, -1].values.reshape(-1, 1)
file2 = pd.read_csv('预测值6.csv')
x1 = file2.iloc[:, 1].values.reshape(-1, 1)
x2 = file2.iloc[:, 2].values.reshape(-1, 1)
x3 = file2.iloc[:, 3].values.reshape(-1, 1)
x4 = file2.iloc[:, 4].values.reshape(-1, 1)
x5 = file2.iloc[:, 5].values.reshape(-1, 1)
x6 = file2.iloc[:, 6].values.reshape(-1, 1)
x7 = file2.iloc[:, 7].values.reshape(-1, 1)
x8 = file2.iloc[:, 8].values.reshape(-1, 1)
x9 = file2.iloc[:, 9].values.reshape(-1, 1)
x10 = file2.iloc[:, 10].values.reshape(-1, 1)
x11 = file2.iloc[:, 11].values.reshape(-1, 1)
x12 = file2.iloc[:, 12].values.reshape(-1, 1)
x13 = file2.iloc[:, 13].values.reshape(-1, 1)
'''x14 = file2.iloc[:, 14].values.reshape(-1, 1)
x15 = file2.iloc[:, 15].values.reshape(-1, 1)'''


def fitness(X):
    w1 = X[0]
    w2 = X[1]
    w3 = X[2]
    w4 = X[3]
    w5 = X[4]
    w6 = X[5]
    w7 = X[6]
    w8 = X[7]
    w9 = X[8]
    w10 = X[9]
    w11 = X[10]
    w12 = X[11]
    w13 = X[12]
    w14 = X[13]
    w15 = X[14]

    pred = w1*x1 + w2*x2 + w3*x3 + w4*x4 + w5*x5 + w6*x6 + w7*x7 + w8*x8 + w9*x9 + w10*x10 + w11*x11 + w12*x12 + w13*x13 #+ w14*x14 + w15*x15
    O = np.sqrt(np.mean(np.square(pred - y)))  # rmse
    return O


# 主函数
# 设置参数
pop = 300  # 种群数量
MaxIter = 1000  # 最大迭代次数
dim = 16  # 维度
lb = [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2]  # 下边界
ub = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]  # 上边界
# 适应度函数选择
fobj = fitness
# 改进麻雀
GbestScore, GbestPositon, Curve1 = MSSA(pop, dim, lb, ub, MaxIter, fobj)

print('------改进麻雀结果--------------')
print('最优适应度值：', GbestScore)
# print('最优解：', GbestPositon1[0, 0], GbestPositon1[0, 1], GbestPositon1[0, 2], GbestPositon1[0, 3], GbestPositon1[0, 4], GbestPositon1[0, 5], GbestPositon1[0, 6], GbestPositon1[0, 7], GbestPositon1[0, 8], GbestPositon1[0, 9], GbestPositon1[0, 10], GbestPositon1[0, 11], GbestPositon1[0, 12], GbestPositon1[0, 13], GbestPositon1[0, 14])
pred1 = GbestPositon[0, 0]*x1 + GbestPositon[0, 1]*x2 + GbestPositon[0, 2]*x3 + GbestPositon[0, 3]*x4 + GbestPositon[0, 4]*x5 \
        + GbestPositon[0, 5]*x6 + GbestPositon[0, 6]*x7 + GbestPositon[0, 7]*x8 + GbestPositon[0, 8]*x9 + GbestPositon[0, 9]*x10 \
        + GbestPositon[0, 10]*x11 + GbestPositon[0, 11]*x12 + GbestPositon[0, 12]*x13 #+ GbestPositon[0, 13]*x14 + GbestPositon[0, 14]*x15
pred1 = pd.DataFrame(pred1)
pred1.to_csv('MSSA预测.csv')
# 绘制适应度曲线-
plt.figure(1)
plt.semilogy(Curve1, 'r-', linewidth=2)
plt.xlabel('Iteration', fontsize='medium')
plt.ylabel("Fitness", fontsize='medium')
plt.grid()
plt.title('MSSA', fontsize='large')
plt.show()







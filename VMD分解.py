# -*- coding: utf-8 -*-
from vmdpy import VMD
from IVMD import MSSA, fitness
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# In[] 加载数据
file = pd.read_excel('15朝阳广场进站.xlsx')
data = file.iloc[:, -1].values.reshape(-1,)
data[data < 0, ] = 0
t = np.arange(data.shape[0])
tau = 0
DC = 0
init = 1
tol = 1e-7
# 优化VMD的参数
GbestScore1, GbestPositon1, Curve1 = MSSA(30, 2, [1, 1], [20, 10000], 20, fitness)

# 获得最佳K和α
alpha = GbestPositon1[0, 1]
K = int(GbestPositon1[0, 0])

# VMD分解
imf, u_hat, omega = VMD(data, alpha, tau, K, DC, init, tol)

plt.figure()
plt.plot(t, data)
plt.ylabel('Original Data')
plt.xlabel('Times')
plt.show()


plt.figure()
for i in range(imf.shape[0]):
    plt.subplot(imf.shape[0], 1, i+1)
    plt.plot(t, imf[i, :])
    plt.ylabel('IMF'+str(i+1))
plt.show()

test_label = np.array(imf).reshape(imf.shape[0], -1).sum(axis=0).reshape(-1, )
dataframe = pd.DataFrame(test_label)
dataframe.to_csv('重构值.csv')

plt.figure()
plt.plot(data, c='r', label='true')
plt.plot(test_label, c='b', label='predict')
plt.legend()
plt.show()

for i in range(K):
    a = imf[i, :]
    dataframe = pd.DataFrame({'v{}'.format(i+1): a})
    dataframe.to_csv("VMD分解/15朝阳广场进站IVMDban-%d.csv" % (i+1), index=False, sep=',')

# 保存分解的IMF数据 与原始数据用于后续建模
np.savez('结果/15朝阳广场进站IVMD-data.npz', imf=imf, data=data)
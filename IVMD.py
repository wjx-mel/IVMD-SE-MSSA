import numpy as np
import random
import copy
import pandas as pd
from matplotlib import pyplot as plt
from vmdpy import VMD
from scipy.fftpack import hilbert
from math import log

tau = 0
DC = 0
init = 1
tol = 1e-7
file = pd.read_excel('15朝阳广场进站.xlsx')
x = file.iloc[:, -1].values.reshape(-1, )


# 精英反向策略
def initial(pop, dim, ub, lb, fun):
    X = np.zeros([pop, dim])
    XAll = np.zeros([2 * pop, dim])
    for i in range(pop):
        for j in range(dim):
            XAll[i, j] = random.random() * (ub[j] - lb[j]) + lb[j]
            XAll[i + pop, j] = (ub[j] + lb[j]) - XAll[i, j]  # 求反向种群
            if XAll[i, j] > ub[j]:
                XAll[i, j] = ub[j]
            if XAll[i, j] < lb[j]:
                XAll[i, j] = lb[j]
            if XAll[i + pop, j] > ub[j]:
                XAll[i + pop, j] = ub[j]
            if XAll[i + pop, j] < lb[j]:
                XAll[i + pop, j] = lb[j]
        fitness = fun(XAll[i, :])
        fitnessBack = fun(XAll[i + pop, :])
        if fitnessBack < fitness:  # 反向解更好的给原始解
            XAll[i, :] = XAll[i + pop, :]

    X = XAll[0:pop, :]
    # 获取精英边界
    lbT = np.min(X, 0)
    ubT = np.max(X, 0)

    for i in range(X.shape[0]):
        X[i, :] = random.random() * (lbT + ubT) - X[i, :]
        for j in range(dim):
            if X[i, j] > ub[j]:
                X[i, j] = ub[j]
            if X[i, j] < lb[j]:
                X[i, j] = lb[j]
    return X, lb, ub


# 边界检查函数
def BorderCheck(X, ub, lb, pop, dim):
    for i in range(pop):
        for j in range(dim):
            if X[i, j] > ub[j]:
                X[i, j] = ub[j]
            elif X[i, j] < lb[j]:
                X[i, j] = lb[j]
    return X


# 计算适应度函数
def CaculateFitness(X, fun):
    pop = X.shape[0]
    fitness = np.zeros([pop, 1])
    for i in range(pop):
        fitness[i] = fun(X[i, :])
    return fitness


# 适应度排序
def SortFitness(Fit):
    fitness = np.sort(Fit, axis=0)
    index = np.argsort(Fit, axis=0)
    return fitness, index


# 根据适应度对位置进行排序
def SortPosition(X, index):
    Xnew = np.zeros(X.shape)
    for i in range(X.shape[0]):
        Xnew[i, :] = X[index[i], :]
    return Xnew


# 麻雀发现者更新
def PDUpdate(X, PDNumber, ST, Max_iter, dim, GbestPositon):
    X_new = copy.copy(X)
    R2 = random.random()
    for j in range(PDNumber):
        if R2 < ST:
            X_new[j, :] = X[j, :] * np.exp(-j / (random.random() * Max_iter))
        else:
            # 改进点： 蝴蝶优化改进
            X_new[j, :] = X[j, :] + np.random.randn() * (GbestPositon - X[j, :])
    return X_new


# 麻雀加入者更新
def JDUpdate(X, PDNumber, pop, dim):
    X_new = copy.copy(X)
    for j in range(PDNumber + 1, pop):
        if j > (pop - PDNumber) / 2 + PDNumber:
            X_new[j, :] = np.random.randn() * np.exp((X[-1, :] - X[j, :]) / j ** 2)
        else:
            # 产生-1，1的随机数
            A = np.ones([dim, 1])
            for a in range(dim):
                if (random.random() > 0.5):
                    A[a] = -1
        AA = np.dot(A, np.linalg.inv(np.dot(A.T, A)))
        X_new[j, :] = X[1, :] + np.abs(X[j, :] - X[1, :]) * AA.T

    return X_new


# 危险更新
def SDUpdate(X, pop, SDNumber, fitness, BestF):
    X_new = copy.copy(X)
    Temp = range(pop)
    RandIndex = random.sample(Temp, pop)
    SDchooseIndex = RandIndex[0:SDNumber]
    for j in range(SDNumber):
        if fitness[SDchooseIndex[j]] > BestF:
            X_new[SDchooseIndex[j], :] = X[0, :] + np.random.randn() * np.abs(X[SDchooseIndex[j], :] - X[1, :])
        elif fitness[SDchooseIndex[j]] == BestF:
            K = 2 * random.random() - 1
            X_new[SDchooseIndex[j], :] = X[SDchooseIndex[j], :] + K * (
                        np.abs(X[SDchooseIndex[j], :] - X[-1, :]) / (fitness[SDchooseIndex[j]] - fitness[-1] + 10E-8))
    return X_new


# 改进麻雀搜索算法
def MSSA(pop, dim, lb, ub, Max_iter, fun):
    wu = []
    ST = 0.8  # 预警值
    PD = 0.7  # 发现者的比列，剩下的是加入者
    SD = 0.2  # 意识到有危险麻雀的比重
    PDNumber = int(pop * PD)  # 发现者数量
    SDNumber = int(pop * SD)  # 意识到有危险麻雀数量
    X, lb, ub = initial(pop, dim, ub, lb, fun)  # 初始化种群
    fitness = CaculateFitness(X, fun)  # 计算适应度值
    fitness, sortIndex = SortFitness(fitness)  # 对适应度值排序
    X = SortPosition(X, sortIndex)  # 种群排序
    GbestScore = copy.copy(fitness[0])
    GbestPositon = np.zeros([1, dim])
    GbestPositon[0, :] = copy.copy(X[0, :])
    Curve = np.zeros([Max_iter, 1])
    for i in range(Max_iter):

        BestF = fitness[0]

        X = PDUpdate(X, PDNumber, ST, Max_iter, dim, GbestPositon)  # 发现者更新

        X = JDUpdate(X, PDNumber, pop, dim)  # 加入者更新

        X = SDUpdate(X, pop, SDNumber, fitness, BestF)  # 危险更新

        X = BorderCheck(X, ub, lb, pop, dim)  # 边界检测

        fitness = CaculateFitness(X, fun)  # 计算适应度值

        # 改进点： 自适应t分布逐维变异
        avgF = np.mean(fitness)
        for j in range(pop):
            if fitness[j] < avgF:
                Temp = X[j, :] * (1 + np.random.randn())
                for a in range(dim):
                    if (Temp[a] > ub[a]):
                        Temp[a] = ub[a]
                    if (Temp[a] < lb[a]):
                        Temp[a] = lb[a]
                fitnew = fun(Temp)
                if fitnew < fitness[j]:
                    X[j, :] = copy.copy(Temp)
                    fitness[j] = copy.copy(fitnew)
            else:
                a = 0.7
                Temp = np.zeros([1, dim])

                x0 = random.random()  # 初始点
                TentValue = x0
                if x0 < a:
                    TentValue = x0 / a
                if x0 >= a:
                    TentValue = (1 - x0) / (1 - a)
                x0 = TentValue
                Temp[0, :] = X[j, :] * (1 + TentValue)

                for a in range(dim):
                    if (Temp[0, a] > ub[a]):
                        Temp[0, a] = ub[a]
                    if (Temp[0, a] < lb[a]):
                        Temp[0, a] = lb[a]
                fitnew = fun(Temp[0, :])
                if fitnew < fitness[j]:
                    X[j, :] = copy.copy(Temp[0, :])
                    fitness[j] = copy.copy(fitnew)
        fitness, sortIndex = SortFitness(fitness)  # 对适应度值排序
        X = SortPosition(X, sortIndex)  # 种群排序
        if (fitness[0] <= GbestScore):  # 更新全局最优
            GbestScore = copy.copy(fitness[0])
            GbestPositon[0, :] = copy.copy(X[0, :])
        Curve[i] = GbestScore
        wu.append(GbestScore)
        wo = pd.DataFrame(wu)
        wo.to_csv('MSSA-VMD.csv')
    return GbestScore, GbestPositon, Curve


def fitness(X):
    K = int(X[0])
    alpha = X[1]
    u, u_hat, omega = VMD(x, alpha, tau, K, DC, init, tol)
    #  最小平均包络熵
    EP = []
    for i in range(K):
        H = np.abs(hilbert(u[i, :]))
        e1 = []
        for j in range(len(H)):
            p = H[j] / np.sum(H)
            e = -p * log(p, 10)
            e1.append(e)
        E = np.sum(e1)
        EP.append(E)
    O = min(EP)
    return O












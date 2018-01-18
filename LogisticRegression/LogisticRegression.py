#coding=utf-8

# @Author: yangenneng
# @Time: 2018-01-18 15:30
# @Abstract：非线性回归-逻辑回归算法

import numpy as np
import random

'''
# function:产生一些数据，用来做拟合
# numPoints：实例个数
# bias：随机生成y时的偏好
# variance：一组数据的方差
'''
def genData(numPoints,bias,variance):
    # 生成numPoints行2列的零矩阵  shape形状
    x = np.zeros(shape=(numPoints, 2))
    y = np.zeros(shape=(numPoints))
    # 循环numPoints次，及i=0 到 numPoints-1
    for i in range(0,numPoints):
        x[i][0] = 1
        x[i][1] = i
        # random.uniform(0,1)  从0-1之间随机产生一些数
        y[i] = (i+bias)+random.uniform(0, 1) * variance
    return x, y


'''
# 梯度下降算法
# x：自变量矩阵，每行表示一个实例
# y：实例的真实值
# theta：待求的参数
# alpha：学习率
# m：总共m个实例
# numIterations：重复更新的次数（重复更新直至收敛（小于设定的阈值））
'''
def gradientDescent(x,y,theta,alpha,m,numIterations):
    # 矩阵转置
    xTran = np.transpose(x)
    # 循环次数
    for i in range(0,numIterations):
        # 公式中的Z
        hypothesis = np.dot(x,theta)
        # loss:预测值-实际值
        loss = hypothesis-y
        # cost就是公式中J(Θ)，这里定义的是一个简单的函数
        cost = np.sum(loss ** 2) / (2 * m)
        # 每次更新的更新量，即更新法则
        gradient=np.dot(xTran,loss)/m
        # Θ
        theta = theta-alpha * gradient
        print ("Iteration %d | cost :%f" % (i, cost))
    return theta

# 参数数据
x,y = genData(100, 25, 10)
print "x:", x
print "y:", y

# 查看产生数据的行列
m,n = np.shape(x)
n_y = np.shape(y)
print("m:"+str(m)+" n:"+str(n)+" n_y:"+str(n_y))

# 求Θ
numIterations = 100000
alpha = 0.0005
theta = np.ones(n)
theta= gradientDescent(x, y, theta, alpha, m, numIterations)
print(theta)


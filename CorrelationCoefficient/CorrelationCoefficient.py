#coding=utf-8
# @Author: yangenneng
# @Time: 2018-01-19 14:49
# @Abstract：

import numpy as np
from astropy.units import Ybarn
import math

# 计算相关系数
def computeCorrelation(X,Y):
    xBar = np.mean(X)
    yBar = np.mean(Y)
    SSR = 0
    varX = 0   # 公式中分子部分
    varY = 0   # 公式中分母部分
    for i in range(0, len(X)):
        diffXXBar = X[i]-xBar
        diffYYBar = Y[i]-yBar
        SSR += (diffXXBar * diffYYBar)
        varX += diffXXBar**2
        varY += diffYYBar**2

    SST = math.sqrt(varX*varY)
    return SSR/SST

# 假设有多个自变量的相关系数
def polyfit(x, y, degree):
    # 定义字典
    result={}
    # polyfit 自动计算回归方程：b0、b1...等系数  degree为x的几次方的线性回归方程
    coeffs=np.polyfit(x,y,degree)
    # 转为list存入字典
    result['polynomial']=coeffs.tolist()
    # poly1d 返回预测值
    p=np.poly1d(coeffs)
    # 给定一个x的预测值为多少
    y_hat = p(x)
    # 均值
    ybar = np.sum(y)/len(y)
    ssreg = np.sum((y_hat-ybar)**2)
    sstot = np.sum((y-ybar)**2)
    result['determination'] = ssreg/sstot
    return result

testX = [1, 3, 8, 7, 9]
testY = [10, 12, 24, 21, 34]

print "相关系数r:",computeCorrelation(testX,testY)
print "简单线性回归r^2:",str(computeCorrelation(testX,testY)**2)

# 此处x为一维的，所以多元退化为一元，结果应该与一元一样
print "多元回归r^2:",polyfit(testX,testY,1)



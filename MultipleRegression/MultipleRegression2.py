#coding=utf-8
# @Author: yangenneng
# @Time: 2018-01-17 16:11
# @Abstract：多元线性回归(Multiple Regression)算法  含类别变量

from numpy import genfromtxt
import numpy as np
from sklearn import linear_model

datapath=r"D:\Python\PyCharm-WorkSpace\MachineLearningDemo\MultipleRegression\data\data2.csv"
#从文本文件中提取数据并转为numpy Array格式
deliveryData = genfromtxt(datapath,delimiter=',')

print "data"
# print deliveryData

# 读取自变量X1...x5
x= deliveryData[1:,1:-1]
# 读取因变量
y = deliveryData[1:,-1]

print "x:",x
print "y:",y

# 调用线性回归模型
lr = linear_model.LinearRegression()
# 装配数据
lr.fit(x, y)

print lr

print("coefficients:")
print lr.coef_

print("intercept:")
print lr.intercept_

#预测
xPredict = [90,2,0,0,1]
yPredict = lr.predict(xPredict)
print("predict:")
print yPredict




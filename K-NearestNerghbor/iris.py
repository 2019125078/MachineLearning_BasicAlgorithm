#coding=utf-8
# @Author: yangenneng
# @Time: 2018-01-10 14:53
# @Abstract：调用库实现KNN算法

# 导入包: from 包 import 模块
from sklearn import neighbors
from sklearn import datasets

# 调用KNN分类器
knn=neighbors.KNeighborsClassifier()
# 复制变量 load_iris()会返回一个数据库,在datasets的iris里
iris=datasets.load_iris()

print iris
# 模型建立,装配数据,传入特征值150*4的矩阵，传如一维列向量
knn.fit(iris.data,iris.target)
#进行预测，根据 萼片长度、宽度 花片长度、宽度
predictedLable=knn.predict([[0.1,0.2,0.3,0.4]])

print predictedLable

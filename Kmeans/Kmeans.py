#coding=utf-8
# @Author: yangenneng
# @Time: 2018-01-20 15:53
# @Abstract：Kmeans算法

import numpy as np

'''
# X:数据集
# k:分为多少类
# maxIt:最多迭代多少次
'''
def kmeans(X,k,maxIt):
    # 返回行、列  即数据集的形状
    numPoints,numDim=X.shape
    # 给数据集多加一列
    dataSet = np.zeros((numPoints,numDim+1))
    # 把前numDim列=X
    dataSet[:,:-1] = X
    # # 随机选取K个中心点
    # centroids=dataSet[np.random.randint(numPoints,size=k),:]

    # 故意选择中心点为前两个，对应上述例子
    centroids=dataSet[0:2,:]

    # 初始化中心点的最后一列
    centroids[:,-1]=range(1,k+1)

    iterations=0
    # 储存旧的中心点
    oldCentroids=None

    # 只要不停止（新、旧中心点不一致）或（循环了maxIt次，当前为iterations次）
    while not shouldStop(oldCentroids,centroids,iterations,maxIt):
        print "循环到第"+str(iterations+1)+"次:\n"
        print "数据集dataSet:\n",dataSet
        print "中心点centroids:\n",centroids

        # 不能直接赋值，否则会同步更新，所以用复制
        oldCentroids=np.copy(centroids)
        iterations += 1
        # 更新数据集所属标签
        updateLables(dataSet,centroids)
        # 更新中心点
        centroids=getCentroids(dataSet,k)

    return dataSet

# 是否停止
def shouldStop(oldCentroids,centroids,iterations,maxIt):
    if iterations>maxIt:
        return True
    return np.array_equal(oldCentroids,centroids)

def updateLables(dataSet,centroids):
    numPoints,numDim=dataSet.shape
    for i in range(0,numPoints):
        dataSet[i,-1]=getLableFormClosetCentroid(dataSet[i,:-1],centroids)
'''
#  dataSetRow 当前行
#  centroids  中心点
# 返回当前点和那个中心点最近，即标签
'''
def getLableFormClosetCentroid(dataSetRow,centroids):
    label=centroids[0,-1]
    # np.linalg.norm算两个向量的距离
    minDist=np.linalg.norm(dataSetRow-centroids[0,:-1])

    for i in range(1,centroids.shape[0]):
        # 不算最后一列 因为最后一列是标签
        dist=np.linalg.norm(dataSetRow-centroids[i,:-1])
        if dist<minDist:
            minDist=dist
            label=centroids[i,-1]

    print "minDist:",minDist
    return label

def getCentroids(dataSet,k):
    result=np.zeros((k,dataSet.shape[1]))
    for i in range(1,k+1):
        # 把所有分类为i的找出来
        oneCluster=dataSet[dataSet[:,-1]==i,:-1]
        result[i-1,:-1]=np.mean(oneCluster,axis=0)
        result[i-1, -1]=i

    return result

'''
# ..........................主程序...............................
'''
x1=np.array([1,1])
x2=np.array([2,1])
x3=np.array([4,3])
x4=np.array([5,4])
# 把四个点纵向的堆起来
testX=np.vstack((x1,x2,x3,x4))

result=kmeans(testX,2,10)
print "final result:",result

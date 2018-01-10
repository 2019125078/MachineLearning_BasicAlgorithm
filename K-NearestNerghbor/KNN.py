#coding=utf-8
# @Author: yangenneng
# @Time: 2018-01-10 15:17
# @Abstract：自己实现KNN算法

import csv
import random
import math
import operator

# 加载数据集
# 参数：
# filename    数据集.txt文件
# split       把一部分数据集作为训练数据集，用来训练产生模型；另一部分用来测试，看每一个实例预测与实际归类的比较,以split为分界线分为两部分
# trainingSet 数据集.txt分出来的训练数据集
# testSet     数据集.txt分出来的测试数据集
def loadDataSet(filename,split,trainingSet=[],testSet=[]):
    # 打开文件 装载为csvfile，即以逗号分隔
    with open(filename,'rb') as csvfile:
        # 读取所有行
        lines=csv.reader(csvfile)
        # 所有行转化为list形式
        dataset=list(lines)
        # 把数据集.txt文件分隔为两部分：如果此次产生的随机数小于split就加入训练集，否则加入测试集
        for x in range(len(dataset)-1):
            for y in range(4):
                dataset[x][y]=float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])


# 计算距离:Eulidean Distance(每个点可以多维度)
# 参数：
# instance1 实例1
# instance2 实例2
# length    实例维度
def eulideanDistance(instance1,instance2,length):
    distance=0
    # 每一维进行差运算
    for x in range(length):
        distance += pow((instance1[x]-instance2[x]),2)
    return math.sqrt(distance)


# 返回最近的K个临近点
# 参数：
# trainningSet 训练数据集
# testInstance 测试数据集的一个实例
# k            返回K个最近的点
def getNeighbors(trainningSet,testInstance,k):
    # 装所有的距离
    distances=[]
    # 测试实例的维度
    length=len(testInstance)-1
    # 训练集中的每一个数到测试集的距离
    for x in range(len(trainningSet)):
        dist=eulideanDistance(testInstance,trainningSet[x],length)
        distances.append((trainningSet[x],dist))
    #  距离排序
    distances.sort(key=operator.itemgetter(1))
    # 取前k个距离
    neighbors=[]
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

# 根据返回的临距根据少数服从多数的投票判定要预测的实例归为哪一类
# 参数:
# neighbors 测试集中最近的前k个距离
def getResponse(neighbors):
    classvotes={}
    # 看每一个邻距属于哪个分类
    for x in range(len(neighbors)):
        response=neighbors[x][-1];
        if response in classvotes:
            classvotes[response]+=1
        else:
            classvotes[response]=1
    # 把每一类投票个数从大到小排列
    sortVotes=sorted(classvotes.iteritems(),key=operator.itemgetter(1),reverse=True)
    # 返回第一个，即投票最多的类别
    return sortVotes[0][0]

# 判断预测出的所有值与它实际的值比较准确率是多少
# 参数：
# testSet       测试数据集
# predictions   测试数据集预测出的分类
def getAccuracy(testSet,predictions):
    correct=0
    # 和实际的分类比较，看预测正确的有几个，即看精确率如何
    for x in range(len(testSet)):
        # [-1]指最后一列的值，Python的特殊语法
        # 判断预测与实际是否正确
        if testSet[x][-1]==predictions[x]:
            correct+=1
    # 计算精确率 预测对的/总的 *100.0%
    return (correct/float(len(testSet)))*100.0

# 主函数
def main():
    # 两个空的训练集 测试集
    trainingSet=[]
    testSet=[]
    split=0.67  #取2/3的做训练集  取1/3做测试集

    # 加载数据 r表示后面的字符串不做特殊转化
    loadDataSet(r'D:\Python\PyCharm-WorkSpace\MachineLearningDemo\K-NearestNerghbor\data\irisdata.txt',split,trainingSet,testSet)
    print "trainingSet:"+repr(len(trainingSet))
    print "testSet:"+repr(len(testSet))

    # 存储预测来的类别得值
    predictions=[]
    k=3

    for x in range(len(testSet)):
        # 取最近3个邻距
        neighbors=getNeighbors(trainingSet , testSet[x] , k)
        # 归类得志
        result=getResponse(neighbors)
        # 加入归类
        predictions.append(result)
        print ('> predictions='+repr(result)+',actual='+repr(testSet[x][-1]))
    # 计算精确度
    accuracy = getAccuracy(testSet , predictions)
    print ('accuracy:'+repr(accuracy)+'%')


main()
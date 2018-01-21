#coding=utf-8
# @Author: yangenneng
# @Time: 2018-01-21 14:06
# @Abstract：Hierarchical clustering 层次聚类算法

from numpy import *

# 定义一个结点类
class cluster_node:
    # self：当前实例 <=> java中的this
    def __init__(self,vec,left=None,right=None,distance=0.0,id=None,count=1):
        self.left=left
        self.right=right
        self.vec=vec
        self.distance=distance
        self.id=id
        self.count=count

# 距离衡量方法1：两两之间绝对值
def L1dist(v1, v2):
    return sum(abs(v1 - v2))

# 距离衡量方法2：两两之间距离的平方
def L2dist(v1,v2):
    return sqrt(sum(v1-v2)**2)

# 建立好树形结构
# features ：numpyArray 矩阵
# distance ：距离衡量方法 默认L2dist()
def hcluster(features,distance=L2dist):
    distances={}
    currentclustid=-1
    # 每一个实例自称一类 cluster_node(array(features[i]),id=i)相当于调用构造函数
    clust=[cluster_node(array(features[i]),id=i) for i in range(len(features))]

    # 只要聚类的数量还大于1 => 继续
    while len(clust)>1:
        # 初始化两个最相近的距离，后面比较再更新
        lowestpair=(0,1)
        closet=distance(clust[0].vec,clust[1].vec)

        for i in range(len(clust)):
            for j in range(i+1,len(clust)):
                if(clust[i].id,clust[j].id) not in distances:
                    distances[(clust[i].id,clust[j].id)]=distance(clust[i].vec,clust[j].vec)

                d=distances[(clust[i].id,clust[j].id)]

                if d<closet:
                    closet = d
                    lowestpair=(i,j)
        # 两两中的平均距离
        mergevec=[(clust[lowestpair[0]].vec[i]+clust[lowestpair[1]].vec[i])/2.0 for i in range(len(clust[0].vec))]

        # 创造了一个新的结点
        newcluster=cluster_node(array(mergevec),left=clust[lowestpair[0]],
                                right=clust[lowestpair[1]],distance=closet,id=currentclustid)

        currentclustid-=1
        del clust[lowestpair[1]]
        del clust[lowestpair[0]]
        clust.append(newcluster)
    return clust[0]

# 取出建立好的树形结构
# clust：建立好的树形结构
# dist：阈值
def extract_clusters(clust,dist):
    clusters={}
    if clust.distance<dist:
        return [clust]
    else:
        cl=[]
        cr=[]
        if clust.left!=None:
            cl=extract_clusters(clust.left,dist=dist)
        if clust.right!=None:
            cr=extract_clusters(clust.right,dist=dist)
        return cl+cr

def get_cluster_elements(clust):
    if clust.id >= 0:
        return [clust.id]
    else:
        cl=[]
        cr=[]
        if clust.left!=None:
            cl=get_cluster_elements(clust.left)
        if clust.right!=None:
            cr=get_cluster_elements(clust.right)
        return cl+cr

def printclust(clust,labels=None,n=0):
    for i in range(n):print ' ',
    if clust.id<0:
        print '-'
    else:
        if labels == None:print clust.id
        else:print labels[clust.id]

    if clust.left!=None:printclust(clust.left,labels=labels,n=n+1)
    if clust.right!=None:printclust(clust.right,labels=labels,n=n+1)

# 求树的高度
def getHeight(clust):
    if clust.left==None and clust.right==None:return 1
    return max(getHeight(clust.left),getHeight(clust.right))

# 求树的深度
def getDepth(clust):
    if clust.left==None and clust.right==None:return 0
    return max(getDepth(clust.left),getDepth(clust.right))+clust.distance


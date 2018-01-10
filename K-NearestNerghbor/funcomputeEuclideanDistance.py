#coding=utf-8
# @Author: yangenneng
# @Time: 2018-01-10 14:19
# @Abstract：K-Nearest Nerghbor 临近取样算法

import math

# 定义计算距离的函数
def computeEuclideanDistance(x1,y1,x2,y2):
    d=math.sqrt(math.pow(x1-x2,2)+math.pow(y1-y2,2))
    return d

d_ag=computeEuclideanDistance(3,104,18,90) #AG的距离
print("d_ag:",d_ag)
d_bg=computeEuclideanDistance(2,100,18,90) #BG的距离
print("d_bg:",d_bg)
d_cg=computeEuclideanDistance(1,81,18,90) #CG的距离
print("d_cg:",d_cg)
d_dg=computeEuclideanDistance(101,10,18,90) #DG的距离
print("d_dg:",d_dg)
d_eg=computeEuclideanDistance(99,5,18,90) #EG的距离
print("d_eg:",d_eg)
d_fg=computeEuclideanDistance(98,2,18,90) #FG的距离
print("d_fg:",d_fg)


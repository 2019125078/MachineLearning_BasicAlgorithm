#coding=utf-8
# @Author: yangenneng
# @Time: 2018-01-11 15:39
# @Abstract：支持向量机 support vector machine

# 导入矩阵运算的数据包
import numpy as np
# 导入python画图的数据包
import pylab as pl
from sklearn import svm

# 创建40个点

# seed(0)每次运行时产生的点不变
np.random.seed(0)
# randn(20,2) 随机产生20个2维的点：通过正态分布-[2,2]（均值 方差都为2），靠下
X=np.r_[np.random.randn(20,2) - [2,2],np.random.randn(20,2)+[2,2]]
# 前20个点归为一类0，后20个点归为一类1
Y=[0]*20+[1]*20

clf=svm.SVC(kernel="linear")
# 装配数据
clf.fit(X,Y)

# 计算w0
w=clf.coef_[0]
# 计算直线斜率
a=-w[0]/w[1]
# 从(-5,5)产生一些连续的值eg:-5 -4 -3 -2 -1 0 1 2 3 4 5
xx=np.linspace(-5,5)
# 直线方程 (clf.intercept_[0])/w[1]截距
yy=a * xx -(clf.intercept_[0])/w[1]

# 和support相平行的两条线

# 取第一个支持向量带入计算下面那条线的截距
b=clf.support_vectors_[0]
yy_down=a*xx + (b[1]-a*b[0])
# 取最后一个支持向量带入计算上面那条线的截距
b=clf.support_vectors_[-1]
yy_up = a*xx+(b[1]-a*b[0])

print "w:",w
print "a:",a

print "clf.support_vectors_",clf.support_vectors_
print "clf.coef_",clf.coef_

# 画出三条线
pl.plot(xx,yy,'k+')
pl.plot(xx,yy_up,'k--')
pl.plot(xx,yy_down,'k--')

# 画出周围的点
pl.scatter(clf.support_vectors_[:,0],clf.support_vectors_[:,1],s=80,facecolors='none')
pl.scatter(X[:,0],X[:,1],c=Y,cmap=pl.cm.Paired)

pl.axis('tight')
pl.show()




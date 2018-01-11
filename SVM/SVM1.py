#coding=utf-8
# @Author: yangenneng
# @Time: 2018-01-11 15:17
# @Abstract：支持向量机 support vector machine

from sklearn import svm
# 定义三个点
X = [[2, 0], [1, 1], [2, 3]]
# 分类标记 用0，1代表两类问题的分类
y = [0, 0, 1]
# 分类器 kernel和函数 用的是线性的 SVC（）就是支持向量机
clf=svm.SVC(kernel='linear')
# X矩阵 每行一个实例 y 每个实例对应的class lable（分类标记）
clf.fit(X,y)

# 分类器
print ("'clf:'",clf)
# 哪几个点是支持向量 ('clf.support_vectors_:', array([[ 1.,  1.],[ 2.,  3.]]))  => [1, 1], [2, 3]是支持向量
print ("clf.support_vectors_:",clf.support_vectors_)
# 传入的点中下标为多少的是支持向量 ('clf.support_:', array([1, 2])) => 第二个、第三个是支持向量
print ("clf.support_:",clf.support_)
# 有多少个点是支持向量 ('clf.n_support_:', array([1, 1])) => 两类每类里找出了一个支持向量
print ("clf.n_support_:",clf.n_support_)

# 预测
print clf.predict([2, .0])

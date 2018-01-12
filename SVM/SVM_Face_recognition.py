#coding=utf-8
# @Author: yangenneng
# @Time: 2018-01-12 16:59
# @Abstract：SVM linear inseparable 人脸识别

#coding=utf-8
# @Author: yangenneng
# @Time: 2018-01-12 14:52
# @Abstract：SVM-linear inseparable-人脸识别

from __future__ import print_function
from time import time

import logging
# 绘图的包
import matplotlib.pyplot  as plt

from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC

# print(__doc__)
# 打印程序进展中的一些进展信息打印出来
logging.basicConfig(level=logging.INFO,format="%(asctime)s %(message)s")

# 数据集下载：fetch_lfw_people下载名人库Loader for the Labeled Faces in the Wild (LFW) people dataset
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# 返回数据集有多少个实例 h是多少 w是多少
n_samples, h, w = lfw_people.images.shape

# x矩阵用来装特征向量 得到数据集的所有实例
X = lfw_people.data
# 特征向量是多少维度的 [1]对应列数
n_features = X.shape[1]

# 每个实例对应的类别，即人的身份
y=lfw_people.target
# 返回所有的类别里人的名字
target_names=lfw_people.target_names
# 有多少类，即有多少个人要进行识别
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)


#####################拆分为训练集和测试集#############################
#X_train训练集的特征向量 X；test训练集的分类
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


#####################特征值降维度######################################
# 组成元素的数量
n_components=150
print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, X_train.shape[0]))
# 初始时间
t0 = time()
# 降维
pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)
print("done in %0.3fs" % (time() - t0))
# 提取特征量 eigenfaces从一张人脸上提取一些特征值
eigenfaces = pca.components_.reshape((n_components, h, w))

print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
# 把训练集特征向量转为更低维的矩阵
X_train_pca = pca.transform(X_train)
# 把训练集特征向量转为更低维的矩阵
X_test_pca = pca.transform(X_test)
print("done in %0.3fs" % (time() - t0))


#####################把降维过的特征向量结合SVM分类器进行分类######################################
print("Fitting the classifier to the training set")
t0 = time()
# 测试哪对 C和gamma 组合会产生最好的归类精确度  30中组合
# C:Penalty parameter C of the error term
# gamma:kernal function  多少的特征点被使用
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
# 调用SVM进行分类 搜索哪对组合会产生最好的归类精确度 kernel：rbf高斯径向基核函数   class_weight权重
clf = GridSearchCV(SVC(kernel='rbf', class_weight='auto'), param_grid)
# 专配数据 找出边际最大的超平面
clf = clf.fit(X_train_pca, y_train)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)


#####################进行评估准确率计算######################################
print("Predicting people's names on the test set")
t0 = time()
# 预测新的分类
y_pred = clf.predict(X_test_pca)
print("done in %0.3fs" % (time() - t0))
# classification_report真实的分类和预测的分类进行比较
print(classification_report(y_test, y_pred, target_names=target_names))
# 建立n*n的矩阵 横行和竖行分别代表真实的标记和预测出的标记的区别 对角线上数值越多表示准确率越高
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))

#####################打印图像######################################
def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""

    # 建立图作为背景
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

# 预测函数归类标签和实际归类标签打印
def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

# 预测出的人名
prediction_titles = [title(y_pred, y_test, target_names, i)
                     for i in range(y_pred.shape[0])]

# 测试集的特征向量矩阵和要预测的人名打印
plot_gallery(X_test, prediction_titles, h, w)

# 打印原图和预测的信息
eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)

plt.show()
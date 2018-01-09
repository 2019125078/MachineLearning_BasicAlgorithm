#coding=utf-8
# @Author: yangenneng
# @Time: 2018-01-09 16:17
# @Abstract：Decision_Tree.py   决策树应用

# DictVectorizer： 读取原始数据
import csv
# 导入机器学习包

# DictVectorizer： 将dict类型的list数据，转换成numpy array
from sklearn.feature_extraction import DictVectorizer
from sklearn import tree
from sklearn import preprocessing
from sklearn.externals.six import StringIO

# 主程序

#读取使用的数据文件
allElectroncsData=open(r'D:\Python\PyCharm-WorkSpace\MachineLearningDemo\Decision_Tree\data\buyComputer.csv','rb')
#csv中按行读取函数reader()
reader=csv.reader(allElectroncsData)
headers=reader.next() #读取第一行标题

# print (headers)

featureList=[]  # 装取特征值
lableList=[]    # 装取类别 buy:yes|no

#逐行读取
for row in reader:
    lableList.append(row[len(row)-1]) #每行最后一列的值加入lableList
    rowDict={} #字典 key为对并的属性名:eg：age  value为属性值：eg：youth
    for i in range(1,len(row)-1):
        rowDict[headers[i]]=row[i]

    featureList.append(rowDict)

# print featureList

#使用python提供的DictVectorizer()进行转化我们需要的特征值dummyx格式
vec=DictVectorizer()
dummyx=vec.fit_transform(featureList).toarray()

# print ("dummyx:"+str(dummyx))
# print (vec.get_feature_names())

# print ("lableList:",str(lableList))

#使用python提供的DictVectorizer()进行转化我们需要的分类dummyy格式
lb=preprocessing.LabelBinarizer()
dummyy=lb.fit_transform(lableList)

# print ("dummyy:"+str(dummyy))

# using decision tree for classfication

# 使用ID3算法，即用Information Gain来分类
clf=tree.DecisionTreeClassifier(criterion='entropy')
# 建模   参数：特征值矩阵 分类列矩阵
clf=clf.fit(dummyx,dummyy)

# print ("clf",str(clf))

# 产生dot文件
with open("allElectroncInfomationGainori.dot",'w') as f:
    # 原始的数据变为0 1了，再画决策树时要还原之前定义的feature，即feature_names=vec.get_feature_names()
    f=tree.export_graphviz(clf,feature_names=vec.get_feature_names(),out_file=f)

# 取第一行
oneRowX=dummyx[0,:]
print ("oneRowX:"+str(oneRowX))

#预测新数据
newRowx=oneRowX
newRowx[0]=1
newRowx[2]=0

print ("newRowx:"+str(newRowx))

predictedY=clf.predict(newRowx)
print  ("predictedY:"+str(predictedY))







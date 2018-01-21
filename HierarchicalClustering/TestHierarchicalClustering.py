#coding=utf-8
# @Author: yangenneng
# @Time: 2018-01-21 15:28
# @Abstract：

from PIL import Image,ImageDraw
from HierarchicalClustering import hcluster
from HierarchicalClustering import getHeight
from HierarchicalClustering import getDepth
import numpy as np
import os

def drawdendrogram(clust,imlist,jpeg='clusters.jpg'):
    h=getHeight(clust)*1000
    w=1200
    depth=getDepth(clust)

    scaling =float(w-150)/depth

    img=Image.new('RGB',(w,h),(255,255,255))
    draw=ImageDraw.Draw(img)

    draw.line((0,h/2,50,h/2),fill=(255,0,0))

    drawnode(draw,clust,50,int(h/2),scaling,imlist,img)
    img.save(jpeg)

def drawnode(draw,clust,x,y,scaling,imlist,img):
    if clust.id<0:
        h1=getHeight(clust.left)*200
        h2=getHeight(clust.right)*200
        top=y-(h1+h2)/2
        bottom=y+(h1+h2)/2

        ll=clust.distance*scaling

        draw.line((x,top+h1/2,x,bottom-h2/2),fill=(255,0,0))
        draw.line((x,top+h1/2,x+ll,top+h1/2),fill=(255,0,0))
        draw.line((x,bottom-h2/2,x+ll,bottom-h2/2),fill=(255,0,0))

        drawnode(draw,clust.left,x+ll,top+h1/2,scaling,imlist,img)
        drawnode(draw,clust.right,x+ll,bottom-h2/2,scaling,imlist,img)
    else:
        nodeim=Image.open(imlist[clust.id])
        nodeim.thumbnail((50,50))
        ns=nodeim.size
        print x,y-ns[1]//2
        print x+ns[0]
        print (img.paste(nodeim, (int(x), int(y - ns[1] // 2), int(x + ns[0]), int(y + ns[1] - ns[1] // 2))))
        # img.paste()

imlist=[]
folderPath=r'D:\Python\PyCharm-WorkSpace\MachineLearningDemo\HierarchicalClustering\data'
for filename in os.listdir(folderPath):
    if os.path.splitext(filename)[1]=='.jpg':
        imlist.append(os.path.join(folderPath,filename))

n=len(imlist)
print n

features=np.zeros((n,3))
for i in range(n):
    im=np.array(Image.open(imlist[i]))
    R=np.mean(im[:,:,0].flatten())
    G=np.mean(im[:,:,1].flatten())
    B=np.mean(im[:,:,2].flatten())
    features[i]=np.array([R,G,B])

tree=hcluster(features)
drawdendrogram(tree,imlist,jpeg='sunSet.jpg')



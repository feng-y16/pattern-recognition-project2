from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import sklearn
import math
from numpy import *
from matplotlib import pyplot as plt
from shuffle_fy import shuffledata
from draw import plotall 
from mpl_toolkits.mplot3d import Axes3D
#r=np.load("10genes_withoutnorm.npz")
r=np.load("10genes.npz")
train_data=r["arr_0"]
train_label=r["arr_1"]
test_data=r["arr_2"]
test_label=r["arr_3"]
label_num=r["arr_4"]
train_size=r["arr_5"]
test_size=r["arr_6"]
#######################################################################################################
index=range(0,10)#shuffle
train_total=shuffledata(train_data,train_label,train_size,10)
train_data=train_total[:,index]
train_label=train_total[:,10]
#######################################################################################################
times=20#改变隐层节点数核隐层层数，运行时间较长
nodesmax=15
layersmax=5
scores=np.zeros((layersmax,nodesmax))
for i in range(1,layersmax+1):
    for j in range(1,nodesmax+1):
        p=i*[j]
        p=tuple(p)
        #print(i)
        for k in range(0,times):
            score=0
            model = MLPClassifier(hidden_layer_sizes=p,activation='relu',max_iter=5000)
            X_train, X_test, y_train, y_test = train_test_split(train_data, train_label,test_size=0.1, random_state=k)
            model.fit(X_train, y_train)
            scores[i-1][j-1]=scores[i-1][j-1]+model.score(X_test,y_test)
        scores[i-1][j-1]=scores[i-1][j-1]/times
print(scores)
np.save("hw2_2_10genes_test.npy",scores)
#scores=np.load("hw2_2_10genes_test.npy")
score=np.mat(scores)
x = np.arange(1, nodesmax+1, 1)
y = np.arange(1, layersmax+1, 1)
fig = plt.figure()
ax = Axes3D(fig)
x, y = np.meshgrid(x, y)
ax.plot_surface(x, y, score, rstride=1, cstride=1, cmap=plt.cm.jet)
ax.set_zlabel('Average accuracy')
ax.set_ylabel('Layers')
ax.set_xlabel('Nodes')
plt.savefig('2-2.png')

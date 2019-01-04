import sys
sys.path.append("F:/api/libsvm/python")
from svm import *
from svmutil import *
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
times=20#改变高斯核的C，g
cmax=5
cmin=0.5
cnum=10
gmax=1
gmin=0.05
gnum=10
scores=np.zeros((cnum,gnum))
for i in range(1,cnum+1):
    for j in range(1,gnum+1):
        c=i*cmin
        g=j*gmin
        command='-q -t 2 -v 10 -c '+str(c)+' -g '+str(g)
        for k in range(0,times):
            acc = svm_train(train_label,train_data,command)
            scores[i-1][j-1]=scores[i-1][j-1]+acc/100
        scores[i-1][j-1]=scores[i-1][j-1]/times
print(scores)
np.save("hw2_1_10genes_test1.npy",scores)
#scores=np.load("hw2_1_10genes_test1.npy")
score=np.mat(scores)
x = cmin*np.arange(1, cnum+1, 1)
y = gmin*np.arange(1, gnum+1, 1)
fig = plt.figure()
ax = Axes3D(fig)
x, y = np.meshgrid(x, y)
ax.plot_surface(x, y, score, rstride=1, cstride=1, cmap=plt.cm.jet)
ax.set_zlabel('Average accuracy')
ax.set_ylabel('g')
ax.set_xlabel('C')
plt.savefig('1-2.png')
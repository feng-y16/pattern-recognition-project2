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
#r=np.load("2genes_withoutnorm.npz")
r=np.load("2genes.npz")
train_data=r["arr_0"]
train_label=r["arr_1"]
test_data=r["arr_2"]
test_label=r["arr_3"]
label_num=r["arr_4"]
train_size=r["arr_5"]
test_size=r["arr_6"]
#######################################################################################################
index=range(0,2)#shuffle
train_total=shuffledata(train_data,train_label,train_size,2)
train_data=train_total[:,index]
train_label=train_total[:,2]
#######################################################################################################
times=200#改变线性核的C
cmax=10
cmin=0.1
cnum=100
scores=np.zeros(cnum)
for i in range(1,cnum+1):
    c=i*cmin
    command='-q -t 0 -v 10 -c '+str(c)
    for k in range(0,times):
        acc = svm_train(train_label,train_data,command)
        scores[i-1]=scores[i-1]+acc/100
    scores[i-1]=scores[i-1]/times
print(scores)
np.save("hw2_1_2genes_test2.npy",scores)
#scores=np.load("hw2_1_2genes_test2.npy")
C = cmin*np.arange(1, cnum+1, 1)
plt.plot(C,scores)
plt.xlabel('C')
plt.ylabel('Average accuracy')
plt.savefig('1-3.png')
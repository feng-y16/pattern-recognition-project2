import sys
sys.path.append("F:/api/libsvm/python")
from svm import *
from svmutil import *
import numpy as np
import pandas as pd
import sklearn
import math
from numpy import *
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from shuffle_fy import shuffledata
from draw import plotall

r=np.load("2genes_withoutnorm.npz")
#r=np.load("2genes.npz")
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
times=5000
score=0
for i in range(0,times):
    #acc = svm_train(train_label,train_data,'-q -t 0 -v 10')#线性核函数
    acc = svm_train(train_label,train_data,'-q -t 2 -v 10')#高斯核函数
    score=score+acc/100
score=score/times
print("average:",end="")
print(score)
#Linear
#withoutnorm
#accuracy=0.9762534059945227
#withnorm
#accuracy=0.9753198910081488
#Gauss
#withoutnorm
#accuracy=0.9815449591280636
#withnorm
#accuracy=0.9782452316075467
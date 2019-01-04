import sys
sys.path.append("F:/api/libsvm/python")
from svm import *
from svmutil import *
import numpy as np
import pandas as pd
import sklearn
import math
from numpy import *
from matplotlib import pyplot as plt
from shuffle_fy import shuffledata
from draw import plotall

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
#model = svm_train(train_label,train_data,'-t 0')#线性核函数
model = svm_train(train_label,train_data, '-t 2')#高斯核函数
print("Training data predict:")
predict1=svm_predict(train_label,train_data,model)
print("Test data predict:")
predict2=svm_predict(test_label,test_data,model)
#Linear
#withnorm
#train accuracy=0.978202
#test accuracy=0.934066
#withoutnorm
#train accuracy=0.986376
#test accuracy=0.934066
#Gauss
#withnorm
#train accuracy=0.989101
#test accuracy=0.978022
#withoutnorm
#train accuracy=0.99455
##test accuracy=0.978022
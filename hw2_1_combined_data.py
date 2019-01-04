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

#r=np.load("2genes_withoutnorm.npz")
r=np.load("2genes.npz")
train_data=r["arr_0"]
train_label=r["arr_1"]
test_data=r["arr_2"]
test_label=r["arr_3"]
label_num=r["arr_7"]
train_size=r["arr_5"]
test_size=r["arr_6"]
#######################################################################################################
datatemp=np.zeros((train_size+test_size,2))#shuffle
labeltemp=np.zeros(train_size+test_size)
datatemp[range(0,train_size),:]=train_data
datatemp[range(train_size,train_size+test_size),:]=test_data
labeltemp[range(0,train_size)]=train_label
labeltemp[range(train_size,train_size+test_size)]=test_label
total=shuffledata(datatemp,labeltemp,train_size+test_size,2)
index=range(0,2)
data=total[:,index]
label=total[:,2]
#######################################################################################################
model = svm_train(label,data,'-t 0 -c 0.2')#线性核函数
#model = svm_train(train_label,train_data, '-t 2 -c 1 -g 0.05')#高斯核函数
print("Training data predict:")
predict=svm_predict(label,data,model)
#######################################################################################################
x_begin=min(data[:,0])#扫描分界面位置
x_end=max(data[:,0])
y_begin=min(data[:,1])
y_end=max(data[:,1])+1
pointnumber=1200
linex=np.linspace(x_begin,x_end,pointnumber)
liney=np.zeros(pointnumber)
linextest=np.ones(pointnumber)
tempdata=np.zeros((pointnumber,2))
tempdata[:,1]=np.linspace(y_begin,y_end,pointnumber)
def coutline():
    for i in range(0,pointnumber):
        tempdata[:,0]=linex[i]*linextest
        temp=svm_predict(np.zeros(pointnumber),tempdata,model,'-q')
        for j in range(1,pointnumber):
            if(temp[0][j]+temp[0][j-1]==1):
                break
        liney[i]=tempdata[j][1]
    return liney
coutline()
#######################################################################################################
#输出数据
np.savez("SVM_Linear",linex,liney)
#np.savez("SVM_Gauss",linex,liney)
plotall(data,label,np.array(predict[0]),None,None,None,train_size+test_size,0,False,linex,liney)
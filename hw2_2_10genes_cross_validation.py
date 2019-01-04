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
times=100
score=0
for i in range(0,times):
    model = MLPClassifier(hidden_layer_sizes=(10,10),activation='relu',max_iter=1000)
    X_train, X_test, y_train, y_test = train_test_split(train_data, train_label,test_size=0.1, random_state=i)
    model.fit(X_train, y_train)
    score=score+model.score(X_test,y_test)
    print(model.score(X_test,y_test))
print("average=",end="")
print(score/times)
#withoutnorm
#average=0.9756756756756757
#withnorm
#average=0.9805405405405405
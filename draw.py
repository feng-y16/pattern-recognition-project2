import numpy as np
import math
from numpy import *
from collections import OrderedDict
from matplotlib import pyplot as plt
def plotline(data,label,decide,size,feature):
    plt.scatter(x=data[:,0],y=data[:,1],s=10,c=label)
    plt.show()
    return 0
#######################################################################################################
def plotall(train_data,train_label,train_decide,test_data,test_label,test_decide,train_size,test_size,test_exist=False,linex=None,liney=None):
    colors = ['b','g','r','orange']#作分图
    labelpool=['E3_right','E3_wrong','E5_right','E5_wrong']
    train_decide=train_decide.astype(int)
    for i in range(0,train_size):
        if train_label[i]==0:
            plt.scatter(x=train_data[i,0],y=train_data[i,1],s=9,c=colors[train_decide[i]],marker='o',label=labelpool[train_decide[i]])
        else:
            plt.scatter(x=train_data[i,0],y=train_data[i,1],s=9,c=colors[train_decide[i]],marker='v',label=labelpool[3-train_decide[i]])
    if(test_exist):
        test_decide=test_decide.astype(int)
        for i in range(0,test_size):
            if test_label[i]==0:
                plt.scatter(x=test_data[i,0],y=test_data[i,1],s=9,c=colors[test_decide[i]],marker='o',label=labelpool[train_decide[i]])
            else:
                plt.scatter(x=test_data[i,0],y=test_data[i,1],s=9,c=colors[test_decide[i]],marker='v',label=labelpool[3-train_decide[i]])
    plt.plot(linex,liney,color="red")
    plt.xlabel('normalized_log(BUB1+1)')
    plt.ylabel('normalized_log(DNMT1+1)')
    handles, labels = plt.gca().get_legend_handles_labels()  
    by_label = OrderedDict(zip(labels, handles))  
    handle=[]
    keys=[]
    flag=[0,0,0,0]
    for j in range(0,4):
        for i in range(0,len(list(by_label.keys()))):
            if list(by_label.keys())[i]==labelpool[j]:
                handle.append(list(by_label.values())[i])
                flag[j]=1
    for i in range(0,4):
        if flag[i]==1:
            keys.append(labelpool[i])
    plt.legend(handle, keys,loc = 'upper left')
    #plt.title("Perceptron")
    plt.show()
    return 0
#######################################################################################################
if __name__ == '__main__':#作总图
    SVM_Linear=np.load("SVM_Linear.npz")
    SVM_Gauss=np.load("SVM_Gauss.npz")
    MLP=np.load("MLP.npz")
    r=np.load("2genes.npz")
    train_data=r["arr_0"]
    train_label=r["arr_1"]
    test_data=r["arr_2"]
    test_label=r["arr_3"]
    label_num=r["arr_7"]
    train_size=r["arr_5"]
    test_size=r["arr_6"]
    for i in range(0,train_size):
        if train_label[i]==0:
            plt.scatter(x=train_data[i,0],y=train_data[i,1],s=9,c='b',marker='o',label='E3')
        else:
            plt.scatter(x=train_data[i,0],y=train_data[i,1],s=9,c='g',marker='v',label='E5')
    for i in range(0,test_size):
        if test_label[i]==0:
            plt.scatter(x=test_data[i,0],y=test_data[i,1],s=9,c='b',marker='o',label='E3')
        else:
            plt.scatter(x=test_data[i,0],y=test_data[i,1],s=9,c='g',marker='v',label='E5')
    plt.plot(SVM_Linear["arr_0"],SVM_Linear["arr_1"],label='SVM_Linear')
    plt.plot(SVM_Gauss["arr_0"],SVM_Gauss["arr_1"],label='SVM_Gauss')
    plt.plot(MLP["arr_0"],MLP["arr_1"],label='MLP')
    handles, labels = plt.gca().get_legend_handles_labels()  
    by_label = OrderedDict(zip(labels, handles))  
    plt.legend(by_label.values(),by_label.keys())
    plt.xlabel('normalized_log(BUB1+1)')
    plt.ylabel('normalized_log(DNMT1+1)')
    plt.savefig("all.png")
    plt.show()
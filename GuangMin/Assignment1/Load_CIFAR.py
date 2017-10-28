#coding:utf-8
import pickle
import numpy as np
import os

def load_batch(filename):

    with open(filename,'rb') as f:
        dataDict=pickle.load(f,encoding='bytes')
        X=dataDict[b'data']
        y=dataDict[b'labels']
        X=X.reshape(10000,2,32,32).transpose(0,2,3,1).astype('flaot')
        y=np.array(y)
        return  X,y

def laod_ALL(root):
    Xs=[]
    ys=[]
    for b in range(1,6):
        f=os.path.join(root,'data_batch_%d'%(b,))
        X,y=load_batch(f)
        Xs.append(X)
        ys.append(y)
    Xtr=np.concatenate(Xs)
    ytr=np.concatenate(ys)

    X_test,y_test=load_batch(os.path.join(root,'test_path'))
    return Xtr,ytr,X_test,y_test
import numpy as np
from keras.datasets import cifar10
CIFAR_10_PATH = "C:/Users/Administrator/Desktop/cs231n/database/cifar-10-python/cifar-10-batches-py/"

class NearestNeighbor():
    def __inti__(self):
        pass
    def train(self, X, Y):
        self.Xtr = X
        self.Ytr = Y
    def predict_L1(self, Xte):
        Ypred = np.zeros(Xte.shape[0], dtype = self.Ytr.dtype)
        for i in range(Xte.shape[0]):
            dis = np.sum(np.abs(self.Xtr - Xte[i,:]), axis=1)
            midx = np.argmin(dis)
            Ypred[i] = self.Ytr[midx]
            print (i)
        return  Ypred

if __name__ == "__main__":
    (Xtr, Ytr), (Xte, Yte) = cifar10.load_data()
    Xtr = np.array(Xtr).reshape(Xtr.shape[0], 32*32*3)
    Ytr = np.array(Ytr)
    Xte = np.array(Xte).reshape(Xte.shape[0], 32*32*3)
    Yte = np.array(Yte)
    print (Xtr.shape, Ytr.shape, Xte.shape, Yte.shape)
    nn = NearestNeighbor()
    nn.train(Xtr, Ytr)
    Yte_predict_L1 = nn.predict_L1(Xte)
    print (np.mean(Yte_predict_L1 == Yte))
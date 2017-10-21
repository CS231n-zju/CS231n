import numpy as np
import pickle as p
CIFAR_10_PATH = "C:/Users/Administrator/Desktop/cs231n/database/cifar-10-python/cifar-10-batches-py/"
def load_CIFAR10(filename):
    for i in range(5):
        with open(filename + "data_batch_" + str(i+1), 'rb') as f:
            dict = p.load(f)
            data = np.array(dict['data'])
            labels = np.array(dict['labels'])
            if i==0:
                data_train = data
                labels_train = labels
            else:
                data_train = np.vstack((data_train, data))
                labels_train = np.hstack((labels_train, labels))

    with open(filename + "test_batch", 'rb') as f:
        dict = p.load(f)
        data_test = np.array(dict['data'])
        labels_test = np.array(dict['labels'])
    return data_train, labels_train, data_test, labels_test

class NearestNeighbor():
    def __inti__(self):
        pass
    def train(self, X, Y):
        self.Xtr = X
        self.Ytr = Y
    def predict_L1(self, Xte):
        Ypred = np.zeros(Xte.shape[0], dtype = self.Ytr.dtype)
        for i in xrange(Xte.shape[0]):
            dis = np.sum(np.abs(self.Xtr - Xte[i,:]), axis=1)
            midx = np.argmin(dis)
            Ypred[i] = self.Ytr[midx]
        return  Ypred

    def predict_L2(self, Xte):
        Ypred = np.zeros(Xte.shape[0], dtype = self.Ytr.dtype)
        for i in xrange(Xte.shape[0]):
            dis = np.sqrt(np.sum(np.square(self.Xtr - Xte[i,:]), axis=1))
            midx = np.argmin(dis)
            Ypred[i] = self.Ytr[midx]
            
            
            
        return  Ypred

if __name__ == "__main__":
    Xtr, Ytr, Xte, Yte = load_CIFAR10(CIFAR_10_PATH)
    print Xtr.shape, Ytr.shape, Xte.shape, Yte.shape
    nn = NearestNeighbor()
    nn.train(Xtr, Ytr)
    Yte_predict_L1 = nn.predict_L1(Xte)
    print 'NN accuracy L1: %f' % (np.mean(Yte_predict_L1 == Yte))
    Yte_predict_L2 = nn.predict_L2(Xte)
    print 'NN accuracy L2: %f' % (np.mean(Yte_predict_L2 == Yte))
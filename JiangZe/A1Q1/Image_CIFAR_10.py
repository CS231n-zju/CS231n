import pickle as p
import numpy as np
from PIL import Image

CIFAR_10_PATH = "C:/Users/Administrator/Desktop/cs231n/database/cifar-10-python/cifar-10-batches-py/"

def load_CIFAR_batch(filename):
    with open(filename, 'rb') as f:
        datadict = p.load(f)
        X = datadict['data']
        Y = datadict['labels']
        names = datadict['filenames']
        X = X.reshape(10000, 3, 32, 32)
        # arr = np.array(range(24))
        # print arr.reshape(2,3,4)
        Y = np.array(Y)
        return X, Y, names

def load_CIFAR_Labels(filename):
    with open(filename, 'rb') as f:
        datadict = p.load(f)
        Labels = datadict['label_names']
        return Labels

if __name__=="__main__":
    label_names = load_CIFAR_Labels(CIFAR_10_PATH + "batches.meta")
    for i in range(5):
        data, labels, data_names= load_CIFAR_batch(CIFAR_10_PATH + "data_batch_" + str(i+1))
        for i in xrange(data.shape[0]):
            imgs = data[i]
            img0 = imgs[0]
            img1 = imgs[1]
            img2 = imgs[2]
            i0 = Image.fromarray(img0)
            i1 = Image.fromarray(img1)
            i2 = Image.fromarray(img2)
            img = Image.merge("RGB", (i0, i1, i2))
            name = data_names[i]
            img.save("C:/Users/Administrator/Desktop/cs231n/database/cifar-10/" + name, "png")


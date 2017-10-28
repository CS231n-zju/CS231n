#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by wjbKimberly on 17-10-20

import numpy as np

#create normal array
a=np.array([1,2,3])
print(type(a))
print(a.shape)
print(a[0],a[1],a[2])
a[0]=5
print(a)

b=np.array([[1,2,3],[4,5,6]])
print(b.shape)
print(b)

#create special array
a=np.zeros((2,2))
print (a)

b=np.ones((1,2))
print(b)

#fill up the matrix with specific numbers
c=np.full((2,2),7)
print (c)

#identity matrix
d=np.eye(2)
print(d)

#index
a=np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print(a)
b=a[:2,1:3]
print(b)
print(a[[1,2,2],[0,1,0]])

#reshape
v = np.array([1,2,3])  # v has shape (3,)
w = np.array([4,5])    # w has shape (2,)
# To compute an outer product, we first reshape v to be a column
# vector of shape (3, 1); we can then broadcast it against w to yield
# an output of shape (3, 2), which is the outer product of v and w:
# [[ 4  5]
#  [ 8 10]
#  [12 15]]
print(np.reshape(v, (3, 1)) * w)


if __name__ == '__main__':
    pass

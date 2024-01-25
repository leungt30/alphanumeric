import numpy as np

a = np.array([25.67689262, -9.6024265 ])
b = np.array([5., 2.3,  0.1,  2.,  -1. ])


print(a)
print(a.shape)
a= a.reshape((-1,1))
print(a)
print("----------")
print(b)
print(b.shape)
b= b.reshape((1,-1))


c = a.dot(b)
print("Result:", c)
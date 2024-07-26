import numpy as np

X = np.array(   [[1,2], 
                [3,4],
                [5,6]])
#print(X)
print(X[1:])
print(X.shape[0])

rgen = np.random.RandomState(1)
w_ = rgen.normal(loc = 0.0, scale = 0.01, size = X.shape[0]) 

print(w_)
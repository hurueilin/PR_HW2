import numpy as np


np.random.seed(1)
N = 10
b1 = np.random.rand(N)
b2 = np.random.rand(N)
X = np.column_stack([b1, b2])

row, col = X.shape
print('shape of X:', X.shape)

X -= X.mean(axis=0) 
print(X.mean(axis=0))

# fact = N - 1 
fact = row - 1

by_hand = np.dot(X.T, X.conj()) / fact
print(by_hand)
# [[ 0.04735338  0.01242557]
#  [ 0.01242557  0.07669083]]

using_cov = np.cov(b1, b2)

print(using_cov)

# assert np.allclose(by_hand, using_cov)
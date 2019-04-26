import numpy as np
import math
import torch
import torch.nn as nn
from util import *
from scipy.sparse import csr_matrix
from scipy.stats import rankdata

from psutil import cpu_count

truth = np.array([[0, 0, 1, 1, 0], [1,0,0,0,0]])
pred = np.array([[1,2,3,4,2],[6,1,3,4,5]])
pred2 = np.array([[2,3,5,7,9],[4,2,5,6,7]])

print(calculate_lwlrap(truth, pred))
print(calculate_lwlrap(truth, pred2))

a = torch.Tensor([-14.1])
b = torch.sigmoid(a)
print(b)
'''
from sklearn.model_selection import KFold, StratifiedKFold
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([[0, 1, 0], [0, 2, 0], [0, 3, 0], [0, 4, 0]])
kf = KFold(n_splits=3)

kf.get_n_splits(X)

print(kf)

for train_index, test_index in kf.split(X):
   print("TRAIN:", train_index, "TEST:", test_index)
   X_train, X_test = X[train_index], X[test_index]
   y_train, y_test = y[train_index], y[test_index]
   print(X_train, X_test)
   print(y_train, y_test)


classes = get_classes_name()
print(classes)

m = nn.Sigmoid()
loss = nn.BCELoss(reduce=False, reduction='mean')
input = torch.randn([16, 8], requires_grad=True)
# print(input)
# print(m(input))
target = torch.empty([16, 8]).random_(2)
output = loss(m(input), target)
# output.backward()

print(input.shape)
print(target.shape)
print(output)
'''

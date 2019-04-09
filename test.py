import numpy as np
import math
import torch
import torch.nn as nn
from util import *
from scipy.sparse import csr_matrix
from scipy.stats import rankdata

def label_ranking_average_precision_score(y_true, y_score, sample_weight=None):
    """Compute ranking-based average precision
    Label ranking average precision (LRAP) is the average over each ground
    truth label assigned to each sample, of the ratio of true vs. total
    labels with lower score.
    This metric is used in multilabel ranking problem, where the goal
    is to give better rank to the labels associated to each sample.
    The obtained score is always strictly greater than 0 and
    the best value is 1.
    Read more in the :ref:`User Guide <label_ranking_average_precision>`.
    Parameters
    ----------
    y_true : array or sparse matrix, shape = [n_samples, n_labels]
        True binary labels in binary indicator format.
    y_score : array, shape = [n_samples, n_labels]
        Target scores, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by "decision_function" on some classifiers).
    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.
    Returns
    -------
    score : float
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import label_ranking_average_precision_score
    >>> y_true = np.array([[1, 0, 0], [0, 0, 1]])
    >>> y_score = np.array([[0.75, 0.5, 1], [1, 0.2, 0.1]])
    >>> label_ranking_average_precision_score(y_true, y_score) \
        # doctest: +ELLIPSIS
    0.416...
    """

    y_true = csr_matrix(y_true)
    y_score = -y_score
    n_samples, n_labels = y_true.shape
    out = 0.
    for i, (start, stop) in enumerate(zip(y_true.indptr, y_true.indptr[1:])):
        relevant = y_true.indices[start:stop]

        if (relevant.size == 0 or relevant.size == n_labels):
            # If all labels are relevant or unrelevant, the score is also
            # equal to 1. The label ranking has no meaning.
            out += 1.
            continue

        scores_i = y_score[i]
        rank = rankdata(scores_i, 'max')[relevant]
        L = rankdata(scores_i[relevant], 'max')
        aux = (L / rank).mean()
        if sample_weight is not None:
            aux = aux * sample_weight[i]
        out += aux

    if sample_weight is None:
        out /= n_samples
    else:
        print(out)
        out /= np.sum(sample_weight)

    return out


y_true = np.array([[1, 0, 0], [0, 0, 1]])
y_score = np.array([[0.75, 0.5, 1], [1, 0.2, 0.1]])
print(y_true)
print(y_score)
sw = [3, 2]

lrap = label_ranking_average_precision_score(y_true, y_score, sw)
print(lrap)
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

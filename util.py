import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import pandas as pd
import shutil
import pickle
import librosa
import logging
import os
from networks import *
from config import Config
import math
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import train_test_split
import sklearn.metrics
from tqdm import tqdm
import time
import itertools
import visdom


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def save_data(filename, data):
    """Save variable into a pickle file

    Parameters
    ----------
    filename: str
        Path to file

    data: list or dict
        Data to be saved.

    Returns
    -------
    nothing

    """
    pickle.dump(data, open(filename, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    # pickle.dump(data, open(filename, 'w'))


def load_data(filename):
    """Load data from pickle file

    Parameters
    ----------
    filename: str
        Path to file

    Returns
    -------
    data: list or dict
        Loaded file.

    """

    return pickle.load(open(filename, "rb"), encoding='latin1')


def create_logging(log_dir, filemode):

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    i1 = 0

    while os.path.isfile(os.path.join(log_dir, "{:04d}.log".format(i1))):
        i1 += 1

    log_path = os.path.join(log_dir, "{:04d}.log".format(i1))
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename=log_path,
                        filemode=filemode)

    # Print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    return logging


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def  __init__(self):
        self.reset()

    def  reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(state, is_best, fold, config, filename='../model/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        best_name = config.model_dir + '/model_best.' + str(fold) + '.pth.tar'
        shutil.copyfile(filename, best_name)


def run_method_by_string(name):
    p = globals().copy()
    p.update(globals())
    method = p.get(name)
    if not method:
        raise NotImplementedError('Method %s not implement' % name)
    return method


def make_dirs():
    dirs = ['../input/features/wave-44100', '../prediction', '../log', '../model',
            '../model', '../submission', '../input/features/logmel+delta_w80_s10_m64',
            '../input/features/mfcc+delta_w80_s10_m64']

    for dir in dirs:
        if not os.path.exists(dir):
            os.mkdir(dir)
            print('Make dir: %s' %dir)


def make_one_hot(target, num_class=41):
    """
    convert index tensor into one-hot tensor
    """
    assert isinstance(target, torch.LongTensor)
    return torch.zeros(target.size()[0], num_class).scatter_(1, target.view(-1, 1), 1)


def multilabel_to_onehot(labels, label_idx, num_class=80):
    """
    :param labels: multi-label separated by comma.
    :param num_class: number of classes, length of one-hot label.
    :return: one-hot label, such as [0, 1, 0, 0, 1,...]
    """
    # one_hot = np.zeros(num_class)
    one_hot = torch.zeros(num_class)
    for l in labels.split(','):
        one_hot[label_idx[l]] = 1.0
    return one_hot


def cross_entropy_onehot(input, target, size_average=True):
    """
    Cross entropy  that accepts soft targets (like [0, 0.1, 0.1, 0.8, 0]).
    """
    assert input.size() == target.size()

    if size_average:
        return torch.mean(torch.sum(-target * F.log_softmax(input, dim=1), dim=1))
    else:
        return torch.sum(torch.sum(-target * F.log_softmax(input, dim=1), dim=1))


def get_classes_name():
    file = pd.read_csv('../input/sample_submission.csv')
    return(list(file.columns)[1:])


# FROM: https://colab.research.google.com/drive/1AgPdhSp7ttY18O3fEoHOQKlt_3HJDLi8#scrollTo=FJv0Rtqfsu3X
# Calculate the overall lwlrap using sklearn.metrics function.
# def calculate_lwlrap(truth, scores):
#     """Calculate the overall lwlrap using sklearn.metrics.lrap."""
#     # sklearn doesn't correctly apply weighting to samples with no labels, so just skip them.
#     sample_weight = np.sum(truth > 0, axis=1)
#     nonzero_weight_sample_indices = np.flatnonzero(sample_weight > 0)
#     overall_lwlrap = sklearn.metrics.label_ranking_average_precision_score(
#         truth[nonzero_weight_sample_indices, :] > 0,
#         scores[nonzero_weight_sample_indices, :],
#         sample_weight=sample_weight[nonzero_weight_sample_indices])
#     return overall_lwlrap


# Core calculation of label precisions for one test sample.

def _one_sample_positive_class_precisions(scores, truth):
    """Calculate precisions for each true class for a single sample.

    Args:
      scores: np.array of (num_classes,) giving the individual classifier scores.
      truth: np.array of (num_classes,) bools indicating which classes are true.

    Returns:
      pos_class_indices: np.array of indices of the true classes for this sample.
      pos_class_precisions: np.array of precisions corresponding to each of those
        classes.
    """
    num_classes = scores.shape[0]
    pos_class_indices = np.flatnonzero(truth > 0)
    # Only calculate precisions if there are some true classes.
    if not len(pos_class_indices):
        return pos_class_indices, np.zeros(0)
    # Retrieval list of classes for this sample.
    retrieved_classes = np.argsort(scores)[::-1]
    # class_rankings[top_scoring_class_index] == 0 etc.
    class_rankings = np.zeros(num_classes, dtype=np.int)
    class_rankings[retrieved_classes] = range(num_classes)
    # Which of these is a true label?
    retrieved_class_true = np.zeros(num_classes, dtype=np.bool)
    retrieved_class_true[class_rankings[pos_class_indices]] = True
    # Num hits for every truncated retrieval list.
    retrieved_cumulative_hits = np.cumsum(retrieved_class_true)
    # Precision of retrieval list truncated at each hit, in order of pos_labels.
    precision_at_hits = (
            retrieved_cumulative_hits[class_rankings[pos_class_indices]] /
            (1 + class_rankings[pos_class_indices].astype(np.float)))
    return pos_class_indices, precision_at_hits


# All-in-one calculation of per-class lwlrap.

def calculate_per_class_lwlrap(truth, scores):
    """Calculate label-weighted label-ranking average precision.

    Arguments:
      truth: np.array of (num_samples, num_classes) giving boolean ground-truth
        of presence of that class in that sample.
      scores: np.array of (num_samples, num_classes) giving the classifier-under-
        test's real-valued score for each class for each sample.

    Returns:
      per_class_lwlrap: np.array of (num_classes,) giving the lwlrap for each
        class.
      weight_per_class: np.array of (num_classes,) giving the prior of each
        class within the truth labels.  Then the overall unbalanced lwlrap is
        simply np.sum(per_class_lwlrap * weight_per_class)
    """
    assert truth.shape == scores.shape
    num_samples, num_classes = scores.shape
    # Space to store a distinct precision value for each class on each sample.
    # Only the classes that are true for each sample will be filled in.
    precisions_for_samples_by_classes = np.zeros((num_samples, num_classes))
    for sample_num in range(num_samples):
        pos_class_indices, precision_at_hits = (
            _one_sample_positive_class_precisions(scores[sample_num, :],
                                                  truth[sample_num, :]))
        precisions_for_samples_by_classes[sample_num, pos_class_indices] = (
            precision_at_hits)
    labels_per_class = np.sum(truth > 0, axis=0)
    weight_per_class = labels_per_class / float(np.sum(labels_per_class))
    # Form average of each column, i.e. all the precisions assigned to labels in
    # a particular class.
    per_class_lwlrap = (np.sum(precisions_for_samples_by_classes, axis=0) /
                        np.maximum(1, labels_per_class))
    # overall_lwlrap = simple average of all the actual per-class, per-sample precisions
    #                = np.sum(precisions_for_samples_by_classes) / np.sum(precisions_for_samples_by_classes > 0)
    #           also = weighted mean of per-class lwlraps, weighted by class label prior across samples
    #                = np.sum(per_class_lwlrap * weight_per_class)
    return per_class_lwlrap, weight_per_class


def calculate_lwlrap(truth, scores):
    per_class_lwlrap, weight_per_class = calculate_per_class_lwlrap(truth, scores)
    return np.sum(per_class_lwlrap * weight_per_class)

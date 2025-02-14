import numpy as np
from torch.utils.data import Dataset
import torch
import scipy


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """_summary_: Updates the average meter with the new value and
        the number of samples
        Args:
            val (_type_): value
            n (int, optional):  Defaults to 1.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """_summary_

    Args:
        output (tensor): output of the model
        target (_type_): target
        topk (tuple, optional): topk. Defaults to (1,).

    Returns:
        float: accuracy
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class ExpertDatasetTensor(Dataset):
    """Generic dataset with expert predictions and labels and images"""

    def __init__(self, images, targets, exp_preds):
        self.images = images
        self.targets = np.array(targets)
        self.exp_preds = np.array(exp_preds)

    def __getitem__(self, index):
        """Take the index of item and returns the image, label,
        expert prediction and index in original dataset"""
        label = self.targets[index]
        image = self.images[index]
        expert_pred = self.exp_preds[index]
        return torch.FloatTensor(image), label, expert_pred

    def __len__(self):
        return len(self.targets)


def argmax_constrained(array1, array2, tol):
    if len(array2.shape) == 1:
        indices = np.where(np.abs(array2) < tol)
    else:
        idxs = []
        for i in range(array2.shape[1]):
            idxs.append(np.where(np.abs(array2[:, i]) < tol))
        indices = np.intersect1d(*idxs)
    if isinstance(indices, tuple):
        indices = indices[0]
        if len(indices) == 0:
            return None
    else:
        if indices.shape[0] == 0:
            return None
    max_index = np.argmax(array1[indices])
    max_index = indices[max_index]


def pareto(X, Y):

    convex_hull = scipy.spatial.ConvexHull(np.array([X, Y]).T)
    convex_hull_vertices = convex_hull.vertices
    X = X[convex_hull_vertices]
    Y = Y[convex_hull_vertices]
    pareto_X = []
    pareto_Y = []
    for i in range(len(X)):
        is_pareto = True
        for j in range(len(X)):
            if X[j] > X[i] and Y[j] < Y[i]:
                is_pareto = False
                break
        if is_pareto:
            pareto_X.append(X[i])
            pareto_Y.append(Y[i])
    # sort the pareto front
    pareto_X, pareto_Y = zip(*sorted(zip(pareto_X, pareto_Y)))
    # append (0.6, x[0]) for the first (x[0], y[0]) point
    new_point = (0.6, pareto_Y[0])
    new_point_2 = (pareto_X[-1], 0.37)
    pareto_X = [new_point[0]] + list(pareto_X) + [new_point_2[0]]
    pareto_Y = [new_point[1]] + list(pareto_Y) + [new_point_2[1]]
    return pareto_X, pareto_Y

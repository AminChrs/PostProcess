from abc import ABC, abstractmethod
from types import SimpleNamespace
import numpy as np


class BaseDataset(ABC):
    """Abstract method for learning to defer methods"""

    @abstractmethod
    def __init__(self, *args, **kwargs):
        """must at least have data_dir, test_split, val_split, batch_size,
        transforms"""
        pass

    @abstractmethod
    def generate_data(self):
        """generates the data loader, called on init

        should generate the following must:
            self.data_train_loader
            self.data_val_loader
            self.data_test_loader
            self.d (dimension)
            self.n_dataset (number of classes in target)
        """
        pass


class dataset:
    def __init__(self):
        self.X, self.y, self.s, self.MY, self.M, self.L =\
            {}, {}, {}, {}, {}, {}

    def finalize(self):
        self.process_score_labels()
        self.calculate_ps()

    def process_score_labels(self, val=True):

        # n_s = 1 + 1
        if val:
            all_sets = ['train', 'test', 'validation']
        else:
            all_sets = ['train', 'test']
        for data_type in all_sets:
            self.L[data_type] = 2 * self.M[data_type] + self.y[data_type]

    def calculate_ps(self):
        length = self.s['train'].shape[0]
        pa0 = np.sum(self.s['train'] == 0)/length
        pa1 = np.sum(self.s['train'] != 0)/length
        pa1y1 = np.sum((self.s['train'] != 0)*(self.y['train'] == 1))/length
        pa1y0 = np.sum((self.s['train'] != 0)*(self.y['train'] == 0))/length
        pa0y1 = np.sum((self.s['train'] == 0)*(self.y['train'] == 1))/length
        pa0y0 = np.sum((self.s['train'] == 0)*(self.y['train'] == 0))/length
        self.ps = SimpleNamespace(pa0=pa0,
                                  pa1=pa1,
                                  pa1y1=pa1y1,
                                  pa1y0=pa1y0,
                                  pa0y1=pa0y1,
                                  pa0y0=pa0y0)

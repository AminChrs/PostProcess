from abc import ABC, abstractmethod
import copy
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import random
import shutil
import time
import torch.utils.data as data
import sys
import pickle
import logging
from tqdm import tqdm
sys.path.append("..")
from helpers.utils import *
from helpers.metrics import *


class BaseMethod(ABC):
    """Abstract method for learning to defer methods"""

    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def fit(self, *args, **kwargs):
        """this function should fit the model and be enough to evaluate the model"""
        pass

    def fit_hyperparam(self, *args, **kwargs):
        """This is an optional method that fits and optimizes hyperparameters over a validation set"""
        return self.fit(*args, **kwargs)

    @abstractmethod
    def test(self, dataloader):
        """this function should return a dict with the following keys:
        'defers': deferred binary predictions
        'preds':  classifier predictions
        'labels': labels
        'hum_preds': human predictions
        'rej_score': a real score for the rejector, the higher the more likely to be rejected
        'class_probs': probability of the classifier for each class (can be scores as well)
        """
        pass


class BaseSurrogateMethod(BaseMethod):
    """Abstract method for learning to defer methods based on a surrogate model"""

    def __init__(self, alpha, plotting_interval, model, device, learnable_threshold_rej = False):
        '''
        alpha: hyperparameter for surrogate loss 
        plotting_interval (int): used for plotting model training in fit_epoch
        model (pytorch model): model used for surrogate
        device: cuda device or cpu
        learnable_threshold_rej (bool): whether to learn a treshold on the reject score (applicable to RealizableSurrogate only)
        '''
        self.alpha = alpha
        self.plotting_interval = plotting_interval
        self.model = model
        self.device = device
        self.threshold_rej = 0
        self.learnable_threshold_rej = learnable_threshold_rej 
        self.threshs = torch.zeros(2, 3).to(self.device)

    @abstractmethod
    def surrogate_loss_function(self, outputs, hum_preds, data_y):
        """surrogate loss function"""
        pass

    def fit_epoch(self, dataloader, optimizer, verbose=False, epoch=1):
        """
        Fit the model for one epoch
        model: model to be trained
        dataloader: dataloader
        optimizer: optimizer
        verbose: print loss
        epoch: epoch number
        """
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        end = time.time()
        self.model.train()
        for batch, (data_x, data_y, hum_preds, _) in enumerate(dataloader):
            data_x = data_x.to(self.device)
            data_y = data_y.to(self.device)
            hum_preds = hum_preds.to(self.device)
            outputs = self.model(data_x)
            loss = self.surrogate_loss_function(outputs, hum_preds, data_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            prec1 = accuracy(outputs.data, data_y, topk=(1,))[0]
            losses.update(loss.data.item(), data_x.size(0))
            top1.update(prec1.item(), data_x.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if torch.isnan(loss):
                print("Nan loss")
                logging.warning(f"NAN LOSS")
                break
            if verbose and batch % self.plotting_interval == 0:
                logging.info(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Prec@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                        epoch,
                        batch,
                        len(dataloader),
                        batch_time=batch_time,
                        loss=losses,
                        top1=top1,
                    )
                )

    def fit(
        self,
        dataloader_train,
        dataloader_val,
        dataloader_test,
        epochs,
        optimizer,
        lr,
        scheduler=None,
        verbose=True,
        test_interval=5,
    ):
        optimizer = optimizer(self.model.parameters(), lr=lr)
        if scheduler is not None:
            scheduler = scheduler(optimizer, len(dataloader_train) * epochs)
        best_acc = 0
        # store current model dict
        best_model = copy.deepcopy(self.model.state_dict())
        for epoch in tqdm(range(epochs)):
            self.fit_epoch(dataloader_train, optimizer, verbose, epoch)
            if epoch % test_interval == 0 and epoch > 1 :
                if self.learnable_threshold_rej:
                    self.fit_treshold_rej(dataloader_val)
                data_test = self.test(dataloader_val)
                val_metrics = compute_deferral_metrics(data_test)
                if val_metrics["system_acc"] >= best_acc:
                    best_acc = val_metrics["system_acc"]
                    best_model = copy.deepcopy(self.model.state_dict())
                if verbose:
                    logging.info(compute_deferral_metrics(data_test))
            if scheduler is not None:
                scheduler.step()
        self.model.load_state_dict(best_model)
        if self.learnable_threshold_rej:
            self.fit_treshold_rej(dataloader_val)
        final_test = self.test(dataloader_test)
        return compute_deferral_metrics(final_test)

    def fit_treshold_rej(self, dataloader):
        data_test = self.test(dataloader)
        rej_scores = np.unique(data_test["rej_score"])
        # sort by rejection score
        # get the 100 quantiles for rejection scores
        rej_scores_quantiles = np.quantile(rej_scores, np.linspace(0, 1, 100))
        # for each quantile, get the coverage and accuracy by getting a new deferral decision
        all_metrics = []
        best_treshold = 0
        best_accuracy = 0
        for q in rej_scores_quantiles:
            # get deferral decision
            defers = (data_test["rej_score"] > q).astype(int)
            copy_data = copy.deepcopy(data_test)
            copy_data["defers"] = defers
            # compute metrics
            metrics = compute_deferral_metrics(copy_data)
            if metrics['system_acc'] > best_accuracy:
                best_accuracy = metrics['system_acc']
                best_treshold = q
        self.threshold_rej = best_treshold
        
    
    def test(self, dataloader):
        """
        Test the model
        dataloader: dataloader
        """
        threshs = self.threshs
        defers_all = []
        truths_all = []
        hum_preds_all = []
        predictions_all = []  # classifier only
        rej_score_all = []  # rejector probability
        class_probs_all = []  # classifier probability
        sensitive_all = []  # sensitive params
        self.model.eval()
        with torch.no_grad():
            for batch, (data_x, data_y, hum_preds, s) in enumerate(dataloader):
                data_x = data_x.to(self.device)
                data_y = data_y.to(self.device)
                hum_preds = hum_preds.to(self.device)

                outputs = self.model(data_x)
                # add the first threshold if s==1, and if s==2 add the second threshold
                for i in range(len(data_x)):
                    if s[i] == 1:
                        outputs[i, :] += threshs[0, :]
                    elif s[i] == 2:
                        outputs[i, :] += threshs[1, :]

                outputs_class = F.softmax(outputs[:, :-1], dim=1)
                outputs = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                max_probs, predicted_class = torch.max(outputs.data[:, :-1], 1)
                predictions_all.extend(predicted_class.cpu().numpy())
     
                defer_scores = [(outputs.data[i][-1].item()
                                 - outputs.data[i][predicted_class[i]].item())
                                * (1-outputs.data[i][-1].item())
                                for i in range(len(outputs.data))]
                defer_binary = [int(defer_score >= self.threshold_rej) for defer_score in defer_scores]
                defers_all.extend(defer_binary)
                truths_all.extend(data_y.cpu().numpy())
                hum_preds_all.extend(hum_preds.cpu().numpy())
                for i in range(len(outputs.data)):
                    rej_score_all.append(
                        outputs.data[i][-1].item()
                        - outputs.data[i][predicted_class[i]].item()
                    )
                class_probs_all.extend(outputs_class.cpu().numpy())
                sensitive_all.extend(s.cpu().numpy())

        # convert to numpy
        defers_all = np.array(defers_all)
        truths_all = np.array(truths_all)
        hum_preds_all = np.array(hum_preds_all)
        predictions_all = np.array(predictions_all)
        rej_score_all = np.array(rej_score_all)
        class_probs_all = np.array(class_probs_all)
        sensitive_all = np.array(sensitive_all)
        data = {
            "defers": defers_all,
            "labels": truths_all,
            "hum_preds": hum_preds_all,
            "preds": predictions_all,
            "rej_score": rej_score_all,
            "class_probs": class_probs_all,
            "sensitive_params": sensitive_all,
        }
        return data
        

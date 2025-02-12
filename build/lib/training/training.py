from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from helpers.bootstrap import bootstrap_func
from helpers.embedding import process_score_labels
from helpers.embedding import estimate_witnesses, true_witness
from helpers.embedding import witness, witness_loss


def train_scores(X_train, y_train, MY_train, L_train, constraint='eo'):
    # witness
    if Path("rfs.npz").exists():
        rfs = joblib.load("rfs.npz")
        # rfs = [rfs[0], rfs[1], rfs[2]]
        return rfs
    else:
        # save random forests
        rf_clf_witness = RandomForestClassifier(n_jobs=-2)
        rf_clf_witness.fit(X_train, L_train)

        # classifier
        rf_clf = RandomForestClassifier(n_jobs=-2)
        rf_clf.fit(X_train, y_train)
        # rejector
        rf_my = RandomForestClassifier(n_jobs=-2)
        rf_my.fit(X_train, MY_train)
        rfs = [rf_clf, rf_my, rf_clf_witness]
        joblib.dump(rfs, "rfs.npz")

        return rf_clf, rf_my, rf_clf_witness
    

def find_thresholds(wits, true_wits, X_val, y_val,
                    MY_val, L_val, X_test, y_test, MY_test,
                    s_val, s_test,
                    L_test, threshold_space, tolerance_space,):

    loss, w = wits(X_val, s_val)
    loss_test, w_test = wits(X_test, s_test)
    # turn y_val into a one-hot vector
    loss_real, w_real = true_wits(L_test, s_test, MY_test, y_test)
    loss_real_val, w_real_val = true_wits(L_val, s_val, MY_val, y_val)

    losses_val = []
    ws_val = []
    tt = 0
    for k in threshold_space:
        tt += 1
        print("Threshold:", tt)
        # go over val_data that is a pandas dataframe
        # if k is int, then I need to tile it
        if isinstance(k, float):
            lagrang = loss-k*w
            # print("lagrang:", lagrang)
        else:
            w_sum = k[0]*w[:, 0]
            for i in range(len(k)):
                w_sum += k[i]*w[:, i]
            lagrang = loss-w_sum

        argmax_val = np.argmax(lagrang, axis=1)
        # Here, empirically I find the loss and witness on true labels
        max_loss_val = loss_real_val[np.arange(len(loss_real_val)), argmax_val]
        if len(w_real_val.shape) == 2:
            max_w_val = w_real_val[np.arange(len(w_real_val)), argmax_val]
        elif len(w_real_val.shape) == 3:
            max_w_val = w_real_val[np.arange(len(w_real_val)), :, argmax_val]
        max_loss_val = np.average(max_loss_val, axis=0)
        max_w_val = np.average(max_w_val, axis=0)
        losses_val.append(max_loss_val)
        ws_val.append(max_w_val)
    # now order in terms of w and find corresponding loss and thresholds
    losses_val = np.array(losses_val)
    ws_val = np.array(ws_val)
    loss_out = np.zeros([len(tolerance_space)])
    if len(ws_val.shape) == 1:
        threshold_out = np.zeros([len(tolerance_space)])
        ws_out = np.zeros([len(tolerance_space)])
        ws_std_out = np.zeros([len(tolerance_space)])
    else:
        threshold_out = np.zeros([len(tolerance_space),
                                  len(threshold_space[0])])
        ws_out = np.zeros([len(tolerance_space), len(ws_val[0])])
        ws_std_out = np.zeros([len(tolerance_space), len(ws_val[0])])
    loss_std_out = np.zeros([len(tolerance_space)])
    j = 0
    for tol in tolerance_space:
        print("tolerance:", tol)
        # find the minimum loss for all the ones with ws < tol
        if len(ws_val.shape) == 1:
            indices = np.where(np.abs(ws_val) < np.abs(tol))
        else:
            idxs = []
            for i in range(ws_val.shape[1]):
                idxs.append(np.where(np.abs(ws_val[:, i]) < np.abs(tol)))
            print(idxs)      
            indices = np.intersect1d(*idxs)
        if isinstance(indices, tuple):
            indices = indices[0]
            if len(indices) == 0:
                j += 1
                continue
        else:
            if indices.shape[0] == 0:
                j += 1
                continue
        max_index = np.argmax(losses_val[indices])
        print("max_index:", max_index)
        print("indices:", indices)
        max_index = indices[max_index]
        # find corresponding k
        k_val = threshold_space[max_index]
        # Now find the  argmax_test

        def loss_and_w(loss_est, w_est, lreal, wreal):
            if isinstance(k_val, float):
                lagrang_test = loss_est-k_val*w_est
            else:
                w_sum = k_val[0]*w_est[:, 0]
                for i in range(len(k_val)):
                    w_sum += k_val[i]*w_est[:, i]
                lagrang_test = loss_est-w_sum
            argmax_test = np.argmax(lagrang_test, axis=1)
            max_loss_test = lreal[np.arange(len(lreal)), argmax_test]
            if len(wreal.shape) == 2:
                max_w_test = wreal[np.arange(len(wreal)), argmax_test]
            else:
                max_w_test = wreal[np.arange(len(wreal)), :, argmax_test]
            max_loss_test = np.average(max_loss_test, axis=0)
            max_w_test = np.average(max_w_test, axis=0)
            return max_loss_test, max_w_test
        (loss_out[j], ws_out[j]), (loss_std_out[j], ws_std_out[j]) = \
            bootstrap_func(loss_and_w, loss_test, w_test, loss_real, w_real)
        threshold_out[j] = k_val
        j += 1
    return threshold_out, loss_out, ws_out, loss_std_out, ws_std_out


def train(tolerance_space, Dataset):
    s_train = Dataset.s_train
    y_train = Dataset.y_train
    M_train = Dataset.M_train
    X_train = Dataset.X_train
    MY_train = Dataset.MY_train
    M_test = Dataset.M_test
    M_val = Dataset.M_val
    y_test = Dataset.y_test
    y_val = Dataset.y_val
    s_test = Dataset.s_test
    s_val = Dataset.s_val
    X_test = Dataset.X_test
    X_val = Dataset.X_val
    MY_test = Dataset.MY_test
    MY_val = Dataset.MY_val
    # Dataset = BrowardDataset("../data", test_split=0.2, val_split=0.1)
    pa0 = np.sum(s_train == 0)/(s_train.shape[0])
    pa1 = np.sum(s_train != 0)/(s_train.shape[0])
    pa1y1 = np.sum((s_train != 0)*(y_train == 1))/(s_train.shape[0])
    pa1y0 = np.sum((s_train != 0)*(y_train == 0))/(s_train.shape[0])
    pa0y1 = np.sum((s_train == 0)*(y_train == 1))/(s_train.shape[0])
    pa0y0 = np.sum((s_train == 0)*(y_train == 0))/(s_train.shape[0])
    ps = (pa0, pa1, pa0y0, pa0y1, pa1y0, pa1y1)
    threshold_space = np.linspace(-.5, .5, 100)  # Eodds + ACS
    threshold_space = np.meshgrid(threshold_space, threshold_space)
    # make a list of threshold_space
    threshold_space = list(zip(threshold_space[0].flatten(),
                               threshold_space[1].flatten()))

    L_train, L_test, L_val, n_s = process_score_labels(M_train, M_test, M_val,
                                                       y_train, y_test, y_val,)
    if Path("rfs.npz").exists():
        rfs = joblib.load("rfs.npz")
        rfs = [rfs[0], rfs[1], rfs[2]]
    else:
        rfs = train_scores(X_train, y_train, MY_train, L_train)
        joblib.dump(rfs, "rfs.npz")

    def wits(X, A):
        return estimate_witnesses(witness, witness_loss, X, A, ps, rfs)

    def true_wits(L, A, MY, y):
        return true_witness(witness, witness_loss, L, A, ps, MY, y, n_s)

    threshold, max_loss, max_w, max_loss_std, \
        max_w_std = find_thresholds(wits, true_wits, X_val,
                                    y_val, MY_val, L_val, X_test,
                                    y_test, MY_test,
                                    s_val, s_test,
                                    L_test,
                                    threshold_space,
                                    tolerance_space,)
    return tolerance_space, max_loss, max_w, max_loss_std, max_w_std

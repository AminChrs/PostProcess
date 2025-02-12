import sys
sys.path.append("../")
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from pulp import LpVariable, LpProblem, lpSum
from pulp import LpMaximize, PULP_CBC_CMD, LpConstraint, LpBinary
import folktables
import joblib
from datasetsdefer.broward import BrowardDataset
import torch
from networks.linear_net import LinearNet
from torch import optim
import torch.nn.functional as F
import matplotlib
from baselines.compare_confidence import CompareConfidence
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# from madras.src.codebase.dataset import Dataset
# from madras.src.codebase.models import *
# from madras.src.codebase.trainer import Trainer, make_dir_if_not_exist
# from madras.src.codebase.defaults import get_default_kwargs

ACS_TASK = "ACSIncome"
SEED = 42
EPS = 1e-6
data_dir = Path("~").expanduser() / "data" / "folktables" / "train=0.6_test=0.2_validation=0.2_max-groups=4"
ACS_CATEGORICAL_COLS = {
    'COW',  # class of worker
    'MAR',  # marital status
    'OCCP',  # occupation code
    'POBP',  # place of birth code
    'RELP',  # relationship status
    'SEX',
    'RAC1P',  # race code
    'DIS',  # disability
    'ESP',  # employment status of parents
    'CIT',  # citizenship status
    'MIG',  # mobility status
    'MIL',  # military service
    'ANC',  # ancestry
    'NATIVITY',
    'DEAR',
    'DEYE',
    'DREM',
    'ESR',
    'ST',
    'FER',
    'GCL',
    'JWTR',
    # 'PUMA',
    # 'POWPUMA',
}


def split_X_Y_S(data, label_col: str, sensitive_col: str,
                ignore_cols=None, unawareness=False) -> tuple:
    ignore_cols = ignore_cols or []
    ignore_cols.append(label_col)
    if unawareness:
        ignore_cols.append(sensitive_col)

    feature_cols = [c for c in data.columns if c not in ignore_cols]

    return (
        data[feature_cols],                           # X
        data[label_col].to_numpy().astype(int),       # Y
        data[sensitive_col].to_numpy().astype(int),   # S
    )


def load_ACS_data(dir_path: str, task_name: str,
                  sensitive_col: str = None) -> pd.DataFrame:
    """Loads the given ACS task data from pre-generated datasets.

    Returns
    -------
    dict[str, tuple]
        A list of tuples, each tuple composed of (features, label,
        sensitive_attribute).
        The list is sorted as follows" [<train data tuple>, <test data tuple>,
        <val. data tuple>].
    """
    # Load task object
    task_obj = getattr(folktables, task_name)

    # Load train, test, and validation data
    data = dict()
    for data_type in ['train', 'test', 'validation']:
        # Construct file path
        path = Path(dir_path) / f"{task_name}.{data_type}.csv"

        if not path.exists():
            print(f"Couldn't find data\
                  for '{path.name}' \
                  (this is probably expected).")
            continue

        # Read data from disk
        df = pd.read_csv(path, index_col=0)

        # Set categorical columns
        cat_cols = ACS_CATEGORICAL_COLS & set(df.columns)
        df = df.astype({col: "category" for col in cat_cols})

        data[data_type] = split_X_Y_S(
            df,
            label_col=task_obj.target,
            sensitive_col=sensitive_col or task_obj.group,
        )

    return data


def IP_baseline(X_train, Y_train, M_train,
                s_train, MY_train,
                X_val, Y_val, M_val, s_val,
                MY_val,
                ps,
                witness, witness_loss, tolerance):

    L_train, L_val, _, n_s = process_score_labels(M_train, M_val, M_val,
                                                  Y_train, Y_val, Y_val)

    rfs = train_scores(X_train, Y_train, MY_train, L_train)
    rf_out = rfs[0].predict_proba(X_val)

    def true_wits(L, A, MY, y):
        return true_witness(witness, witness_loss, L, A, ps, MY, y, n_s)

    loss_real_val, w_real_val = true_wits(L_val, s_val, MY_val, Y_val)
    num_samples = len(Y_val)
    prob = LpProblem("IntegerProgramming", LpMaximize)
    # Define decision variable
    r = LpVariable.dicts("r", range(num_samples), cat=LpBinary)
    # binary decision variables

    # Define objective function
    objective_terms = []
    for i in range(num_samples):
        r_reshaped = [r[i]]
        r_tiled = [r_reshaped[0], r_reshaped[0]]
        cc = np.concatenate([rf_out[i, :] * (1 - r_tiled[0]), r_reshaped])
        objective_terms.append(loss_real_val[i] * cc)
    prob += lpSum(objective_terms)

    # Define constraint
    if len(w_real_val.shape) == 2:
        constraint_terms = []
        for i in range(num_samples):
            r_reshaped = [r[i]]
            r_tiled = [r_reshaped[0], r_reshaped[0]]
            cc = np.concatenate([rf_out[i] * (1 - r_tiled[0]), r_reshaped])
            # print("cc:", cc)
            # print("w_real_val[i]:", w_real_val[i])
            constraint_terms.append(w_real_val[i] * cc)
            # print("constraint_terms:", constraint_terms[i])
        constraint = LpConstraint(e=lpSum(constraint_terms), sense=-1,
                                  rhs=tolerance * num_samples)
        # print("constraint:", constraint)
        prob += constraint
        constraint = LpConstraint(e=lpSum(constraint_terms), sense=+1,
                                  rhs=-tolerance * num_samples)
        prob += constraint
    else:
        for t in range(w_real_val.shape[1]):
            constraint_terms = []
            for i in range(num_samples):
                r_reshaped = [r[i]]
                r_tiled = [r_reshaped[0], r_reshaped[0]]
                cc = np.concatenate([rf_out[i] * (1 - r_tiled[0]), r_reshaped])
                # print("cc:", cc)
                # print("w_real_val[i]:", w_real_val[i])
                constraint_terms.append(w_real_val[i, t] * cc)
                # print("constraint_terms:", constraint_terms[i])
            constraint = LpConstraint(e=lpSum(constraint_terms), sense=-1,
                                      rhs=tolerance * num_samples)
            # print("constraint:", constraint)
            prob += constraint
            constraint = LpConstraint(e=lpSum(constraint_terms), sense=+1,
                                      rhs=-tolerance * num_samples)
            prob += constraint

    # Solve the problem
    prob.solve(PULP_CBC_CMD(msg=1))
    r_values = [r[i].varValue for i in range(len(Y_val))]
    # print("r_values:", r_values)
    # print("constraint:", constraint)
    # convert r_values into binary
    r_values = np.array(r_values)
    for i in range(num_samples):
        if r_values[i] is None:
            r_values[i] = 0
    # assert r_values is either 0.0 or 1.0
    r_values = np.where(r_values > 0.5, 1, 0)
    # check whether with the new r_values, the constraint is satisfied
    constraint_ter = []
    for i in range(num_samples):
        r_reshaped = [r_values[i]]
        r_tiled = [r_reshaped[0], r_reshaped[0]]
        cc = np.concatenate([rf_out[i] * (1 - r_tiled[0]), r_reshaped])
        # print("rf_out[i]:", rf_out[i])
        # print("r_tiled[0]:", r_tiled[0])
        # print("cc:", cc)
        # print("r[i]:", r_values[i])
        # print("w*cc:", w_real_val[i] * cc)
        # print("constraint_terms:", constraint_terms[i])
        constraint_ter.append(w_real_val[i] * cc)
    constraint_ter = np.array(constraint_ter)
    cc = np.sum(constraint_ter, axis=0)
    # print("cc:", cc)
    # print("np.sum(cc):", np.sum(cc))
    # print("num_samples:", num_samples)
    # print("tolerance*num_samples:", tolerance*num_samples)
    assert np.sum(cc) <= tolerance*num_samples

    # Now I train a classifier on the defer labels
    rf_defer = RandomForestClassifier(n_jobs=-2)
    rf_defer.fit(X_val, r_values)
    return rfs[0], rf_defer


def find_IP_basline(X_train, Y_train, M_train, s_train,
                    MY_train,
                    X_val, Y_val, M_val, s_val,
                    MY_val,
                    X_test, Y_test, M_test, s_test,
                    MY_test,
                    witness, witness_loss, tolerance_space,
                    ps):
    loss_out = []
    ws_out = []
    j = 0
    for tolerance in tolerance_space:
        print("Tolerance:", tolerance)
        rf_baseline, rf_defer = IP_baseline(X_train, Y_train, M_train,
                                            s_train, MY_train,
                                            X_val, Y_val, M_val, s_val,
                                            MY_val, ps,
                                            witness, witness_loss, tolerance)

        def loss_w(X, Y, M, MY):
            Yhat_test = rf_baseline.predict(X)
            Rhat_test = rf_defer.predict(X)
            # print("Average Rhat_test:", np.average(Rhat_test))
            # Here I convert Yhat and Rhat into a one hot vector that
            # takes the third value if Rhat is 1 and the value of Yhat
            # otherwise
            oh_val = np.zeros([len(Y), 3])
            oh_val[np.arange(len(Y)), 2] = Rhat_test[np.arange(len(Y))]
            oh_val[np.arange(len(Y)), (1-Rhat_test)*Yhat_test] = 1
            # convert that into a normal vector
            noh = oh_val[np.arange(len(Y)), 1] +\
                2*oh_val[np.arange(len(Y)), 2]
            # change into int
            noh = noh.astype(int)
            # print(noh)
            # rf_real_witness = np.zeros([len(s_test), 2*n_s])
            # rf_real_witness[np.arange(len(s_test)), s_test] = 1
            # rf_real_my = np.zeros([len(M_test), 2])
            # rf_real_my[np.arange(len(M_test)), M_test] = 1
            # rf_real = np.zeros([len(Y_test), 2])
            # rf_real[np.arange(len(Y_test)), Y_test] = 1
            # loss_real = witness_loss(rf_real, rf_real_my)
            # w_real = witness(rf_real_witness)
            L_test, _, _, n_s = process_score_labels(M, M, M,
                                                     Y, Y, Y)
            loss_real, w_real = true_witness(witness, witness_loss, L_test,
                                             s_test,
                                             ps, MY, Y, n_s)
            max_loss_test = loss_real[np.arange(len(loss_real)), noh]
            if len(w_real.shape) == 2:
                max_w_test = w_real[np.arange(len(w_real)), noh]
            else:
                max_w_test = np.zeros([len(w_real), w_real.shape[1]])
                for i in range(w_real.shape[1]):
                    max_w_test[:, i] = w_real[np.arange(len(w_real)), i, noh]
            return max_loss_test, max_w_test
        (max_loss_test, max_w_test) = \
            loss_w(X_test, Y_test, M_test, MY_test)
        # max_w_test = w_real[np.arange(len(w_real)), noh]
        max_loss_test = np.average(max_loss_test, axis=0)
        max_w_test = np.average(max_w_test, axis=0)
        # print("max_w_test:", max_w_test)
        loss_out.append(max_loss_test)
        ws_out.append(max_w_test)
        # print("max_w_test:", max_w_test)
        # print("max_loss_test:", max_loss_test)

        j += 1
    return loss_out, ws_out


def human_sim(y_train, y_test, y_val, s_train, s_test, s_val, val=True):
    # Now I fit a 3-depth tree to the data

    # dt_clf = DecisionTreeClassifier(max_depth=3)
    # dt_clf.fit(X_train, y_train)
    # if s_test==0, then predict y_test+noise(0.1)
    # else predict y_test+noise(0.3)
    # print("ys: ", y_test)
    # print("ys.shape: ", y_test.shape)
    M_test = np.zeros(len(y_test))
    M_val = np.zeros(len(y_val))

    def noisy(y, s):
        m = np.zeros(len(y))
        for i in range(len(y)):
            if s[i] == 0:
                random = np.random.uniform(0, 1)
                if random < 0.85:
                    m[i] = y[i]
                else:
                    m[i] = 1-y[i]
            else:
                random = np.random.uniform(0, 1)
                if random < 0.6:
                    m[i] = y[i]
                else:
                    m[i] = 1-y[i]
        m = m.astype(int)
        return m
    M_test = noisy(y_test, s_test)
    M_train = noisy(y_train, s_train)
    # M_train = dt_clf.predict(X_train)
    # M_test = dt_clf.predict(X_test)
    if val:
        M_val = noisy(y_val, s_val)
        # M_val = dt_clf.predict(X_val)
    else:
        print("No validation data.")
    return M_train, M_test, M_val


def process_data():
    #  Load and pre-process data
    all_data = load_ACS_data(
        dir_path=data_dir, task_name=ACS_TASK,
    )
    # Unpack into features, label, and group
    X_train, y_train, s_train = all_data["train"]
    X_test, y_test, s_test = all_data["test"]
    if "validation" in all_data:
        X_val, y_val, s_val = all_data["validation"]
    else:
        print("No validation data.")
    # reduce X_val, y_val, s_val to the first 1000 samples
    # X_val = X_val[:300]
    # y_val = y_val[:300]
    # s_val = s_val[:300]
    # n_groups = len(np.unique(s_train))

    actual_prevalence = np.sum(y_train) / len(y_train)
    print(f"Global prevalence: {actual_prevalence:.1%}")

    # EPSILON_TOLERANCE = 0.05

    # FALSE_POS_COST = 1
    # FALSE_NEG_COST = 1

    # L_P_NORM = np.inf
    # constrain l-infinity distance between ROC points
    # L_P_NORM = 1
    # constrain l-1 distance between ROC points
    # L_P_NORM = 2
    # constrain l-2 distance between ROC points
    # Now I generate the predictions of the tree and call it M_train, M_test,
    # M_val
    if "validation" in all_data:
        val_data = True
    M_train, M_test, M_val = human_sim(y_train, y_test, y_val, s_train, s_test,
                                       s_val,
                                       val=val_data)

    # find the new labels that are M==Y
    MY_train = np.where(y_train == M_train, 1, 0)
    MY_test = np.where(y_test == M_test, 1, 0)
    if "validation" in all_data:
        MY_val = np.where(y_val == M_val, 1, 0)
        # print("MY_val:", MY_val)
    else:
        print("No validation data.")
    return X_train, y_train, s_train, X_test, y_test, s_test, X_val, y_val, \
        s_val, M_train, M_test, M_val, MY_train, MY_test, MY_val


def process_score_labels(M_train, M_test, M_val, y_train, y_test, y_val,
                         val=True, constraint='dp'):
    # Here I make labels L_train, L_test, L_val
    n_s = 1 + 1
    L_train = 2 * M_train + y_train
    L_test = 2 * M_test + y_test
    # print("L_test:", L_test)
    if val:
        L_val = 2 * M_val + y_val
        return L_train, L_test, L_val, n_s
    else:
        return L_train, L_test, n_s


def witness(rf_out_witness, A,
            ps,
            type_witness="dp"):
    # rf_out = rf_out_witness
    pa0, pa1, pa0y0, pa0y1, pa1y0, pa1y1 = ps
    if type_witness == "dp":
        # pa0x_pre = np.array([[rf_out_witness[j, i] for i in idx_s[0]]
        #                     for j in range(rf_out_witness.shape[0])])
        # pa0x = np.sum(pa0x_pre, axis=1)
        # pa1x = 1-pa0x

        # find the intersection of idx_s[0] and idx_M[1]
        # idxm1a0 = np.intersect1d(idx_y[0], idx_M[1])
        # pm1a0x = rf_out_witness[:, idxm1a0]
        # idxm1a1 = np.intersect1d(idx_s[1], idx_M[1])
        # pm1a1x = rf_out_witness[:, idxm1a1]
        t = []
        for i in range(len(A)):
            if A[i] == 0:
                t.append(1/pa0)
            elif A[i] != 0:
                t.append(-1/pa1)
        t = np.array(t)
        idxm1 = idx_M[1]
        pm1x = np.sum(rf_out_witness[:, idxm1], axis=1)
        ret = np.zeros([rf_out_witness.shape[0], 3])

        ret[:, 0] = 0.0*rf_out_witness.shape[0]
        ret[:, 1] = t
        ret[:, 2] = t*pm1x

    elif type_witness == "eo":
        t0 = []
        for i in range(len(A)):
            if A[i] == 1:
                t0.append(1/pa1y0)
            else:
                t0.append(-1/pa0y0)
        t1 = []
        for i in range(len(A)):
            if A[i] == 0:
                t1.append(1/pa0y1)
            elif A[i] != 0:
                t1.append(-1/pa1y1)
        idxm1y1 = np.intersect1d(idx_M[1], idx_y[1])
        idxy1 = idx_y[1]
        idxy0 = idx_y[0]
        idxm1y0 = np.intersect1d(idx_M[1], idx_y[0])
        py1x = np.sum(rf_out_witness[:, idxy1], axis=1)
        py0x = np.sum(rf_out_witness[:, idxy0], axis=1)
        pm1y1x = rf_out_witness[:, idxm1y1] 
        pm1y0x = rf_out_witness[:, idxm1y0]
        ret = np.zeros([rf_out_witness.shape[0], 2, 3])
        ret[:, 0, 0] = 0.0*rf_out_witness.shape[0]
        ret[:, 0, 1] = t1*py1x
        ret[:, 0, 2] = t1*pm1y1x[:, 0]
        ret[:, 1, 0] = 0.0*rf_out_witness.shape[0]
        ret[:, 1, 1] = t0*py0x
        ret[:, 1, 2] = t0*pm1y0x[:, 0]
        # In this case, the witness function is as
        # 0, py1a1x/py1a1-py1a0x/py1a0, pm1y1a1x/py1a1-pm1y1a0x/py1a0
    return ret


def witness_loss(rf_out, rf_out_my):
    pmy = rf_out_my[:, 1]
    ret = np.zeros([len(rf_out), 3])
    ret[:, :-1] = rf_out
    ret[:, -1] = pmy
    return ret


def bootstrap_func(function, *args, n_iter=10):
    results = []
    current_args = args
    for i in range(n_iter):
        choice = np.random.choice(len(current_args[0]), len(current_args[0]))
        # choice = np.arange(len(current_args[0]))
        new_args = [arg[choice] for arg in current_args]
        results.append(function(*new_args))
    # if there are more than one output of results, then do average on all
    avg = []
    std = []
    for i in range(len(results[0])):
        res_dim_i = [result[i] for result in results]
        res_dim_i = np.array(res_dim_i)
        # print("res shape:", res_dim_i.shape)
        avg.append(np.average(res_dim_i, axis=0))
        std.append(np.std(res_dim_i, axis=0))
    return avg, std


# def find_baseline_Madras(X_train, Y_train, M_train, s_train,
#                          X_test, Y_test, M_test, s_test,
#                          witness, witness_loss, tolerance_space,
#                          name="ACS"):
#     # make M_train_npz
#     M_train_npz = np.concatenate([M_train, M_val], axis=0)
#     # make a new train using train+valid
#     X_train_npz = np.concatenate([X_train, X_val], axis=0)
#     # pre-process X_train_npz and X_test changing from categorical to one-hot
#     # the data is 1200000x11
#     # I first find the indices of the categorical columns
#     df = pd.DataFrame(X_train_npz)
#     cat_cols = ACS_CATEGORICAL_COLS & set(df.columns)
#     cat_cols = list(cat_cols)
#     df = pd.get_dummies(df, columns=cat_cols)
#     X_train_npz = df.to_numpy()
#     # now renormalize the data
#     X_train_npz = (X_train_npz - np.mean(X_train_npz, axis=0)) / \
#         np.std(X_train_npz, axis=0)
#     # now do the same for X_test
#     df = pd.DataFrame(X_test)
#     df = pd.get_dummies(df, columns=cat_cols)
#     X_test = df.to_numpy()
#     X_test = (X_test - np.mean(X_test, axis=0)) / np.std(X_test, axis=0)

#     print("X_train_npz.shape:", X_train_npz.shape)
#     print("X_train:", X_train_npz)

#     y_train_npz = np.concatenate([Y_train, y_val], axis=0)
#     s_train_npz = np.concatenate([s_train, s_val], axis=0)
#     # if s_train_npz is one-dimensional, I make it two-dimensional
#     if len(s_train_npz.shape) == 1:
#         s_train_npz = s_train_npz[:, np.newaxis]
#     if len(s_test.shape) == 1:
#         s_test = s_test[:, np.newaxis]
#     if len(y_train_npz.shape) == 1:
#         y_train_npz = y_train_npz[:, np.newaxis]
#     if len(Y_test.shape) == 1:
#         Y_test = Y_test[:, np.newaxis]
#     if len(M_train_npz.shape) == 1:
#         M_train_npz = M_train_npz[:, np.newaxis]
#     if len(M_test.shape) == 1:
#         M_test = M_test[:, np.newaxis]
#     # indices of train and indices of valid
#     train_inds = np.arange(X_train.shape[0])
#     valid_inds = np.arange(X_train.shape[0], X_train.shape[0]+X_val.shape[0])

#     np.savez("../Madras/data/"+name+"/"+name, x_train=X_train_npz,
#              x_test=X_test,
#              y_train=y_train_npz, y_test=Y_test, attr_train=s_train_npz,
#              attr_test=s_test, ydm_train=M_train_npz, ydm_test=M_test,
#              train_inds=train_inds, valid_inds=valid_inds)


def train_scores(X_train, y_train, MY_train, L_train, constraint='dp'):
    # if constraint == 'dp':
    # witness
    if Path("rfs.npz").exists():
        rfs = joblib.load("rfs.npz")
        # rfs = [rfs[0], rfs[1], rfs[2]]
        return rfs
    else:
        # rfs = train_scores(X_train, y_train, MY_train, L_train)
        # rfs = train_score(Dataset, X_train, L_train, L_val, L_test)
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
    # elif constraint == 'e_odds':
    #     # witness
    #     rf_clf_witness = RandomForestClassifier(n_jobs=-2)
    #     rf_clf_witness.fit(X_train, L_train)

    #     # classifier
    #     rf_clf = RandomForestClassifier(n_jobs=-2)
    #     rf_clf.fit(X_train, y_train)
    #     # rejector
    #     rf_my = RandomForestClassifier(n_jobs=-2)
    #     rf_my.fit(X_train, MY_train)

    #     return rf_clf, rf_my, rf_clf_witness


def train_score_NN(dataset, X_train, L_train, L_val, L_test, constraint='dp'):
    train_loader = dataset.data_train_loader
    val_loader = dataset.data_val_loader
    test_loader = dataset.data_test_loader
    # subsitute y_train with L_train
    train_loader.dataset.y = L_train
    val_loader.dataset.y = L_val
    test_loader.dataset.y = L_test
    # rf_clf_witness = RandomForestClassifier(n_jobs=-2)
    # rf_clf_witness.fit(X_train, L_train)
    optimizer = optim.AdamW
    scheduler = None
    lr = 0.01
    model_L = LinearNet(dataset.d, 4).to(device)
    model_dummy = LinearNet(dataset.d, 2).to(device)
    compareconfidence = CompareConfidence(model_L, model_dummy, device)
    compareconfidence.fit(
        train_loader,
        val_loader,
        test_loader,
        epochs=500,
        optimizer=optimizer,
        scheduler=scheduler,
        lr=lr,
        verbose=False,
        test_interval=5)
    model_class = LinearNet(dataset.d, 2).to(device)
    model_expert = LinearNet(dataset.d, 2).to(device)
    compareconfidence = CompareConfidence(model_class, model_expert, device)
    compareconfidence.fit(
        dataset.data_train_loader,
        dataset.data_val_loader,
        dataset.data_test_loader,
        epochs=500,
        optimizer=optimizer,
        scheduler=scheduler,
        lr=lr,
        verbose=False,
        test_interval=5)
    return model_class, model_expert, model_L


def estimate_witnesses_NN(witness, witness_loss, X, A, ps, rfs):
    rf_clf, rf_my, rf_clf_witness = rfs
    X_loc = torch.tensor(X).to(device).unsqueeze(0)
    rf_out = F.softmax(rf_clf(X_loc), dim=2).detach().cpu().numpy()[0, :, :]
    # rf_out_witness = rf_clf_witness.predict_proba(X)
    rf_out_witness = F.softmax(rf_clf_witness(X_loc),
                               dim=2).detach().cpu().numpy()[0, :, :]
    rf_out_my = F.softmax(rf_my(X_loc), dim=2).detach().cpu().numpy()[0, :, :]
    # print("rf_out:", rf_out)
    # print("rf_out_my:", rf_out_my)
    # sys.exit()
    loss = witness_loss(rf_out, rf_out_my)
    w = witness(rf_out_witness, A, ps)
    return loss, w


def estimate_witnesses(witness, witness_loss, X, A, ps, rfs):
    rf_clf, rf_my, rf_clf_witness = rfs
    rf_out = rf_clf.predict_proba(X)
    rf_out_witness = rf_clf_witness.predict_proba(X)
    rf_out_my = rf_my.predict_proba(X)
    loss = witness_loss(rf_out, rf_out_my)
    w = witness(rf_out_witness, A, ps)
    return loss, w


def true_witness(witness, witness_loss, L, A, ps, MY, y, n_s):

    rf_real = np.zeros([len(y), 2])
    rf_real[np.arange(len(y)), y] = 1
    # turn rf_real_witness into a one-hot vector
    rf_real_witness = np.zeros([len(L), 4])
    rf_real_witness[np.arange(len(L)), L] = 1
    rf_real_my = np.zeros([len(MY), 2])
    rf_real_my[np.arange(len(MY)), MY] = 1
    loss_real = witness_loss(rf_real, rf_real_my)
    w_real = witness(rf_real_witness, A, ps)
    return loss_real, w_real


def find_thresholds(wits, true_wits, X_val, y_val,
                    MY_val, L_val, X_test, y_test, MY_test,
                    s_val, s_test,
                    L_test, threshold_space, tolerance_space,):

    loss, w = wits(X_val, s_val)
    loss_test, w_test = wits(X_test, s_test)
    # print("w_test:", w_test)
    # print("loss_test:", loss_test)
    # sys.exit()
    # turn y_val into a one-hot vector
    loss_real, w_real = true_wits(L_test, s_test, MY_test, y_test)
    # print("w_real:", w_real)
    loss_real_val, w_real_val = true_wits(L_val, s_val, MY_val, y_val)

    losses_val = []
    ws_val = []
    tt = 0
    for k in threshold_space:
        tt += 1
        print("Threshold:", tt)
        # go over val_data that is a pandas dataframe
        # lag = loss-w^T.k
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

    # plt.scatter(ws_val, losses_val)
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
    print("tolerance_space:", tolerance_space)
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
            # print("indices:", indices)
        # print(indices.shape)
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
                # print("two-dimensional w")
                max_w_test = wreal[np.arange(len(wreal)), :, argmax_test]
            max_loss_test = np.average(max_loss_test, axis=0)
            max_w_test = np.average(max_w_test, axis=0)
            # print("max_w_test.shape:", max_w_test.shape)
            return max_loss_test, max_w_test
        (loss_out[j], ws_out[j]), (loss_std_out[j], ws_std_out[j]) = \
            bootstrap_func(loss_and_w, loss_test, w_test, loss_real, w_real)
        # print("ws_out_shape:", ws_out[j].shape)
        # loss_out[j] = max_loss_test
        # print("tolerance:", tol)
        print("ws max_index:", ws_val[max_index])
        print("ws_test:", ws_out[j])
        # print("loss max_index:", losses_val[max_index])
        # print("loss_test:", max_loss_test)
        # sys.exit()
        threshold_out[j] = k_val
        # ws_out[j] = max_w_test
        j += 1
    return threshold_out, loss_out, ws_out, loss_std_out, ws_std_out


def process_data_compas():
    data_dir = './exp_data/data'
    dataset = BrowardDataset(data_dir, test_split=0.2, val_split=0.1)
    # I run over the data and collect x, y, s, and m
    data_train_loader = dataset.data_train_loader
    data_val_loader = dataset.data_val_loader
    data_test_loader = dataset.data_test_loader

    def data_from_loader(loader):
        X = []
        Y = []
        S = []
        M = []
        for i, (x, y, m, s) in enumerate(loader):
            X.append(x)
            Y.append(y)
            S.append(s)
            M.append(m)
        X = torch.cat(X, dim=0)
        Y = torch.cat(Y, dim=0)
        S = torch.cat(S, dim=0)
        M = torch.cat(M, dim=0)
        # convert to numpy
        X = X.numpy()
        # make X into a readable format for sklearn

        Y = Y.numpy()
        S = S.numpy()
        M = M.numpy()
        S -= 1
        return X, Y, S, M
    X_train, y_train, s_train, M_train = data_from_loader(data_train_loader)
    X_val, y_val, s_val, M_val = data_from_loader(data_val_loader)
    X_test, y_test, s_test, M_test = data_from_loader(data_test_loader)
    YM_train = np.where(y_train == M_train, 1, 0)
    YM_test = np.where(y_test == M_test, 1, 0)
    YM_val = np.where(y_val == M_val, 1, 0)
    # print("X_train:", X_train)
    # print("Y_train:", y_train)
    # print("S_train:", s_train)
    # print("M_train:", M_train)
    # print("YM_train:", YM_train)
    return X_train, y_train, s_train, M_train, X_test, y_test, s_test, M_test, X_val, y_val, s_val, M_val, YM_train, YM_test, YM_val, dataset


# Here, for a variety of tolerances between 0 and 1, I find
# the accuracy and the witness of the model on the validation data

# X_train, y_train, s_train, X_test, y_test, s_test, X_val, y_val, \
#     s_val, M_train, M_test, M_val, MY_train, MY_test, MY_val = \
#     process_data()
# print("X_train:", X_train)
# print("y_train:", y_train)
# print("s_train:", s_train)
# print("M_train:", M_train)
# print("MY_train:", MY_train)
idx_y = [[0, 2], [1, 3]]
idx_M = [[0, 1], [2, 3]]


def train(tolerance_space):
    X_train, y_train, s_train, M_train, X_test, y_test, s_test, M_test, \
        X_val, y_val, s_val, M_val, MY_train, MY_test, MY_val, Dataset = \
        process_data_compas()
    # X_train, y_train, s_train, X_test, y_test, s_test, X_val, y_val, \
    #     s_val, M_train, M_test, M_val, MY_train, MY_test, MY_val = \
    #     process_data()
    # Dataset = BrowardDataset("../data", test_split=0.2, val_split=0.1)
    pa0 = np.sum(s_train == 0)/(s_train.shape[0])
    pa1 = np.sum(s_train != 0)/(s_train.shape[0])
    pa1y1 = np.sum((s_train != 0)*(y_train == 1))/(s_train.shape[0])
    pa1y0 = np.sum((s_train != 0)*(y_train == 0))/(s_train.shape[0])
    pa0y1 = np.sum((s_train == 0)*(y_train == 1))/(s_train.shape[0])
    pa0y0 = np.sum((s_train == 0)*(y_train == 0))/(s_train.shape[0])
    ps = (pa0, pa1, pa0y0, pa0y1, pa1y0, pa1y1)
    # print("pa0:", pa0)
    # print("pa1:", pa1)
    # threshold_space = np.linspace(-.5, .5, 20)  # Eodds + ACS
    threshold_space = np.linspace(-5, 5, 10000)
    # threshold_space = np.meshgrid(threshold_space, threshold_space)
    # make a list of threshold_space
    # threshold_space = list(zip(threshold_space[0].flatten(), threshold_space[1].flatten()))
    # tolerance_space = np.linspace(0.01, 0.2, 10)  # Eodds + ACS 100
    # tolerance_space = np.linspace(0.01, 0.5, 2)

    max_losses = []
    max_ws = []
    L_train, L_test, L_val, n_s = process_score_labels(M_train, M_test, M_val,
                                                       y_train, y_test, y_val,)
    # search for rfs.npz
    # if False:
    if Path("rfs.npz").exists():
        rfs = joblib.load("rfs.npz")
        rfs = [rfs[0], rfs[1], rfs[2]]
    else:
        # rfs = train_scores(X_train, y_train, MY_train, L_train)
        rfs = train_score_NN(Dataset, X_train, L_train, L_val, L_test)
        # save random forests
        joblib.dump(rfs, "rfs.npz")

    def wits(X, A):
        return estimate_witnesses_NN(witness, witness_loss, X, A, ps, rfs)

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
    # print(tolerance_space, max_loss, max_w, max_loss_std, max_w_std)
    # print("Before finding the baseline")
    # loss_baseline, ws_baseline = find_IP_basline(X_train, y_train, M_train,
    # s_train,
    #                                              MY_train,
    #                                              X_val, y_val, M_val, s_val,
    #                                             MY_val,
    #                                              X_test, y_test, M_test,
    # s_test,
    #                                              MY_test,
    #                                              witness, witness_loss,
    # tolerance_space,
    #                                              ps)
    return tolerance_space, max_loss, max_w, max_loss_std, max_w_std
    # return tolerance_space, loss_baseline, ws_baseline
    # sys.exit()
    # return threshold, max_loss, max_w, max_loss_std, max_w_std


def postprocess():
    tolerance_space = np.linspace(0.01, 0.2, 100)
    threshold = []
    max_loss = []
    max_w = []
    max_loss_std = []
    max_w_std = []
    max_iter = 1
    for i in range(max_iter):
        th, ml, mw, mls, mws = train(tolerance_space)
        if i == 0:
            threshold = th
            max_loss = ml
            max_w = mw
            max_loss_std = mls**2
            max_w_std = mws**2
        else:
            threshold += th
            max_loss += ml
            max_w += mw
            max_loss_std += mls**2
            max_w_std += mws**2

    threshold /= max_iter
    max_loss /= max_iter
    max_w /= max_iter
    max_loss_std /= max_iter
    max_w_std /= max_iter
    max_loss_std = np.sqrt(max_loss_std)
    max_w_std = np.sqrt(max_w_std)


    # remove the 0 accuracies, because it means the tolerance is not achievable
    idx_Z = np.where(np.abs(max_loss) > 1e-8)
    idx_Z = idx_Z[0]
    tols = tolerance_space[idx_Z]
    max_loss = max_loss[idx_Z]
    max_loss_std = max_loss_std[idx_Z]
# I plot the accuracy and the witness for the validation data
    plot = False
    if plot:
        plt.fill_between(tols,
                        max_loss-max_loss_std,
                        max_loss+max_loss_std, alpha=0.5)
        max_w_o = []
        max_w_std_o = []
        if isinstance(max_w[0], float):
            max_w_o = np.array(max_w)
            max_w_std_o = np.array(max_w_std)
            plt.fill_between(tols,
                            np.abs(max_w[idx_Z]-max_w_std[idx_Z]),
                            np.abs(max_w[idx_Z]+max_w_std[idx_Z]), alpha=0.5)
        else:
            for i in range(len(max_w[0])):
                max_w_i = np.array([max_w[j][i] for j in range(len(max_w))])
                max_w_std_i = np.array([max_w_std[j][i]
                                        for j in range(len(max_w_std))])
                # now I make a shadowed plot
                max_w_o.append(max_w_i)
                max_w_std_o.append(max_w_std_i)
                plt.fill_between(tols,
                                np.abs(max_w_i[idx_Z]-max_w_std_i[idx_Z]),
                                np.abs(max_w_i[idx_Z]+max_w_std_i[idx_Z]),
                                alpha=0.5)
    return tols, max_loss, max_w[idx_Z], max_loss_std, max_w_std[idx_Z]


def plot_tabular():
    tolerance_space = np.linspace(0.01, 0.2, 100)
    tolerance_space, max_loss, max_w, max_loss_std, max_w_std = train()
    # remove the max_losses that are zero, because they are not achievable
    # tols, loss, ws = train()
    idx_Z = np.where(np.abs(max_loss) > 1e-8)
    idx_Z = idx_Z[0]
    tols = tolerance_space[idx_Z]
    max_loss = max_loss[idx_Z]
    max_loss_std = max_loss_std[idx_Z]
    # if isinstance(max_w[0], float):
    #     # change this into w[idx_Z]

    # else:
    #     max_w = np.array([max_w[j] for j in range(len(max_w))])
    #     max_w_std = np.array([max_w_std[j] for j in range(len(max_w_std))])
    # print(max_loss-max_loss_std)
    # print(max_loss)
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    # fill
    plt.fill_between(tols, max_loss-max_loss_std, max_loss+max_loss_std,
                     alpha=0.5)
    plt.plot(tols, max_loss)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    plt.grid()
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=15)
    plt.xlabel("Constraint Tolerance", fontsize=20)
    plt.ylabel("Test Accuracy", fontsize=20)
    fig_size = plt.rcParams["figure.figsize"]

    fig_size[0] = 6
    fig_size[1] = 4.2
    plt.savefig('obj.pdf', bbox_inches='tight', dpi=1000)
    plt.show()
    print("max_w:", max_w)
    # print("max_w shape:", max_w.shape)
    const_names = ["TPR", "TNR"]
    colors = ['blue', 'red']
    if isinstance(max_w[0], float):
        plt.fill_between(tols, max_w-max_w_std, max_w+max_w_std, alpha=0.5)
        plt.xlabel("Constraint Tolerance")
        plt.ylabel("Constraint Violation")
        plt.show()
    else:
        for i in range(len(max_w[0])):
            max_w_i = np.array([np.abs(max_w[j][i])
                                for j in range(len(max_w))])
            max_w_std_i = np.array([max_w_std[j][i]
                                    for j in range(len(max_w_std))])
            # print("max_w_i[idx_Z]:", max_w_i[idx_Z])
            # print("max_w_std_i[idx_Z]:", max_w_std_i[idx_Z])
            plt.fill_between(tols, max_w_i[idx_Z]-max_w_std_i[idx_Z],
                             max_w_i[idx_Z]+max_w_std_i[idx_Z],
                             alpha=0.5, color=colors[i])
            plt.plot(tols, max_w_i[idx_Z], label=const_names[i],
                     color=colors[i])
            plt.xlabel("Constraint Tolerance", fontsize=20)
            plt.ylabel("Constraint Violation", fontsize=20)
            plt.legend()
            # plt.show()
    # plt.fill_between(tols, max_w-max_w_std, max_w+max_w_std, alpha=0.5,
            # label="Constraint Violation")
    # draw x=y line using dotted
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    plt.grid()
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=15)
    plt.plot(tolerance_space, tolerance_space, 'k--')
    # plt.xlabel("Constraint Tolerance")
    # plt.ylabel("Constraint Violation")
    # plt.plot(tols, ws, label="Witness baseline")
    # plt.plot(tols, max_w, label="Witness")
    # plt.legend()
    plt.savefig('consts.pdf', bbox_inches='tight', dpi=1000)
    plt.show()

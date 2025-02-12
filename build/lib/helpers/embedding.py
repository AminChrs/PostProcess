import numpy as np

idx_y_emb = [[0, 2], [1, 3]]
idx_M_emb = [[0, 1], [2, 3]]


def process_score_labels(M_train, M_test, M_val, y_train, y_test, y_val,
                         val=True, constraint='eo'):
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
            type_witness="eo"):
    # rf_out = rf_out_witness
    pa0, pa1, pa0y0, pa0y1, pa1y0, pa1y1 = ps
    if type_witness == "dp":
        t = []
        for i in range(len(A)):
            if A[i] == 0:
                t.append(1/pa0)
            elif A[i] != 0:
                t.append(-1/pa1)
        t = np.array(t)
        idxm1 = idx_M_emb[1]
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
        idxm1y1 = np.intersect1d(idx_M_emb[1], idx_y_emb[1])
        idxy1 = idx_y_emb[1]
        idxy0 = idx_y_emb[0]
        idxm1y0 = np.intersect1d(idx_M_emb[1], idx_y_emb[0])
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

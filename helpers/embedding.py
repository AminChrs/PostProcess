import numpy as np
from sklearn.ensemble import RandomForestClassifier
from helpers.utils import argmax_constrained

idx_y_emb = [[0, 2], [1, 3]]
idx_M_emb = [[0, 1], [2, 3]]


class embedding:

    def __init__(self, type_emb, net_type):
        self.type_emb = type_emb
        self.net_type = net_type
        self.init_estimator()

    def init_estimator(self):
        if self.net_type == "rf" and self.type_emb != "loss":
            self.net = RandomForestClassifier(n_jobs=-2)
        elif self.net_type == "rf" and self.type_emb == "loss":
            self.nets = []
            self.nets.append(RandomForestClassifier(n_jobs=-2))
            self.nets.append(RandomForestClassifier(n_jobs=-2))

    def fit(self, **kwargs):
        if self.type_emb == "dp":
            return self.fit_dp(**kwargs)
        elif self.type_emb == "eo":
            return self.fit_eo(**kwargs)
        elif self.type_emb == "loss":
            return self.fit_loss(**kwargs)

    def calculate(self, Dataset, data_type, calulation_type):
        if calulation_type == "estimate":
            if self.type_emb != "loss":
                net_out = self.net.predict_proba(Dataset.X[data_type])
            elif self.type_emb == "loss":
                net_out = []
                net_out.append(
                    self.nets[0].predict_proba(Dataset.X[data_type]))
                net_out.append(
                    self.nets[1].predict_proba(Dataset.X[data_type]))
        elif calulation_type == "true":
            if self.type_emb == 'loss':
                net_out = []
                net_out[0] = np.zeros([len(Dataset.y[data_type]), 2])
                net_out[0][np.arange(len(Dataset.y[data_type])),
                           Dataset.y[data_type]] = 1
                net_out[1] = np.zeros([len(Dataset.MY[data_type]), 2])
                net_out[1][np.arange(len(Dataset.MY[data_type])),
                           Dataset.MY[data_type]] = 1
            else:
                net_out = np.zeros([len(Dataset.L[data_type]), 4])
                net_out[np.arange(len(Dataset.L[data_type])),
                        Dataset.L[data_type]] = 1

        if self.type_emb == "loss":
            ret = self.calculate_embedding_from_scores(
                net_out, Dataset.A[data_type], Dataset.ps)
        else:
            ret = self.calculate_embedding_from_scores(net_out)
        return ret

    def fit_dp(self, Dataset):
        self.net.fit(Dataset.X['train'], Dataset.L['train'])

    def fit_eo(self, Dataset):
        self.net.fit(Dataset.X['train'], Dataset.L['train'])

    def fit_loss(self, Dataset):
        self.nets[0].fit(Dataset.X['train'], Dataset.y['train'])
        self.nets[1].fit(Dataset.X['train'], Dataset.MY['train'])

    def calculate_embedding_from_scores(self, **kwargs):
        if self.type_emb == "dp":
            return self.calculate_embedding_dp(**kwargs)
        elif self.type_emb == "eo":
            return self.calculate_embedding_eo(**kwargs)
        elif self.type_emb == "loss":
            return self.calculate_embedding_loss(**kwargs)

    def calculate_embedding_dp(self, net_out, A, ps):

        t = []
        for i in range(len(A)):
            if A[i] == 0:
                t.append(1/ps.pa0)
            elif A[i] != 0:
                t.append(-1/ps.pa1)
        t = np.array(t)
        idxm1 = idx_M_emb[1]
        pm1x = np.sum(net_out[:, idxm1], axis=1)
        ret = np.zeros([net_out.shape[0], 3])

        ret[:, 0] = 0.0*net_out.shape[0]
        ret[:, 1] = t
        ret[:, 2] = t*pm1x
        return ret

    def calculate_embedding_eo(self, net_out, A, ps):
        t0 = []
        for i in range(len(A)):
            if A[i] == 1:
                t0.append(1/ps.pa1y0)
            else:
                t0.append(-1/ps.pa0y0)
        t1 = []
        for i in range(len(A)):
            if A[i] == 0:
                t1.append(1/ps.pa0y1)
            elif A[i] != 0:
                t1.append(-1/ps.pa1y1)
        idxm1y1 = np.intersect1d(idx_M_emb[1], idx_y_emb[1])
        idxy1 = idx_y_emb[1]
        idxy0 = idx_y_emb[0]
        idxm1y0 = np.intersect1d(idx_M_emb[1], idx_y_emb[0])
        py1x = np.sum(net_out[:, idxy1], axis=1)
        py0x = np.sum(net_out[:, idxy0], axis=1)
        pm1y1x = net_out[:, idxm1y1] 
        pm1y0x = net_out[:, idxm1y0]
        ret = np.zeros([net_out.shape[0], 2, 3])
        ret[:, 0, 0] = 0.0*net_out.shape[0]
        ret[:, 0, 1] = t1*py1x
        ret[:, 0, 2] = t1*pm1y1x[:, 0]
        ret[:, 1, 0] = 0.0*net_out.shape[0]
        ret[:, 1, 1] = t0*py0x
        ret[:, 1, 2] = t0*pm1y0x[:, 0]
        return ret

    def calculate_embedding_loss(self, nets_out):
        pmy = nets_out[1][:, 1]
        ret = np.zeros([len(nets_out[0]), 3])
        ret[:, :-1] = nets_out[0]
        ret[:, -1] = pmy
        return ret


class classifier:

    def __init__(self, embs):
        self.embs = embs

    def calculate_embs(self, Dataset, data_type, calculation):
        embs_out = []
        for emb in self.embs:
            embs_out.append(emb.calculate(Dataset, data_type, calculation))
        self.embs_out = embs_out

    def predict(self, k, Dataset, data_type, calculation):
        self.calculate_embs(Dataset, data_type, calculation)
        if isinstance(k, float):
            k = [k]
            self.embs_out[1] = np.tile(self.embs_out[1], (len(k), 1))
        lagrang = self.embs_out[0]
        for i in range(len(k)):
            lagrang -= k[i]*self.embs_out[1][:, i]

        argmax = np.argmax(lagrang, axis=1)
        self.predictions = argmax
        return argmax

    def mean_emb_predict(self, Dataset, data_type, calculation, mean=True):
        self.calculate_embs(Dataset, data_type, calculation)
        mean_out = []
        for emb_out in self.embs_out:
            idxs = np.arange(len(emb_out))
            if len(emb_out.shape) == 2:
                pred_val = emb_out[idxs, self.predictions]
            elif len(emb_out.shape) == 3:
                pred_val = emb_out[idxs, :, self.predictions]
            if mean:
                pred_val = np.average(pred_val, axis=0)
            pred_val = np.array(pred_val)
            mean_out.append(pred_val)
        return mean_out

    def optimal_combination(self,
                            Dataset, coefficient_space, tolerance_space,):

        for k in coefficient_space:
            self.predict(k, Dataset, 'val', 'estimate')
            mean_out = self.mean_emb_predict(Dataset, 'val', 'true')
        coeffs = []
        for tol in tolerance_space:
            # find the minimum loss for all the ones with ws < tol
            max_index = argmax_constrained(mean_out[0],
                                           mean_out[1], np.abs(tol))
            if max_index is None:
                coeffs.append(None)
                continue
            coeff_max = coefficient_space[max_index]
            coeffs.append(coeff_max)
        return coeffs

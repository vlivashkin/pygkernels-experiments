from copy import deepcopy
from itertools import combinations

import numpy as np
from joblib import delayed, Parallel
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm


class RFE:
    def __init__(self, estimator, names, max_features=5, n_jobs=None):
        self.estimator = estimator
        self.names = names
        self.max_features = max_features
        self.n_jobs = n_jobs

    def score_one(self, X_train, y_train, X_val, y_val, ari_val, set_feat_names):
        support = np.array([x in set_feat_names for x in self.names], dtype=np.bool)

        estimator = deepcopy(self.estimator)
        estimator.fit(X_train[:, support], y_train)
        y_pred = estimator.predict(X_val[:, support])
        ari = np.mean(ari_val[range(y_pred.shape[0]), np.argmax(y_pred, axis=1)])

        acc = -1
        f1 = -1
        #         acc = accuracy_score(y_val.ravel(), y_pred.ravel())
        #         f1 = f1_score(y_val.ravel(), y_pred.ravel())

        return set_feat_names, acc, f1, ari, estimator

    def fit(self, X_train, y_train, X_val=None, y_val=None, ari_val=None):
        # for all features first:
        _, acc_all, f1_all, ari_all, estimator_all = self.score_one(X_train, y_train, X_val, y_val, ari_val, self.names)
        print(f'all features, acc={acc_all:.3f}, f1={f1_all:.3f}, ari={ari_all:.3f}')

        for n_features in range(1, self.max_features + 1):
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self.score_one)(X_train, y_train, X_val, y_val, ari_val, set_feat_names)
                for set_feat_names in tqdm(list(combinations(self.names, n_features))))
            best_n_feat, best_acc, best_f1, best_ari, best_estimator = None, 0, 0, 0, None
            for set_feat_names, acc, f1, ari, estimator in results:
                if ari > best_ari:
                    best_n_feat, best_acc, best_f1, best_ari, best_estimator = set_feat_names, acc, f1, ari, estimator
            print(f'{n_features} features, acc={best_acc:.3f}, f1={best_f1:.3f}, ari={best_ari:.3f}, set={best_n_feat}')

        return self


class RFE_LOO(RFE):
    def loo(self, shape, block_size=25 * 7):
        assert shape % block_size == 0
        n_blocks = shape // block_size
        for i in range(n_blocks):
            val_mask = [block_size * i + j for j in range(block_size)]
            train_index = np.ones((shape,), dtype=np.bool)
            train_index[val_mask] = False
            val_index = ~train_index
            yield train_index, val_index

    def score_one(self, X, y, X_val=None, y_val=None, support=None):
        acc, f1 = [], []
        for train_index, val_index in self.loo(X.shape[0]):
            X_train, y_train, X_val, y_val = X[train_index], y[train_index], X[val_index], y[val_index]
            estimator = deepcopy(self.estimator)
            estimator.fit(X_train[:, support], y_train)
            y_pred = estimator.predict(X_val[:, support])
            acc.append(accuracy_score(y_val.ravel(), y_pred.ravel()))
            f1.append(f1_score(y_val.ravel(), y_pred.ravel()))
        return np.mean(acc), np.mean(f1)


class OneVsRest_custom:
    def __init__(self, estimator, weight_samples=True):
        self.estimator = estimator
        self.weight_samples = weight_samples
        self.n_classes = 0
        self.svms = None

    def fit(self, X, y):
        self.n_classes = y.shape[1]
        self.svms = [deepcopy(self.estimator) for _ in range(self.n_classes)]
        for i in range(self.n_classes):
            if self.weight_samples:
                sample_weight = [1 / np.sum(ys == np.max(ys)) if ys[i] == np.max(ys) else 1 for ys in y]
                self.svms[i].fit(X, y[:, i], sample_weight)
            else:
                self.svms[i].fit(X, y[:, i])
        return self

    def predict(self, X):
        y_pred = []
        for i in range(self.n_classes):
            y_pred.append(np.array(self.svms[i].predict(X)))
        return np.stack(y_pred, axis=1)


class OneHotEncoding_custom:
    def __init__(self, estimator, weight_samples=True):
        self.estimator = estimator
        self.weight_samples = weight_samples
        self.n_classes = 0

    def _flattenX(self, X):
        flat_X = []
        for i in range(self.n_classes):
            ones = np.zeros((X.shape[0], self.n_classes))
            ones[:, i] = 1
            flat_X.append(np.concatenate((X, ones), axis=1))
        flat_X = np.concatenate(flat_X, axis=0)
        return flat_X

    def _flattenY(self, y):
        flat_y, flat_sample_weights = [], []
        for i in range(self.n_classes):
            flat_sample_weights.append(
                np.array([1 / np.sum(ys == np.max(ys)) if ys[i] == np.max(ys) else 1 for ys in y]))
            flat_y.append(y[:, i])
        flat_y = np.concatenate(flat_y, axis=0)
        flat_sample_weights = np.concatenate(flat_sample_weights, axis=0)
        return flat_y, flat_sample_weights

    def fit(self, X, y):
        self.n_classes = y.shape[1]
        flat_X = self._flattenX(X)
        flat_y, flat_sample_weights = self._flattenY(y)
        if self.weight_samples:
            self.estimator.fit(flat_X, flat_y, flat_sample_weights)
        else:
            self.estimator.fit(flat_X, flat_y)
        return self

    def predict(self, X):
        flat_X = self._flattenX(X)
        flat_y = self.estimator.predict(flat_X)
        return flat_y.reshape(self.n_classes, -1).transpose((1, 0))

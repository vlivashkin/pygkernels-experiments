from copy import deepcopy
from itertools import combinations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm


class RFE_custom:
    def __init__(self, estimator, max_features=5):
        self.estimator = estimator
        self.max_features = max_features

    def fit(self, X_train, y_train, X_val, y_val, names):
        # for all features first:
        estimator = deepcopy(self.estimator)
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_val)
        acc_all = accuracy_score(y_val.ravel(), y_pred.ravel())
        f1_all = f1_score(y_val.ravel(), y_pred.ravel())
        print(f'all features, acc={acc_all:.3f}, f1={f1_all:.3f}')

        for n_features in range(1, self.max_features):
            best_n_feat, best_acc, best_f1 = None, 0, 0
            for set_feat_names in tqdm(list(combinations(names[:-25], n_features))):
                support = np.array([x in set_feat_names for x in names[:-25]] + [1] * 25, dtype=np.bool)
                estimator = deepcopy(self.estimator)
                estimator.fit(X_train[:, support], y_train)
                y_pred = estimator.predict(X_val[:, support])
                acc = accuracy_score(y_val.ravel(), y_pred.ravel())
                f1 = f1_score(y_val.ravel(), y_pred.ravel())
                if f1 > best_f1:
                    best_n_feat, best_acc, best_f1 = set_feat_names, acc, f1
            print(f'{n_features} features, set={best_n_feat} acc={best_acc:.3f}, f1={best_f1:.3f}')

        return self

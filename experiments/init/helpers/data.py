import numpy as np
from joblib import delayed, Parallel
from tqdm import tqdm


class Data:
    CACHE_ROOT = '../../cache/cache'

    kernel_names = [
        'Katz', 'logKatz',
        'For', 'logFor',
        'Comm', 'logComm',
        'Heat', 'logHeat',
        'NHeat', 'logNHeat',
        'SCT', 'SCCT',
        'RSP', 'FE',
        'PPR', 'logPPR',
        'ModifPPR', 'logModifPPR',
        'HeatPR', 'logHeatPR',
        'DF', 'logDF',
        'Abs', 'logAbs',
        'SP-CT'
    ]

    feature_names = [
        'n', 'k', 'p_in', 'p_out', 'n/k', 'p_in/p_out',
        'log(n)/k * p_in/p_out', 'n/k * p_in/p_out', 'log(n/k) * p_in/p_out', 'log(n/k * p_in/p_out)',
        'sbm_neighbour_score',
        'modularity', 'diameter', 'density',
        'avg_deg', 'std_deg', 'avg(deg | deg > avg_deg)', 'median_deg', 'max_deg',
        'avg_sp', 'std_sp', 'median_sp', 'max_sp',
        'max_clique', 'max_clique/(n/k)'
    ]

    allowed_features = {
        'n': True,
        'k': True,
        'p_in': True,
        'p_out': True,
        'n/k': True,
        'p_in/p_out': True,
        'log(n)/k * p_in/p_out': True,
        'n/k * p_in/p_out': False,
        'log(n/k) * p_in/p_out': False,
        'log(n/k * p_in/p_out)': False,
        'sbm_neighbour_score': True,
        'modularity': True,
        'diameter': True,
        'density': True,
        'avg_deg': True,
        'std_deg': True,
        'avg(deg | deg > avg_deg)': True,
        'median_deg': True,
        'max_deg': False,
        'avg_sp': True,
        'std_sp': True,
        'median_sp': True,
        'max_sp': False,
        'max_clique': False,
        'max_clique/(n/k)': True
    }
    allowed_features_list = [k for k, v in allowed_features.items() if v]

    features_to_log = {
        'n': False,
        'k': False,
        'p_in': True,
        'p_out': True,
        'n/k': False,
        'p_in/p_out': True,
        'log(n)/k * p_in/p_out': True,
        'n/k * p_in/p_out': True,
        'log(n/k) * p_in/p_out': True,
        'log(n/k * p_in/p_out)': False,
        'sbm_neighbour_score': False,
        'modularity': False,
        'diameter': True,
        'density': True,
        'avg_deg': True,
        'std_deg': True,
        'avg(deg | deg > avg_deg)': True,
        'median_deg': True,
        'max_deg': True,
        'avg_sp': True,
        'std_sp': True,
        'median_sp': True,
        'max_sp': True,
        'max_clique': True,
        'max_clique/(n/k)': True
    }
    features_to_log_list = [k for k, v in features_to_log.items() if v]

    def __init__(self):
        self.datasets_partition = None

    def load_precalculated(self):
        pass

    def extract_feature(self, dataset_info, feature, G=None, partition=None, sp=None, max_clique=None):
        pass

    def make_dataset(self, return_clf=True):
        _, results_modularity_any3, _ = self.load_precalculated()
        Xy_list = Parallel(n_jobs=1)(delayed(self.prepare_column)(results_modularity_any3, column)
                                     for column in tqdm(results_modularity_any3.keys(), desc='prepare columns'))

        X, y, X_train, y_train, X_val, y_val = [], [], [], [], [], []
        for Xi, yi in Xy_list:
            Xi = np.array([[np.log(Xii[fname]) if fname in self.features_to_log_list else Xii[fname]
                            for fname in self.allowed_features_list] for Xii in Xi])
            yi = np.array([[yii[kname] for kname in self.kernel_names] for yii in yi])
            if return_clf:
                yi = yi > (np.max(yi, axis=1, keepdims=True) - 0.0001)

            X.append(Xi)
            y.append(yi)

        X, y = np.array(X), np.array(y)  # X: [n_columns, 100, n_allowed_features], y: [n_columns, 100, n_kernels]

        for i, name in enumerate(self.allowed_features_list):
            was_logged = name in self.features_to_log_list
            X[:, :, i] = ((X[:, :, i] - np.mean(X[:, :, i])) if was_logged else X[:, :, i]) / np.std(X[:, :, i])

        assert X.shape[2] == len(self.allowed_features_list)

        return X, y

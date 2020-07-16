import os
import pickle
import random
import sys
from collections import defaultdict
from copy import deepcopy
from itertools import combinations, product

import networkx as nx
import numpy as np
from joblib import delayed, Parallel
from networkx.algorithms.approximation import clique
from scipy.stats import rankdata
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from sbm_neighbour_score import sbm_neighbour_score, graph_neighbour_score

sys.path.append('../../pygkernels')
from pygkernels.data.dataset import Datasets


# continue coefficients for nemenyi 0.05
# plt.plot([1.959964, 2.343701, 2.569032, 2.727774, 2.849705, 2.94832, 3.030879, 3.101730, 3.163684,
#           3.218654, 3.268004, 3.312739, 3.353618, 3.39123, 3.426041, 3.458425, 3.488685, 3.517073, 3.543799])
# a = 1.95 + 1/1.85 * np.log(range(1, 25))
# plt.plot(a)
# print(a)  # 3.56931474 3.5956878  3.62083376 3.64486174 3.66786694

# continue coefficients for nemenyi 0.1
# plt.plot([1.644854, 2.052293, 2.291341, 2.459516, 2.588521, 2.692732, 2.779884, 2.854606, 2.919889,
#           2.977768, 3.029694, 3.076733, 3.119693, 3.159199, 3.195743, 3.229723, 3.261461, 3.291224, 3.319233])
# a = 1.65 + 1/1.76 * np.log(range(1, 25))
# plt.plot(a)
# print(a)  # 3.35212061 3.37984229 3.40627412 3.4315308  3.4557124

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


class SBM_Data(Data):
    SBM_RESULTS_ROOT = '../../cache/kkmeans_init_sbm'

    sbm_columns = [
        (100, 2, 0.05, 0.001), (100, 2, 0.05, 0.002), (100, 2, 0.05, 0.005), (100, 2, 0.05, 0.007),
        (100, 2, 0.05, 0.01), (100, 2, 0.05, 0.02), (100, 2, 0.05, 0.03), (100, 2, 0.05, 0.05),

        (100, 2, 0.1, 0.001), (100, 2, 0.1, 0.002), (100, 2, 0.1, 0.005), (100, 2, 0.1, 0.01),
        (100, 2, 0.1, 0.02), (100, 2, 0.1, 0.03), (100, 2, 0.1, 0.05), (100, 2, 0.1, 0.07),
        (100, 2, 0.1, 0.1),

        (100, 2, 0.1, 0.001, 1.0), (100, 2, 0.1, 0.005, 1.0), (100, 2, 0.1, 0.01, 1.0), (100, 2, 0.1, 0.02, 1.0),
        (100, 2, 0.1, 0.05, 1.0), (100, 2, 0.1, 0.1, 1.0),

        (100, 2, 0.1, 0.001, 2), (100, 2, 0.1, 0.005, 2), (100, 2, 0.1, 0.01, 2),

        (100, 2, 0.15, 0.01), (100, 2, 0.15, 0.03), (100, 2, 0.15, 0.05), (100, 2, 0.15, 0.07),
        (100, 2, 0.15, 0.1), (100, 2, 0.15, 0.15),

        (100, 2, 0.2, 0.05), (100, 2, 0.2, 0.1), (100, 2, 0.2, 0.15),

        (100, 2, 0.3, 0.05), (100, 2, 0.3, 0.1), (100, 2, 0.3, 0.15),

        (102, 3, 0.1, 0.001), (102, 3, 0.1, 0.005), (102, 3, 0.1, 0.01), (102, 3, 0.1, 0.02),
        (102, 3, 0.1, 0.03), (102, 3, 0.1, 0.05), (102, 3, 0.1, 0.1),

        (102, 3, 0.3, 0.05), (102, 3, 0.3, 0.1), (102, 3, 0.3, 0.15),

        (100, 4, 0.1, 0.001), (100, 4, 0.1, 0.005), (100, 4, 0.1, 0.01), (100, 4, 0.1, 0.02),
        (100, 4, 0.1, 0.05), (100, 4, 0.1, 0.1),

        (100, 4, 0.3, 0.1), (100, 4, 0.3, 0.15),

        (150, 2, 0.1, 0.001), (150, 2, 0.1, 0.005), (150, 2, 0.1, 0.01), (150, 2, 0.1, 0.02),
        (150, 2, 0.1, 0.03), (150, 2, 0.1, 0.05), (150, 2, 0.1, 0.1),

        (150, 3, 0.1, 0.001), (150, 3, 0.1, 0.005), (150, 3, 0.1, 0.01), (150, 3, 0.1, 0.02),
        (150, 3, 0.1, 0.03), (150, 3, 0.1, 0.05), (150, 3, 0.1, 0.1),

        (200, 2, 0.1, 0.001), (200, 2, 0.1, 0.005), (200, 2, 0.1, 0.01),

        (200, 2, 0.3, 0.05), (200, 2, 0.3, 0.1), (200, 2, 0.3, 0.15),

        (201, 3, 0.3, 0.1),

        (200, 4, 0.1, 0.001), (200, 4, 0.1, 0.005), (200, 4, 0.1, 0.01), (200, 4, 0.1, 0.02),
        (200, 4, 0.1, 0.03), (200, 4, 0.1, 0.05), (200, 4, 0.1, 0.1),

        (200, 4, 0.3, 0.1), (200, 4, 0.3, 0.15),

        (500, 2, 0.1, 0.001), (500, 2, 0.1, 0.005), (500, 2, 0.1, 0.01), (500, 2, 0.1, 0.02),
        (500, 2, 0.1, 0.05)
    ]

    def __init__(self):
        super().__init__()
        self.datasets = [self.column2str(x) for x in self.sbm_columns]

    @staticmethod
    def column2str(column):
        if len(column) == 4:
            n, k, p_in, p_out = column
            return f'{n}_{k}_{p_in:.2f}_{p_out:.3f}'
        else:
            n, k, p_in, p_out, unbalance = column
            return f'{n}_{k}_{p_in:.2f}_{p_out:.3f}_{unbalance:.2f}'

    @staticmethod
    def str2column(column_str):
        split = column_str.split('_')
        n, k, p_in, p_out = int(split[0]), int(split[1]), float(split[2]), float(split[3])
        unbalance = 0 if len(column_str.split('_')) == 4 else float(split[4])
        return n, k, unbalance, p_in, p_out

    def load_precalculated(self):
        with open(f'{self.CACHE_ROOT}/sbm_inits_bestparam_byari_individual.pkl', 'rb') as f:
            results = pickle.load(f)  # {(dataset, kernel_name, graph_idx): {scorename_initname: best_ari}}
        with open(f'{self.CACHE_ROOT}/sbm_modularity.pkl', 'rb') as f:
            modularity_results = pickle.load(f)  # {(dataset, graph_idx): modularity}

        for key in list(results.keys()):
            if key[0] not in self.datasets:
                del results[key]

        for key in list(modularity_results.keys()):
            if key[0] not in self.datasets:
                del modularity_results[key]

        # {dataset: {graphidx: {kernel_name: best_ari}}}
        results_modularity_any3 = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for (dataset, kernel_name, graph_idx), si_ari in results.items():
            results_modularity_any3[dataset][graph_idx][kernel_name] = si_ari['modularity_any3']

        return results, results_modularity_any3, modularity_results

    def extract_feature(self, column_str, feature, G=None, partition=None, sp=None, max_clique=None):
        # graph-independent features
        n, k, unbalance, p_in, p_out = self.str2column(column_str)
        if feature == 'n':
            return n
        elif feature == 'k':
            return k
        elif feature == 'unbalance':
            return unbalance
        elif feature == 'p_in':
            return p_in
        elif feature == 'p_out':
            return p_out
        elif feature == 'n/k':
            return n / k
        elif feature == 'p_in/p_out':
            return p_in / p_out

        elif feature == 'log(n)':
            return n
        elif feature == 'log(k)':
            return k
        elif feature == 'log(p_in)':
            return p_in
        elif feature == 'log(p_out)':
            return p_out
        elif feature == 'log(n/k)':
            return n / k
        elif feature == 'log(p_in/p_out)':
            return p_in / p_out

        elif feature == 'n/k * p_in/p_out':
            return (n / k) * (p_in / p_out)
        elif feature == 'log(n)/k * p_in/p_out':
            return np.log(n) / k * (p_in / p_out)
        elif feature == 'log(n/k) * p_in/p_out':
            return np.log(n / k) * (p_in / p_out)
        elif feature == 'log(n/k * p_in/p_out)':
            return np.log((n / k) * (p_in / p_out))

        elif feature == 'sbm_neighbour_score':
            return sbm_neighbour_score(int(n), int(k), p_in, p_out)

        # graph-dependant features
        elif feature == 'modularity':
            return nx.community.modularity(G, partition)
        elif feature == 'diameter':
            return nx.diameter(G)
        elif feature == 'density':
            return nx.density(G)
        elif feature == 'avg_deg':
            return np.mean(G.degree)
        elif feature == 'std_deg':
            return np.std(G.degree)
        elif feature == 'avg(deg | deg > avg_deg)':
            deg = np.array(G.degree)
            return np.mean(deg[deg > np.mean(deg)])
        elif feature == 'median_deg':
            return np.median(G.degree)
        elif feature == 'max_deg':
            return np.max(G.degree)
        elif feature == 'avg_sp':
            return nx.average_shortest_path_length(G)
        elif feature == 'std_sp':
            return np.std(sp)
        elif feature == 'median_sp':
            return np.median(sp)
        elif feature == 'max_sp':
            return np.max(sp)
        elif feature == 'max_clique':
            return max_clique
        elif feature == 'max_clique/(n/k)':
            return max_clique / (n / k)
        elif feature == 'class_idx':
            for class_idx, datasetss in enumerate(self.datasets_partition):
                if column_str in datasetss:
                    return class_idx
        else:
            raise Exception()

    def prepare_column(self, results_modularity_any3, column):
        @load_or_calc_and_save(f'{self.CACHE_ROOT}/feature_importance/{column}.pkl')
        def wrapper():
            X, ya = [], []
            filename = f'{column}_100_graphs.pkl'
            with open(f'{self.SBM_RESULTS_ROOT}/graphs/{filename}', 'rb') as f:
                data = pickle.load(f)
            for graph_idx in range(100):
                try:
                    (A, y_true), _ = data[graph_idx]
                except:
                    (A, y_true) = data[graph_idx]
                G = nx.from_numpy_matrix(A)
                partition = ytrue_to_partition(y_true)
                sp = [l for u in G for v, l in nx.single_source_shortest_path_length(G, u).items()]
                max_clique = len(clique.max_clique(G))

                X.append({feature_name: self.extract_feature(column, feature_name, G, partition, sp, max_clique)
                          for feature_name in self.feature_names})
                ya.append(results_modularity_any3[column][graph_idx])
            return X, ya

        return wrapper()


class Datasets_Data(Data):
    DATASETS_RESULTS_ROOT = '../../cache/kkmeans_init_datasets'

    datasets = [
        'cora_DB', 'cora_EC', 'cora_HA', 'cora_HCI', 'cora_IR', 'cora_Net',
        'dolphins',
        'eu-core',
        'eurosis',
        'football',
        'karate',
        'news_2cl1_0.1', 'news_2cl2_0.1', 'news_2cl3_0.1',
        'news_3cl1_0.1', 'news_3cl2_0.1', 'news_3cl3_0.1',
        'news_5cl1_0.1', 'news_5cl2_0.1', 'news_5cl3_0.1',
        'polblogs',
        'polbooks',
        'sp_school_day_1', 'sp_school_day_2'
    ]

    actual_graphs = Datasets()

    def load_precalculated(self):
        with open(f'{self.CACHE_ROOT}/datasets_inits_bestparam_byari_individual_0.1.pkl', 'rb') as f:
            results = pickle.load(f)
        with open(f'{self.CACHE_ROOT}/datasets_modularity_0.1.pkl', 'rb') as f:
            modularity_results = pickle.load(f)

        for key in list(results.keys()):
            if key[0] not in self.datasets:
                del results[key]

        for key in list(results.keys()):
            if key[0] not in self.datasets:
                del modularity_results[key]

        # {dataset: {graphidx: {kernel_name: best_ari}}}
        results_modularity_any3 = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for (dataset, kernel_name, graph_idx), si_ari in results.items():
            results_modularity_any3[dataset][graph_idx][kernel_name] = si_ari['modularity_any3']

        return results, results_modularity_any3, modularity_results

    def extract_feature(self, dataset_name, feature, G=None, partition=None, sp=None, max_clique=None):
        # graph-independent features
        (A, y_true), info = self.actual_graphs[dataset_name]
        n, k, unbalance, p_in, p_out = info['n'], info['k'], info['unbalance'], info['p_in'], info['p_out']
        if feature == 'n':
            return n
        elif feature == 'k':
            return k
        elif feature == 'unbalance':
            return unbalance
        elif feature == 'p_in':
            return p_in
        elif feature == 'p_out':
            return p_out
        elif feature == 'n/k':
            return n / k
        elif feature == 'p_in/p_out':
            return p_in / p_out

        elif feature == 'log(n)':
            return n
        elif feature == 'log(k)':
            return k
        elif feature == 'log(p_in)':
            return p_in
        elif feature == 'log(p_out)':
            return p_out
        elif feature == 'log(n/k)':
            return n / k
        elif feature == 'log(p_in/p_out)':
            return p_in / p_out

        elif feature == 'n/k * p_in/p_out':
            return (n / k) * (p_in / p_out)
        elif feature == 'log(n)/k * p_in/p_out':
            return np.log(n) / k * (p_in / p_out)
        elif feature == 'log(n/k) * p_in/p_out':
            return np.log(n / k) * (p_in / p_out)
        elif feature == 'log(n/k * p_in/p_out)':
            return np.log((n / k) * (p_in / p_out))

        elif feature == 'sbm_neighbour_score':
            return sbm_neighbour_score(int(n), int(k), p_in, p_out)
        elif feature == 'graph_neighbour_score':
            return graph_neighbour_score(A, y_true, weighting='proportional')
        elif feature == 'graph_neighbour_score_noweight':
            return graph_neighbour_score(A, y_true, weighting='noweight')

        # graph-dependant features
        elif feature == 'modularity':
            partition = ytrue_to_partition(y_true)
            return nx.community.modularity(G, partition)
        elif feature == 'diameter':
            return nx.diameter(G)
        elif feature == 'density':
            return nx.density(G)
        elif feature == 'avg_deg':
            return np.mean(G.degree)
        elif feature == 'std_deg':
            return np.std(G.degree)
        elif feature == 'avg(deg | deg > avg_deg)':
            deg = np.array(G.degree)
            return np.mean(deg[deg > np.mean(deg)])
        elif feature == 'median_deg':
            return np.median(G.degree)
        elif feature == 'max_deg':
            return np.max(G.degree)
        elif feature == 'avg_sp':
            return nx.average_shortest_path_length(G)
        elif feature == 'std_sp':
            return np.std(sp)
        elif feature == 'median_sp':
            return np.median(sp)
        elif feature == 'max_sp':
            return np.max(sp)
        elif feature == 'max_clique':
            return max_clique
        elif feature == 'max_clique/(n/k)':
            return max_clique / (n / k)
        elif feature == 'class_idx':
            for class_idx, datasetss in enumerate(self.datasets_partition):
                if dataset_name in datasetss:
                    return class_idx
        else:
            raise Exception()

    def prepare_column(self, results_modularity_any3, dataset_name):
        @load_or_calc_and_save(f'{self.CACHE_ROOT}/feature_importance/{dataset_name}.pkl')
        def wrapper():
            X, ya = [], []
            (A, partition), info = self.actual_graphs[dataset_name]
            G = nx.from_numpy_matrix(A)
            partition = ytrue_to_partition(partition)
            sp = [l for u in G for v, l in nx.single_source_shortest_path_length(G, u).items()]
            max_clique = len(clique.max_clique(G))
            features = {
                feature_name: self.extract_feature(dataset_name, feature_name, G, partition, sp, max_clique)
                for feature_name in self.feature_names}
            for graph_idx in range(7):
                graph_ari = results_modularity_any3[dataset_name][graph_idx]
                X.append(features)
                ya.append(graph_ari)
            return X, ya

        return wrapper()


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


shuffle = lambda x: sorted(x, key=lambda k: random.random())


def load_or_calc_and_save(filename, force_calc=False, ignore_if_exist=False):
    def decorator(func):
        def wrapped(*args, **kwargs):
            if os.path.exists(filename) and not force_calc:
                #                 print(f'{func.__name__}: cache file {filename} found! Skip calculations')
                if not ignore_if_exist:
                    with open(filename, 'rb') as f:
                        result = pickle.load(f)
                else:
                    result = None
            else:
                print(
                    f'{func.__name__}: RECALC {filename}. args: {", ".join(args)}, kwargs: {", ".join([f"{k}={v}" for k, v in kwargs.items()])}')
                result = func(*args, **kwargs)
                with open(filename, 'wb') as f:
                    pickle.dump(result, f)
            return result

        return wrapped

    return decorator


def dict_argmax(dct, score_key):
    best_key = list(dct.keys())[0]
    best_val = dct[best_key]
    for k, v in dct.items():
        if v[score_key] > best_val[score_key]:
            best_key, best_val = k, v
    return best_key, best_val


def calc_avranks(results):  # {dataset: {classifier: accuracy}}
    ranks = defaultdict(list)
    for _, classifier_accuracy in results.items():
        classifiers, accuracies = zip(*list(classifier_accuracy.items()))
        for classifier, rank in zip(classifiers, rankdata(accuracies)):
            ranks[classifier].append(rank)
    ranks = {k: np.mean(v) for k, v in sorted(ranks.items(), key=lambda x: x[0])}
    return list(ranks.values()), list(ranks.keys()), len(results)


def ytrue_to_partition(y_true):
    partition = defaultdict(list)
    for idx, class_ in enumerate(y_true):
        partition[class_].append(idx)
    return list(partition.values())


def best_inits(param_results):
    inits = {
        'one': shuffle([x for x in param_results if x['init'] == 'one']),
        'all': shuffle([x for x in param_results if x['init'] == 'all']),
        'kmp': shuffle([x for x in param_results if x['init'] == 'k-means++'])
    }
    if len(inits['one']) == 0 or len(inits['all']) == 0 or len(inits['kmp']) == 0:
        return None
    inits['any2'] = inits['all'][:15] + inits['kmp'][:15]
    inits['any3'] = inits['one'][:10] + inits['all'][:10] + inits['kmp'][:10]

    # choose best init, structure – {scorename_initname: init}
    score_names = ['inertia', 'modularity']
    bestby = {f'{score_name}_{init_name}': inits[0]
              for score_name, (init_name, inits) in product(score_names, inits.items())}
    for init_name, inits in inits.items():
        for init in inits:
            if init['inertia'] < bestby[f'inertia_{init_name}']['inertia']:
                bestby[f'inertia_{init_name}'] = init
            if init['modularity'] > bestby[f'modularity_{init_name}']['modularity']:
                bestby[f'modularity_{init_name}'] = init
    return bestby


def perform_graph(graph_results):
    # group params, structure – {scorename_initname: {param: init}}
    bestby = defaultdict(dict)
    for param, param_results in graph_results.items():
        param_inits = best_inits(param_results)
        if param_inits is not None:
            for scorename_initname, init in param_inits.items():
                bestby[scorename_initname][param] = init

    # collapse param axis, structure – {scorename_initname: (best_param, init)}
    best_ari = {si: dict_argmax(param_init, 'score_ari') for si, param_init in bestby.items()}
    best_modularity = {si: dict_argmax(param_init, 'modularity') for si, param_init in bestby.items()}

    return bestby, best_ari, best_modularity


def meanby_graphs__bestparam_individual(data, param_by_ari=True, target_score='score_ari'):
    # group graphs, structure – {scorename_initname: [best_ari_for_graph]}
    bestby = defaultdict(list)
    for graph_idx, graph_results in enumerate(data):
        _, best_ari, best_modularity = perform_graph(graph_results['results'])
        graph_best = best_ari if param_by_ari else best_modularity
        for scorename_initname, (_, init) in graph_best.items():
            bestby[scorename_initname].append(init[target_score])

    # collapse graph axis, structure - {scorename_initname: mean_ari}
    best_meanari = {si: np.mean(ari) for si, ari in bestby.items()}

    return bestby, best_meanari


def meanby_graphs__allparams(data, target_score='score_ari'):
    # group graphs, structure – {scorename_initname: {param: [best_ari_for_graph]}}
    bestby = defaultdict(lambda: defaultdict(list))
    for graph_idx, graph_results in enumerate(data):
        graph_bestby, _, _ = perform_graph(graph_results['results'])
        for scorename_initname, param_init in graph_bestby.items():
            for param, init in param_init.items():
                bestby[scorename_initname][param].append(init[target_score])

    # collapse graph axis, structure – {scorename_initname: {param: mean_ari}}
    best_meanari = {si: {param: np.mean(ari) for param, ari in param_ari.items()} for si, param_ari in bestby.items()}

    return bestby, best_meanari


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

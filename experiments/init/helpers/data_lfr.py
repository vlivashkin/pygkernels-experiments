import pickle
import sys
from collections import defaultdict

import networkx as nx
import numpy as np
from networkx.algorithms.approximation import clique

from helpers.data import Data
from helpers.sbm_neighbour_score import sbm_neighbour_score
from helpers.utils import load_or_calc_and_save, ytrue_to_partition

sys.path.append('../../pygkernels')
from pygkernels.data.dataset import Datasets


class SBM_Data(Data):
    SBM_RESULTS_ROOT = '../../cache/kkmeans_init_lfrkkmeans_init_lfr'

    sbm_columns = [
        'dolphins',
        'football',
        'karate',
        'polbooks',
        'sp_school_day_1', 'sp_school_day_2',
        'news_2cl1_0.1', 'news_2cl2_0.1', 'news_2cl3_0.1',
        'news_3cl1_0.1', 'news_3cl2_0.1', 'news_3cl3_0.1',
        'news_5cl1_0.1', 'news_5cl2_0.1', 'news_5cl3_0.1',
        'polblogs',
        'cora_DB', 'cora_EC', 'cora_HA', 'cora_HCI', 'cora_IR', 'cora_Net',
        'eu-core',
        'eurosis'
    ]

    def __init__(self):
        super().__init__()
        self.datasets = [self.column2str(x) for x in self.sbm_columns]
        self.datasets_source = Datasets()

    @staticmethod
    def column2str(column):
        return f'dataset2lfr_{column}'

    def str2column(self, column_str):
        column = column_str[12:]
        _, info = self.datasets_source[column]
        return info['n'], info['k'], None, info['p_in'], info['p_out']

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
            return sbm_neighbour_score(int(n), int(k), cluster_sizes=info['S'], p=info['P'])

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
            filename = f'{column}_graphs.pkl'
            with open(f'{self.SBM_RESULTS_ROOT}/graphs/{filename}', 'rb') as f:
                data = pickle.load(f)
            for graph_idx in range(10):
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

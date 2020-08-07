import pickle
import sys
from collections import defaultdict

import networkx as nx
import numpy as np
from networkx.algorithms.approximation import clique

from helpers.data import Data
from helpers.sbm_neighbour_score import sbm_neighbour_score, graph_neighbour_score
from helpers.utils import ytrue_to_partition, load_or_calc_and_save

sys.path.append('../../pygkernels')
from pygkernels.data.dataset import Datasets


class Datasets_Data(Data):
    DATASETS_RESULTS_ROOT = '../../cache/kkmeans_init_datasets'

    datasets = [
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
            return sbm_neighbour_score(int(n), int(k), p_in=p_in, p_out=p_out)
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

import os
import argparse
import sys
from functools import partial
from typing import Type

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from tqdm import tqdm

sys.path.append('../../pygkernels')
from pygkernels.cluster import KKMeans
from pygkernels.data import StochasticBlockModel
from pygkernels.measure import kernels, Kernel
from pygkernels.score import sns1
from pygkernels.util import load_or_calc_and_save

"""
For every column and measure, we calculate [ ] in parallel for every graph.
[ ]: for every param we calculate inits with scores
"""

CACHE_ROOT = '../../cache/kkmeans_init_sbm'
# CACHE_ROOT = 'cache/kkmeans_init_sbm'
columns = [
    (100, 2, 0.1, 0.001),
    (100, 2, 0.1, 0.005),
    (100, 2, 0.1, 0.01),
    (100, 2, 0.1, 0.02),
    (100, 2, 0.1, 0.05),
    (100, 2, 0.1, 0.1),
    
    (102, 3, 0.1, 0.001),
    (102, 3, 0.1, 0.005),
    (102, 3, 0.1, 0.01),
    (102, 3, 0.1, 0.02),
    (102, 3, 0.1, 0.05),
    (102, 3, 0.1, 0.1),
    
    (150, 3, 0.1, 0.001),
    (150, 3, 0.1, 0.005),
    (150, 3, 0.1, 0.01),
    (150, 3, 0.1, 0.02),
    (150, 3, 0.1, 0.05),
    (150, 3, 0.1, 0.1),

    (100, 4, 0.1, 0.001),
    (100, 4, 0.1, 0.005),
    (100, 4, 0.1, 0.01),
    (100, 4, 0.1, 0.02),
    (100, 4, 0.1, 0.05),
    (100, 4, 0.1, 0.1),
    
    (200, 4, 0.1, 0.001),
    (200, 4, 0.1, 0.005),
    (200, 4, 0.1, 0.01),
    (200, 4, 0.1, 0.02),
    (200, 4, 0.1, 0.05),
    (200, 4, 0.1, 0.1),
]


def generate_graphs(column, n_graphs, root=f'{CACHE_ROOT}/graphs'):
    n, k, p_in, p_out = column
    column_str = f'{n}_{k}_{p_in:.2f}_{p_out:.3f}'

    @load_or_calc_and_save(f'{root}/{column_str}_{n_graphs}_graphs.pkl')
    def _calc(n_graphs=n_graphs, n_params=None, n_jobs=None):
        graphs, _ = StochasticBlockModel(n, k, p_in=p_in, p_out=p_out).generate_graphs(n_graphs, verbose=True)
        return graphs

    return _calc(n_graphs=n_graphs, n_params=None, n_jobs=None)


def perform_graph(graph, kernel_class: Type[Kernel], estimator: KKMeans, n_params, graph_idx):
    try:
        (A, y_true), G = graph
    except:
        A, y_true = graph

    flat_params = np.linspace(0, 1, n_params)
    results = {param_flat: [] for param_flat in flat_params}

    try:
        kernel: Kernel = kernel_class(A)
        for param_flat in flat_params:
            try:
                param = kernel.scaler.scale(param_flat)
                K = kernel.get_K(param)
                inits = estimator.predict(K, explicit=True, A=A)

                param_results = []
                for init in inits:
                    y_pred = init['labels']
                    param_results.append({
                        'labels': y_pred,
                        'inertia': init['inertia'],
                        'modularity': init['modularity'],
                        'init': init['init'],
                        'score_ari': adjusted_rand_score(y_true, y_pred),
                        'score_nmi': normalized_mutual_info_score(y_true, y_pred, average_method='geometric'),
                        'score_sns1': sns1(y_true, y_pred)
                    })
                results[param_flat] = param_results
            except Exception or ValueError or FloatingPointError or np.linalg.LinAlgError as e:
                print(f'{kernel_class.name}, g={graph_idx}, p={param_flat}: {e}')
    except Exception or ValueError or FloatingPointError or np.linalg.LinAlgError as e:
        print(f'{kernel_class.name}, g={graph_idx}, ALL PARAMS: {e}')

    return {
        'results': results,
        'y_true': y_true
    }


def perform_kernel(column, graphs, kernel_class, n_params, n_jobs, n_gpu, root=f'{CACHE_ROOT}/by_column'):
    n, k, p_in, p_out = column
    column_str = f'{n}_{k}_{p_in:.2f}_{p_out:.3f}'

    try:
        os.mkdir(f'{root}/{column_str}')
    except:
        pass
        
    @load_or_calc_and_save(f'{root}/{column_str}/{column_str}_{kernel_class.name}_results.pkl', ignore_if_exist=True)
    def _calc(n_graphs=None, n_params=n_params, n_jobs=n_jobs):
        kmeans = partial(KKMeans, n_clusters=k, init='any', n_init=N_INITS, init_measure='modularity')
        return Parallel(n_jobs=n_jobs)(delayed(perform_graph)(
            graph, kernel_class, kmeans(device=graph_idx % n_gpu, random_state=2000 + graph_idx), n_params=n_params,
            graph_idx=graph_idx
        ) for graph_idx, graph in enumerate(graphs))

    return _calc(n_graphs=None, n_params=n_params, n_jobs=n_jobs)


def perform_column(column, graphs):
    n, k, p_in, p_out = column
    column_str = f'{n}_{k}_{p_in:.2f}_{p_out:.3f}'
    for kernel_class in tqdm(kernels, desc=column_str):
        perform_kernel(column, graphs, kernel_class, n_params=N_PARAMS, n_jobs=N_JOBS, n_gpu=N_GPU)


def perform(n_graphs):
    for column in columns:
        graphs = generate_graphs(column, n_graphs)
        perform_column(column, graphs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_jobs', type=int, default=6, required=False)
    parser.add_argument('--n_gpu', type=int, default=2, required=False)
    parser.add_argument('--n_graphs', type=int, default=100, required=False)
    parser.add_argument('--n_inits', type=int, default=30, required=False)
    parser.add_argument('--n_params', type=int, default=51, required=False)

    args = parser.parse_args()
    print(args)
    
    print('Checking CACHE_ROOT to be available...')
    with open(f'{CACHE_ROOT}/check', 'w') as f:
        f.write('ok\n')
    print('Ok!')

    N_JOBS = args.n_jobs
    N_GPU = args.n_gpu
    N_GRAPHS = args.n_graphs
    N_INITS = args.n_inits
    N_PARAMS = args.n_params
    perform(n_graphs=N_GRAPHS)

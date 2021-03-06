import multiprocessing
import pickle
import sys
import time
import os
import json

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from tqdm import tqdm

sys.path.append('../../pygkernels')
from pygkernels.cluster import KKMeans
from pygkernels.measure import kernels, Kernel
from pygkernels.score import sns1
from pygkernels.data import LFRGenerator


def create_krondecker(partition):
    n = len(partition)
    kron_mask = np.tile(partition, n) == np.repeat(partition, n)
    return np.reshape(kron_mask, (n, n))

def modularity2(AIJ, partition):
    n = len(AIJ)
    m = np.sum(AIJ)  # no of edges

    k = np.sum(AIJ, axis=1)
    expectation = np.reshape(np.tile(k, n) * np.repeat(k, n), (n, n)) / m
    kron = create_krondecker(partition)

    # Q = (1 / 2m) * SUM(AIJ - (ki.kj / 2m)) ∂(ci, cj)
    return (1.0 / m) * np.sum(kron * (AIJ - expectation))


def randfloat(low, high, distribution='linear'):
    if distribution == 'linear':
        delta = high - low
        return low + np.random.random() * delta
    elif distribution == 'power':
        x = np.random.random()
        return np.clip(1 / (1 - x), low, high)
    elif distribution == 'powersqrt':
        x = np.random.random()
        return np.clip(1 / (1 - x)**(1/5), low, high)

def generate_params():
    while True:
        np.random.seed(None)
        n = np.random.randint(10, 1500)
        tau1 = randfloat(1, 100, 'power')
        tau2 = randfloat(1, 200, 'power')
        mu = randfloat(.0, 1.)
        avg_degree = randfloat(.0, n)
        min_community = np.random.randint(1, n)
        
        min_distances = sorted(euclidean_distances(np.array(prepare_info({
            'n': n, 'tau1': tau1, 'tau2': tau2, 'mu': mu, 'average_degree': avg_degree
        }))[None], existing_items)[0].tolist())[:3]
        if any([x > 0.15 for x in min_distances]):
            return n, tau1, tau2, mu, avg_degree, min_community


def generate_graph(return_dict):
    # print("Inside process generate_graph")
    try:
        n, tau1, tau2, mu, avg_degree, min_community = generate_params()
        graphs, info = LFRGenerator(n, tau1, tau2, mu, average_degree=avg_degree, min_community=min_community) \
            .generate_graphs(n_graphs=N_GRAPHS, is_connected=True)
        A, y_true = graphs[0]
        return_dict['result'] = A, y_true, info
        # print("SUCCESS! generate_graph")
        time.sleep(50)
    except:
        # print('Failure to create graph')
        pass
    return True


def generate_proper_graph():
    attempt_no = 1
    result = None
    while result is None:
        print(f'attempt_no: {attempt_no}')
        manager = multiprocessing.Manager()
        return_dict = manager.dict()

        proc = multiprocessing.Process(target=generate_graph, args=(return_dict,))
        proc.start()
        proc.join(30)
        try:
            result = return_dict['result']
        except (ConnectionRefusedError, ConnectionResetError, KeyError) as e:
            # print('Error:', e)
            attempt_no += 1
        if proc.is_alive():
            # print('Terminate')
            proc.terminate()
            
    return result


def endless_generator():
    while True:
        yield True


def perform_graph():
    print('Generate graph...')
    A, y_true, info = generate_proper_graph()

    print('Start evaluate!')
    print(info)

    flat_params = np.linspace(0, 1, N_PARAMS)
    all_results = {}
    for kernel_class in kernels:
        results = {param_flat: [] for param_flat in flat_params}

        try:
            kernel: Kernel = kernel_class(A)
            for param_flat in flat_params:
                try:
                    param = kernel.scaler.scale(param_flat)
                    K = kernel.get_K(param)
                    inits = KKMeans(n_clusters=info['k'], init='any', n_init=N_INITS, init_measure='modularity',
                                    device=np.random.randint(0, N_GPU)).predict(K, explicit=True, A=A)

                    param_results = []
                    for init in inits:
                        y_pred = init['labels']
                        param_results.append({
                            'labels': y_pred,
                            'inertia': init['inertia'],
                            'modularity': init['modularity'],
                            'new_modularity2': modularity2(A, y_pred),
                            'init': init['init'],
                            'score_ari': adjusted_rand_score(y_true, y_pred),
                            'score_nmi': normalized_mutual_info_score(y_true, y_pred, average_method='geometric'),
                            'score_sns1': sns1(y_true, y_pred)
                        })
                    results[param_flat] = param_results
                except Exception or ValueError or FloatingPointError or np.linalg.LinAlgError as e:
                    print(f'{kernel_class.name}, p={param_flat}: {e}')
        except Exception or ValueError or FloatingPointError or np.linalg.LinAlgError as e:
            print(f'{kernel_class.name}, ALL PARAMS: {e}')

        all_results[kernel_class.name] = results

    fn = f'{info["n"]}_{info["tau1"]:.2f}_{info["tau2"]:.2f}_{info["mu"]:.2f}_{info["average_degree"]:.2f}_{info["min_community"]:.2f}'
    with open(f'/media/illusionww/68949C3149F4E819/phd/pygkernels/montecarlo_lfr/{fn}.pkl', 'wb') as f:
        pickle.dump({
            'results': all_results,
            'A': A,
            'y_true': y_true,
            'info': info
        }, f)


def prepare_info(info):
    n = info['n'] / 1500
    tau1 = 1 - (1 / info['tau1'])
    tau2 = 1 - (1 / info['tau2'])
    mu = info['mu']
    density = info['average_degree'] / (info['n'] - 1)
    return [n, tau1, tau2, mu, density]


def euclidean_distances(X, Y):
    X, Y = np.array(X), np.array(Y)
    result = np.sqrt(np.sum(np.power(X[:, None] - Y[None, :], 2), axis=2))
    return result


if __name__ == '__main__':
    N_JOBS = 6
    N_GPU = 2
    N_GRAPHS = 1
    N_INITS = 6
    N_PARAMS = 16
    
    existing_items = []
    with open('all_dataset.json', 'r') as f:
        for info in json.load(f):
            existing_items.append(prepare_info(info))
    print(f'{len(existing_items)} graphs are already exist...')

    Parallel(n_jobs=N_JOBS)(delayed(perform_graph)() for _ in endless_generator())

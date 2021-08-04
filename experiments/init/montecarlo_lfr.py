import json
import multiprocessing as mp
import pickle
import sys
import time
import traceback

import numpy as np
from gpuparallel import GPUParallel, delayed, log_to_stderr
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

sys.path.append('../../pygkernels')
from pygkernels.cluster import KKMeans
from pygkernels.measure import kernels, Kernel
from pygkernels.score import sns1
from pygkernels.data import LFRGenerator

import joblib
from joblib.externals import loky
print(joblib.__version__)
print(loky.__version__)

log = mp.get_logger()
log_to_stderr(log_level='WARNING')

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
        tau1 = randfloat(1, 4, 'linear')
        tau2 = randfloat(1, 200, 'power')
        mu = randfloat(.0, .5)
        avg_degree = randfloat(.0, n)
        min_community = np.random.randint(1, n)
        
        min_distances = sorted(euclidean_distances(np.array(prepare_info({
            'n': n, 'tau1': tau1, 'tau2': tau2, 'mu': mu, 'average_degree': avg_degree
        }))[None], existing_items)[0].tolist())[:3]
        if any([x > 0.11 for x in min_distances]):
            return n, tau1, tau2, mu, avg_degree, min_community


def generate_graph(return_dict):
    # log.warning("Inside process generate_graph")
    try:
        n, tau1, tau2, mu, avg_degree, min_community = generate_params()
        graphs, info = LFRGenerator(n, tau1, tau2, mu, average_degree=avg_degree, min_community=min_community) \
            .generate_graphs(n_graphs=N_GRAPHS, is_connected=True)
        A, y_true = graphs[0]
        return_dict['result'] = A, y_true, info
        # log.warning("SUCCESS! generate_graph")
        time.sleep(50)
    except:
        # log.warning('Failure to create graph')
        pass
    return True


def generate_proper_graph():
    attempt_no = 1
    result = None
    while result is None:
        log.warning(f'attempt_no: {attempt_no}')
        manager = mp.Manager()
        return_dict = manager.dict()

        proc = mp.Process(target=generate_graph, args=(return_dict,))
        proc.start()
        proc.join(30)
        try:
            result = return_dict['result']
        except (ConnectionRefusedError, ConnectionResetError, KeyError) as e:
            # log.warning('Error:', e)
            attempt_no += 1
        if proc.is_alive():
            # log.warning('Terminate')
            proc.terminate()
            
    return result


def endless_generator():
    while True:
        yield True


def perform_graph(device_id, **kwargs):
    log.warning('Generate graph...')
    A, y_true, info = generate_proper_graph()

    log.warning('Start evaluate!')
    log.warning(info)

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
                                    device=device_id).predict(K, explicit=True, A=A)

                    param_results = []
                    for init in inits:
                        y_pred = init['labels']
                        assert y_pred is not None
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
                    log.warning(f'{kernel_class.name}, p={param_flat}: {e}')
                    traceback.print_exc()
        except Exception or ValueError or FloatingPointError or np.linalg.LinAlgError as e:
            log.warning(f'{kernel_class.name}, ALL PARAMS: {e}')

        all_results[kernel_class.name] = results

    fn = f'{info["n"]}_{info["tau1"]:.2f}_{info["tau2"]:.2f}_{info["mu"]:.2f}_{info["average_degree"]:.2f}_{info["min_community"]:.2f}'
    with open(f'/data/phd/pygkernels/montecarlo_lfr/{fn}.pkl', 'wb') as f:
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
    N_JOBS = 8
    N_GPU = 2
    N_GRAPHS = 1
    N_INITS = 6
    N_PARAMS = 16
    
    existing_items = []
    with open('all_dataset.json', 'r') as f:
        for info in json.load(f):
            existing_items.append(prepare_info(info))
    log.warning(f'{len(existing_items)} graphs are already exist...')

    list(GPUParallel(n_gpu=N_GPU, n_workers_per_gpu=5)(delayed(perform_graph)() for _ in endless_generator()))

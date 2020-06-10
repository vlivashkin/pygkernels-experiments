import math
import numpy as np
import operator as op
from functools import reduce
from collections import Counter

def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom

def pe(e, size, p):
    """
    Вероятность иметь e связей из size
    """
    assert 0 <= e <= size
    return ncr(size, e) * p**e * (1-p)**(size-e)
    
def p_interval(l, h, size, p):
    return np.sum([pe(e, size, p) for e in range(l, h+1)])

def sbm_neighbour_score(n, k, p_in, p_out):
    size = n // k
    sum_ = 0
    for e in range(size):
        sum_ += pe(e, size-1, p_in) * (p_interval(0, e-1, size, p_out) ** (k - 1))
    return sum_

def graph_neighbour_score(A, partition):
    # calculate cluster_sizes and all p
    n_clusters = len(set(partition))
    cluster_sizes = dict(Counter(partition))
    class_mapping = {v: k for k, v in enumerate(cluster_sizes.keys())}
    cluster_sizes = list(cluster_sizes.values())
    
    incidence = np.zeros((n_clusters, n_clusters))
    for i in range(A.shape[0]):
        ki = class_mapping[partition[i]]
        for j in range(i + 1, A.shape[1]):
            if A[i, j] == 1:
                kj = class_mapping[partition[j]]
                if ki == kj:
                    incidence[ki, ki] += 1
                else:
                    incidence[ki, kj] += 1
                    incidence[kj, ki] += 1
    
    p = np.zeros((n_clusters, n_clusters))
    for i in range(n_clusters):
        for j in range(i, n_clusters):
            if i == j:
                p_in = incidence[i, j] / (cluster_sizes[i] * (cluster_sizes[i] - 1) / 2)
                p[i, i] = p_in
            else:
                p_out = incidence[i, j] / (cluster_sizes[i] * cluster_sizes[j])
                p[i, j], p[j, i] = p_out, p_out
                
    # let's calculate score for every class, and than weight it
    score = np.zeros((n_clusters,))
    for ki in range(n_clusters):
        for e in range(cluster_sizes[ki]):
            class_score_e = pe(e, cluster_sizes[ki]-1, p[ki, ki])
            for kj in range(n_clusters):
                if ki != kj:
                    class_score_e *= p_interval(0, min(e-1, cluster_sizes[kj]), cluster_sizes[kj], p[ki, kj])
            score[ki] += class_score_e
        
            
    overall = np.sum([sc * size for sc, size in zip(score, cluster_sizes)]) / len(partition)
    return overall
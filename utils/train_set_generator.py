import os
import sys
import pickle
import itertools
import numpy as np
from scipy.io.matlab import loadmat
from scipy.sparse import csr_matrix

if __name__ == '__main__':
    dataset = 'umls'
    p_obs = 0.05
    dest = '../data/%s/' % (dataset)
    file_name = 'train_%.3f.pkl' % (p_obs)
    if dataset == 'umls':
        mat = loadmat('../data/%s/uml.mat' % (dataset))
        T = np.array(mat['Rs'], np.float32)
    elif dataset == 'nation':
        mat = loadmat('../data/%s/dnations.mat' % (dataset))
        T = np.array(mat['R'], np.float32)
    elif dataset == 'kinship':
        mat = loadmat('../data/%s/alyawarradata.mat' % (dataset))
        T = np.array(mat['Rs'], np.float32)

    T = np.swapaxes(T, 1, 2)
    T = np.swapaxes(T, 0, 1)  # [relation, entity, entity]
    T[np.isnan(T)] = 0

    n_relation, n_entity, _ = T.shape

    mask = np.zeros_like(T)
    for r_k, e_i, e_j in itertools.product(range(n_relation), range(n_entity), range(n_entity)):
        if T[r_k,e_i,e_j] and np.random.binomial(1, p_obs):
            mask[r_k, e_i, e_j] = 1

    with open(os.path.join(dest, file_name), 'wb') as f:
        pickle.dump(mask, f)

    print('Total # of observation', np.sum(mask))
    print('Total # of valid triples', np.sum(T))

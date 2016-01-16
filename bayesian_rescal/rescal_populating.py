import os
import numpy as np
from scipy.io.matlab import loadmat
from scipy.sparse import csr_matrix
import rescal

dataset = 'kinship'


if dataset == 'umls':
    mat = loadmat('../data/%s/uml.mat' % (dataset))
    T = np.array(mat['Rs'], np.float32)
elif dataset == 'nation':
    mat = loadmat('../data/%s/uml.mat' % (dataset))
    T = np.array(mat['R'], np.float32)
elif dataset == 'kinship':
    mat = loadmat('../data/%s/uml.mat' % (dataset))
    T = np.array(mat['Rs'], np.float32)

T = np.swapaxes(T, 1, 2)
T = np.swapaxes(T, 0, 1)  # [relation, entity, entity]
T[np.isnan(T)] = 0

budget = 30000
init_obs = 500
n_test = 10
n_relation, n_entity, _ = T.shape
n_dim = 10

dest = '../result/%s/rescal/' % (dataset)

if not os.path.exists(dest):
    os.makedirs(dest, exist_ok=True)

for nt in range(n_test):
    file_name = os.path.join(dest, 'init%d_rescal_n_dim_%d_%d.txt' % (init_obs, n_dim, nt))
    if not os.path.exists(file_name):
        seq = list()
        mask = np.zeros_like(T)
        X = [csr_matrix(np.zeros_like(T[k])) for k in range(n_relation)]

        for i in range(budget):
            if i > init_obs:
                A, R, f, itr, exectimes = rescal.rescal_als(X, n_dim)
            else:
                A = np.random.random([n_entity, n_dim])
                R = np.random.random([n_relation, n_dim, n_dim])

            _X = np.zeros_like(T)
            for k in range(T.shape[0]):
                _X[k] = np.dot(np.dot(A, R[k]), A.T)

            find = False
            while not find:
                _X[mask == 1] = -99999999
                next_idx = np.unravel_index(_X.argmax(), _X.shape)
                mask[next_idx] = 1
                seq.append(next_idx)
                if T[next_idx] == 1:
                    X[next_idx[0]][next_idx[1], next_idx[2]] = 1
                    find = True
                if len(seq) == budget:
                    break
            if len(seq) == budget:
                break

        with open(file_name, 'w') as f:
            for s in seq:
                f.write('%d,%d,%d\n' % (s[0], s[1], s[2]))

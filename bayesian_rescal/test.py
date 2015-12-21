import os
import sys
import pickle
import logging
import itertools
import numpy as np
from scipy.io.matlab import loadmat
from scipy.sparse import csr_matrix
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import rescal
from brescal import BayesianRescal
from seq_brescal import PFBayesianRescal


def read_train(fname, T):
    with open(fname, 'rb') as f:
        seq = pickle.load(f)

    maskT = np.zeros_like(T)
    for s in seq:
        maskT[s] = 1
    return maskT


def gen_train(T, p):
    idx = set()
    p = 0.1
    for k in range(n_relation):
        for i, j in itertools.product(range(n_entity), repeat=2):
            if T[k, i, j] and np.random.binomial(1, p):
                idx.add((k, i, j))
    return idx


if __name__ == '__main__':
    """
    Test with Kinship dataset
    """
    mat = loadmat('../data/alyawarradata.mat')
    T = np.array(mat['Rs'], np.float32)
    T = np.swapaxes(T, 1, 2)
    T = np.swapaxes(T, 0, 1)  # [relation, entity, entity]

    n_relation, n_entity, _ = T.shape
    n_dim = 10
    n_particle = 5
    n_test = 10
    max_iter = 1
    obs_var = 0.01
    unobs_var = 0.1
    p = 0.1

    s_model = sys.argv[1]

    dest = '../result/kinship/'
    if not os.path.exists(dest):
        os.makedirs(dest)

    for nt in range(n_test):
        train_file = os.path.join(dest, 'train_%.2f_%d' % (p, nt))
        if not os.path.exists(train_file):
            train_seq = gen_train(T, p)
            with open(train_file, 'wb') as f:
                pickle.dump(train_seq, f)

    for nt in range(n_test):
        train_file = os.path.join(dest, 'train_%.2f_%d' % (p, nt))
        maskT = read_train(train_file, T)

        if s_model == 'sRESCAL':
            file_name = os.path.join(dest,
                                     'sRESCAL_p_%.2f_dim_%d_par_%d_test_%d_convar_%r.pkl' % (
                                         p, n_dim, n_particle, nt, False))

            if not os.path.exists(file_name):
                model = PFBayesianRescal(n_dim, controlled_var=False, n_particles=n_particle,
                                         compute_score=False)
                seq = model.fit(T, obs_mask=maskT.copy(), max_iter=max_iter)
                with open(file_name, 'wb') as f:
                    pickle.dump([model, seq], f)

            file_name = os.path.join(dest,
                                     'sRESCAL_p_%.2f_dim_%d_par_%d_test_%d_convar_%r.pkl' % (
                                         p, n_dim, n_particle, nt, True))
        elif s_model == 'csRESCAL':
            if not os.path.exists(file_name):
                model = PFBayesianRescal(n_dim, controlled_var=True, obs_var=obs_var, unobs_var=unobs_var,
                                         n_particles=n_particle, compute_score=False)
            seq = model.fit(T, obs_mask=maskT.copy(), max_iter=max_iter)
            with open(file_name, 'wb') as f:
                pickle.dump([model, seq], f)

        elif s_model == 'RESCAL':
            file_name = os.path.join(dest,
                                     'RESCAL_p_%.2f_dim_%d_test_%d.pkl' % (p, n_dim, nt))

            if not os.path.exists(file_name):
                trainT = T.copy()
                trainT[maskT == 0] = 0
                mask = maskT.copy()

                X = list()
                for k in range(n_relation):
                    X.append(csr_matrix(trainT[k]))

                rescal_seq = list()
                for i in range(max_iter):
                    A, R, _, _, _ = rescal.rescal_als(X, n_dim)
                    _X = np.zeros_like(T)
                    for k in range(T.shape[0]):
                        _X[k] = np.dot(np.dot(A, R[k]), A.T)
                    find = False
                    while not find:
                        _X[mask == 1] = -10000
                        next_idx = np.unravel_index(_X.argmax(), T.shape)
                        mask[next_idx] = 1
                        rescal_seq.append(next_idx)
                        if T[next_idx] == 1:
                            X[next_idx[0]][next_idx[1], next_idx[2]] = T[next_idx]
                            find = True
                        if len(rescal_seq) == max_iter:
                            break
                    if len(rescal_seq) == max_iter:
                        break
                with open(file_name, 'wb') as f:
                    pickle.dump([A, R, rescal_seq], f)

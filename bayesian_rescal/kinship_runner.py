import os
import sys
import pickle
import itertools
import numpy as np
from scipy.io.matlab import loadmat
from scipy.sparse import csr_matrix
import rescal
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
    for k in range(n_relation):
        for i, j in itertools.product(range(n_entity), repeat=2):
            if T[k, i, j] and np.random.binomial(1, p):
                idx.add((k, i, j))
    return idx


if __name__ == '__main__':
    if len(sys.argv) != 7:
        print('usage: python kinship_runner.py model_name n_dim n_particle var_x trial_num max_iter')
        raise Exception()
    """
    Test with Kinship dataset
    """
    mat = loadmat('../data/alyawarradata.mat')
    T = np.array(mat['Rs'], np.float32)
    T = np.swapaxes(T, 1, 2)
    T = np.swapaxes(T, 0, 1)  # [relation, entity, entity]

    n_relation, n_entity, _ = T.shape
    s_model = sys.argv[1]
    n_dim = int(sys.argv[2])
    n_particle = int(sys.argv[3])
    var_x = float(sys.argv[4])
    nt = int(sys.argv[5])
    max_iter = int(sys.argv[6])
    obs_var = 0.01
    unobs_var = 0.1

    dest = '../result/kinship/'
    if not os.path.exists(dest):
        os.makedirs(dest)

    # for nt in range(n_test):
    #     train_file = os.path.join(dest, 'train_%.2f_%d' % (p, nt))
    #     if not os.path.exists(train_file):
    #         train_seq = gen_train(T, p)
    #         with open(train_file, 'wb') as f:
    #             pickle.dump(train_seq, f)

    # for nt in range(n_test):
    #     train_file = os.path.join(dest, 'train_%.2f_%d' % (p, nt))
    #     maskT = read_train(train_file, T)

    maskT = np.zeros_like(T)

    if s_model == 'sRESCAL':
        file_name = os.path.join(dest,
                                 'sRESCAL_varx_%.2f_dim_%d_par_%d_test_%d_convar_%r.pkl' % (
                                     var_x, n_dim, n_particle, nt, False))

        if not os.path.exists(file_name):
            log = os.path.splitext(file_name)[0] + ".txt"
            model = PFBayesianRescal(n_dim, controlled_var=False, n_particles=n_particle,
                                     compute_score=False, parallel=False, log=log, dest=file_name)
            seq = model.fit(T, obs_mask=maskT.copy(), max_iter=max_iter)

    elif s_model == 'rbsRESCAL':
        file_name = os.path.join(dest,
                                 'rbsRESCAL_varx_%.2f_dim_%d_par_%d_test_%d_convar_%r.pkl' % (
                                     var_x, n_dim, n_particle, nt, False))

        if not os.path.exists(file_name):
            log = os.path.splitext(file_name)[0] + ".txt"
            model = PFBayesianRescal(n_dim, controlled_var=False, n_particles=n_particle,
                                     compute_score=False, parallel=False, log=log, rbp=True)
            seq = model.fit(T, obs_mask=maskT.copy(), max_iter=max_iter)
            with open(file_name, 'wb') as f:
                pickle.dump([model, seq], f)

    elif s_model == 'csRESCAL':
        file_name = os.path.join(dest,
                                 'csRESCAL_varx_%.2f_dim_%d_par_%d_test_%d_convar_%r.pkl' % (
                                     var_x, n_dim, n_particle, nt, True))

        if not os.path.exists(file_name):
            log = os.path.splitext(file_name)[0] + ".txt"
            model = PFBayesianRescal(n_dim, controlled_var=True, obs_var=obs_var, unobs_var=unobs_var,
                                     n_particles=n_particle, compute_score=True, log=log)
        seq = model.fit(T, obs_mask=maskT.copy(), max_iter=max_iter)
        with open(file_name, 'wb') as f:
            pickle.dump([model, seq], f)

    elif s_model == 'RESCAL':
        file_name = os.path.join(dest,
                                 'RESCAL_dim_%d_test_%d.pkl' % (n_dim, nt))

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

import os
import sys
import pickle
import itertools
import numpy as np
from scipy.io.matlab import loadmat
from scipy.sparse import csr_matrix
import rescal
from seq_brescal import PFBayesianRescal
from seq_sparse_brescal import PFSparseBayesianRescal

if __name__ == '__main__':
    if len(sys.argv) != 7:
        print('usage: python nation_runner.py model_name n_dim n_particle var_x trial_num max_iter')
        raise Exception()
    """
    Test with Nation dataset
    """
    mat = loadmat('../data/nation/dnations.mat')
    T = np.array(mat['R'], np.float32)
    T = np.swapaxes(T, 1, 2)
    T = np.swapaxes(T, 0, 1)  # [relation, entity, entity]
    T[np.isnan(T)] = 0

    n_relation, n_entity, _ = T.shape
    s_model = sys.argv[1]
    n_dim = int(sys.argv[2])
    n_particle = int(sys.argv[3])
    var_x = float(sys.argv[4])
    nt = int(sys.argv[5])
    max_iter = int(sys.argv[6])
    if max_iter == 0:
        max_iter = np.prod(T.shape)

    compositional = False

    if compositional:
        dest = '../result/nation/compositional/'
    else:
        dest = '../result/nation/normal/'

    if not os.path.exists(dest):
        os.makedirs(dest)

    if s_model == 'sRESCAL':
        file_name = os.path.join(dest,
                                 'sRESCAL_varx_%.2f_dim_%d_par_%d_test_%d_convar_%r.pkl' % (
                                     var_x, n_dim, n_particle, nt, False))

        if not os.path.exists(file_name):
            log = os.path.splitext(file_name)[0] + ".txt"
            if os.path.exists(log):
                os.remove(log)
            T = [csr_matrix(T[k]) for k in range(n_relation)]
            model = PFSparseBayesianRescal(n_dim, n_particles=n_particle, compute_score=False, parallel=False,
                                               log=log, dest=file_name, compositional=compositional)
            seq = model.fit(T, max_iter=max_iter)

import os
import sys
import pickle
import itertools
import numpy as np
from scipy.io.matlab import loadmat
from scipy.sparse import csr_matrix
import rescal
from seq_brescal import PFBayesianRescal, load_and_run

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

    model_file = '../result/kinship/sRESCAL_varx_%.2f_dim_%d_par_%d_test_%d_convar_%r.pkl' % (
        var_x, n_dim, n_particle, nt, False)

    if os.path.exists(model_file):
        seq = load_and_run(model_file, T, max_iter)

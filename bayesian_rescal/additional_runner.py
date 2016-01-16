import os
import sys
import numpy as np
from scipy.io.matlab import loadmat
from seq_brescal import PFBayesianRescal, load_and_run
from seq_sparse_brescal import PFSparseBayesianRescal

if __name__ == '__main__':
    if len(sys.argv) != 7:
        print('usage: python additional_runner.py dataset n_dim n_particle var_x trial_num max_iter')
        raise Exception()

    dataset = sys.argv[1]
    n_dim = int(sys.argv[2])
    n_particle = int(sys.argv[3])
    var_x = float(sys.argv[4])
    nt = int(sys.argv[5])
    max_iter = int(sys.argv[6])
    compositional = bool(sys.argv[7])

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

    n_relation, n_entity, _ = T.shape

    if max_iter == 0:
        max_iter = np.prod(T.shape)

    if compositional:
        dest = '../result/%s/compositional/' % (dataset)
    else:
        dest = '../result/%s/normal/' % (dataset)

    model_file = os.path.join(dest, 'sRESCAL_varx_%.2f_dim_%d_par_%d_test_%d_convar_%r.pkl' % (
        var_x, n_dim, n_particle, nt, False))

    if os.path.exists(model_file):
        seq = load_and_run(model_file, T, max_iter)

import os
import sys
import numpy as np
import pickle
from scipy.io.matlab import loadmat
from scipy.sparse import csr_matrix
from seq_brescal import PFBayesianRescal
from seq_sparse_brescal import PFSparseBayesianRescal

if __name__ == '__main__':
    if len(sys.argv) != 8:
        print('usage: python runner.py dataset n_dim n_particle var_x trial_num max_iter')
        raise Exception()

    dataset = sys.argv[1]
    n_dim = int(sys.argv[2])
    n_particle = int(sys.argv[3])
    var_x = float(sys.argv[4])
    nt = int(sys.argv[5])
    max_iter = int(sys.argv[6])
    compositional = False
    if sys.argv[7] == 'True':
        compositional = True

    p_obs = 0.05

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

    with open('../data/%s/train_%.3f.pkl' % (dataset, p_obs), 'rb') as f:
        mask = pickle.load(f)

    if max_iter == 0:
        max_iter = np.prod(T.shape)

    if compositional:
        dest = '../result/%s/compositional_%.3f/' % (dataset, p_obs)
    else:
        dest = '../result/%s/normal_%.3f/' % (dataset, p_obs)

    if not os.path.exists(dest):
        os.makedirs(dest)

    file_name = os.path.join(dest,
                             'sRESCAL_varx_%.2f_dim_%d_par_%d_test_%d_convar_%r.pkl' % (
                                 var_x, n_dim, n_particle, nt, False))

    if not os.path.exists(file_name):
        log = os.path.splitext(file_name)[0] + ".txt"
        if os.path.exists(log):
            os.remove(log)
        model = PFBayesianRescal(n_dim, n_particles=n_particle, compute_score=False, parallel=False,
                                 log=log, dest=file_name, compositional=compositional, gibbs_init=True)

        seq = model.fit(T, obs_mask=mask, max_iter=max_iter)

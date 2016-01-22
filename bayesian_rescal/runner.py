import os
import sys
import numpy as np
import pickle
from scipy.io.matlab import loadmat
from seq_brescal import PFBayesianRescal
from seq_bcomp_rescal import PFBayesianAddCompRescal

if __name__ == '__main__':
    if len(sys.argv) != 8:
        print('usage: python runner.py model dataset n_dim n_particle var_x trial_num max_iter')
        raise Exception()

    model = sys.argv[1]
    dataset = sys.argv[2]
    n_dim = int(sys.argv[3])
    n_particle = int(sys.argv[4])
    var_x = float(sys.argv[5])
    nt = int(sys.argv[6])
    max_iter = int(sys.argv[7])

    if dataset == 'umls':
        mat = loadmat('../data/%s/uml.mat' % (dataset))
        T = np.array(mat['Rs'], np.float32)
    elif dataset == 'nation':
        mat = loadmat('../data/%s/dnations.mat' % (dataset))
        T = np.array(mat['R'], np.float32)
    elif dataset == 'kinship':
        mat = loadmat('../data/%s/alyawarradata.mat' % (dataset))
        T = np.array(mat['Rs'], np.float32)
    elif dataset == 'wordnet':
        T = pickle.load(open('../data/%s/reduced_wordnet.pkl' % (dataset), 'rb'))

    if dataset != 'wordnet':
        T = np.swapaxes(T, 1, 2)
        T = np.swapaxes(T, 0, 1)  # [relation, entity, entity]
        T[np.isnan(T)] = 0

    n_relation, n_entity, _ = T.shape

    if max_iter == 0:
        # set number of iteration to the size of tensor in case where max_iter = 0
        max_iter = np.prod(T.shape)

    # destination folder where model and log files are saved
    dest = '../result/%s/%s/' % (dataset, model)

    if not os.path.exists(dest):
        os.makedirs(dest, exist_ok=True)

    model_file = os.path.join(dest, 'sRESCAL_varx_%.2f_dim_%d_par_%d_test_%d_convar_%r.pkl' % (
        var_x, n_dim, n_particle, nt, False))

    if not os.path.exists(model_file):
        # change file extension from pkl to txt for writing log
        log = os.path.splitext(model_file)[0] + ".txt"
        if os.path.exists(log):
            os.remove(log)

        if model == 'brescal':
            _model = PFBayesianRescal(n_dim, n_particles=n_particle, compute_score=False, parallel=False,
                                      log=log, dest=model_file, sample_all=True)
        elif model == 'bcomp_rescal':
            _model = PFBayesianAddCompRescal(n_dim, n_particles=n_particle, compute_score=False, log=log,
                                                           dest=model_file)

        seq = _model.fit(T, max_iter=max_iter)

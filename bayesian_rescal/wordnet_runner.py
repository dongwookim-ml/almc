import os
import sys
import pickle
import itertools
import numpy as np
import rescal
from seq_sparse_brescal import PFSparseBayesianRescal

if __name__ == '__main__':
    if len(sys.argv) != 7:
        print('usage: python kinship_runner.py model_name n_dim n_particle var_x trial_num max_iter')
        raise Exception()
    """
    Test with Kinship dataset
    """
    T = pickle.load(open('../data/wordnet/wordnet_csr.pkl', 'rb'))

    n_relation = len(T)
    n_entity = T[0].shape[0]
    s_model = sys.argv[1]
    n_dim = int(sys.argv[2])
    n_particle = int(sys.argv[3])
    var_x = float(sys.argv[4])
    nt = int(sys.argv[5])
    max_iter = int(sys.argv[6])

    dest = '../result/wordnet_compositional/'
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
            model = PFSparseBayesianRescal(n_dim, n_particles=n_particle, compute_score=False, parallel=False, log=log,
                                           dest=file_name, compositional=False)
            seq = model.fit(T, max_iter=max_iter)

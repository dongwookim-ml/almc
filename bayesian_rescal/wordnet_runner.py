import os
import sys
import pickle
import itertools
import numpy as np
import rescal
from seq_sparse_brescal import PFSparseBayesianRescal
from scipy.sparse import csr_matrix as sparse_matrix
from sklearn.metrics import mean_squared_error

if __name__ == '__main__':
    if len(sys.argv) != 7:
        print('usage: python wordnet_runner.py model_name n_dim n_particle var_x trial_num max_iter')
        raise Exception()
    T = pickle.load(open('../data/wordnet/wordnet_csr.pkl', 'rb'))

    n_relation = len(T)
    n_entity = T[0].shape[0]
    s_model = sys.argv[1]
    n_dim = int(sys.argv[2])
    n_particle = int(sys.argv[3])
    var_x = float(sys.argv[4])
    nt = int(sys.argv[5])
    max_iter = int(sys.argv[6])

    p = 0.1
    compositional = True

    if compositional:
        dest = '../result/wordnet/compositional/'
    else:
        dest = '../result/wordnet/normal/'
    if not os.path.exists(dest):
        os.makedirs(dest)

    valid = np.sum([T[k].sum() for k in range(n_relation)])
    total = n_relation * n_entity ** 2
    print(valid, total)
    print((valid / total))

    train_file = '../result/wordnet/train_%.3f.pkl' % p
    if os.path.exists(train_file):
        maskT = pickle.load(open(train_file, 'rb'))
    else:
        maskT = [sparse_matrix((n_entity, n_entity)) for k in range(n_relation)]
        for k in range(n_relation):
            nz = T[k].nonzero()
            for i in range(T[k].nnz):
                if np.random.binomial(1, p):
                    maskT[k][nz[0][i], nz[1][i]] = 1
        pickle.dump(maskT, open(train_file, 'wb'))

    print('Total :', np.sum([maskT[k].sum() for k in range(n_relation)]))

    if s_model == 'sRESCAL':
        file_name = os.path.join(dest,
                                 'sRESCAL_varx_%.2f_dim_%d_par_%d_test_%d_convar_%r.pkl' % (
                                     var_x, n_dim, n_particle, nt, False))

        #A, R, _, _, _ = rescal.rescal_als(T, n_dim)

        if not os.path.exists(file_name):
            log = os.path.splitext(file_name)[0] + ".txt"
            if os.path.exists(log):
                os.remove(log)
            model = PFSparseBayesianRescal(n_dim, n_particles=n_particle, compute_score=False, parallel=False, log=log,
                                           dest=file_name, compositional=compositional, gibbs_init=True)
            seq = model.fit(T, obs_mask_csr=maskT, max_iter=max_iter)

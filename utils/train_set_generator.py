import os
import sys
import pickle
import itertools
import numpy as np
from scipy.io.matlab import loadmat
from scipy.sparse import csr_matrix

if __name__ == '__main__':
    dataset = 'umls'
    p_obs = 0.05
    p_test = 0.3
    dest = '../data/%s/' % (dataset)
    include_negative = True
    n_test = 10

    for nt in range(n_test):
        file_name = 'train_test_%.3f_%.3f_%d.pkl' % (p_obs, p_test, nt)
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

        mask = np.random.binomial(1, p_obs, T.shape)
        test_mask = np.random.binomial(1, p_test, T.shape)
        test_mask[mask==1] = 0

        with open(os.path.join(dest, file_name), 'wb') as f:
            pickle.dump([mask, test_mask], f)

        # n_relation, n_entity, _ = T.shape
        #
        # mask = np.zeros_like(T)
        # for r_k, e_i, e_j in itertools.product(range(n_relation), range(n_entity), range(n_entity)):
        #     if T[r_k, e_i, e_j] and np.random.binomial(1, p_obs):
        #         mask[r_k, e_i, e_j] = 1
        #
        # for i in range(int(np.sum(mask))):
        #     r_k = np.random.randint(n_relation)
        #     e_i = np.random.randint(n_entity)
        #     e_j = np.random.randint(n_entity)
        #     mask[r_k, e_i, e_j] = 1
        #
        # with open(os.path.join(dest, file_name), 'wb') as f:
        #     pickle.dump(mask, f)
        #
        # print('Total # of observation', np.sum(mask))
        # print('Total # of valid triples', np.sum(T))

import os
import sys
import numpy as np
import pickle
from scipy.io.matlab import loadmat
from model import AMDC


def load_dataset(dataset):
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

    T[np.isnan(T)] = 0
    return T


if __name__ == '__main__':
    if len(sys.argv) != 8:
        print('usage: python runner.py model dataset n_dim p_obs p_test max_iter')
        raise Exception()

    model = sys.argv[1]
    dataset = sys.argv[2]
    n_dim = int(sys.argv[3])
    p_obs = float(sys.argv[4])
    p_test = float(sys.argv[5])
    max_iter = int(sys.argv[7])
    n_test = 10

    for nt in range(n_test):
        T = load_dataset(dataset)
        n_relation, n_entity, _ = T.shape

        train_file_path = '../data/%s/train_test_%.3f_%.3f_%d.pkl' % (dataset, p_obs, p_test, nt)
        train_mask, test_mask = pickle.load(open(train_file_path, 'rb'))

        train_mask = np.swapaxes(train_mask, 0, 1)
        train_mask = np.swapaxes(train_mask, 1, 2)

        test_mask = np.swapaxes(test_mask, 0, 1)
        test_mask = np.swapaxes(test_mask, 1, 2)

        dest = '../result/%s/amdc/' % (dataset)
        if not os.path.exists(dest):
            os.makedirs(dest, exist_ok=True)

        log = 'query_%s_train_%.3f_test_%.3f_%d_%d.txt' % (model, p_obs, p_test, n_dim, nt)
        log2 = 'auc_%s_train_%.3f_test_%.3f_%d_%d.txt' % (model, p_obs, p_test, n_dim, nt)

        amdc = AMDC(n_dim)

        if model == 'population':
            amdc.population = True
        elif model == 'predictive':
            amdc.population = False

        query_log = os.path.join(dest, log)
        auc_log = os.path.join(dest, log2)
        amdc.do_active_learning(T, train_mask, max_iter, test_mask, query_log, auc_log)

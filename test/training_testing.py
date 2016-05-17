import os
import itertools
import numpy as np
import pickle
from scipy.io.matlab import loadmat
from almc.bayesian_rescal import PFBayesianRescal
from almc.bayesian_rescal import PFBayesianCompRescal
from almc.bayesian_rescal import PFBayesianLogitRescal
from sklearn.metrics import roc_auc_score


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
    elif dataset == 'freebase':
        T, _, _ = pickle.load(open('../data/freebase/subset_3000.pkl', 'rb'))

    if dataset == 'umls' or dataset == 'nation' or dataset == 'kinship':
        T = np.swapaxes(T, 1, 2)
        T = np.swapaxes(T, 0, 1)  # [relation, entity, entity]
        T[np.isnan(T)] = 0
    return T


dataset = 'umls'
models = ['brescal', 'bcomp_mul', 'bcomp_add', 'logit']
# models = ['bcomp_mul', 'bcomp_add', 'logit']
ps = np.linspace(0.01, 0.3, 10)
ps = [0.1]
n_dim = 10
n_particle = 1
var_x = 0.1
var_comp = 1.
n_test = 10
n_test = 1
max_iter = 0

T = load_dataset(dataset)
n_relation, n_entity, _ = T.shape

result = dict()

# keep the same index sequence for any experiments
rnd = np.random.RandomState(seed=45342412)
dest = '../result_tt/%s/' % (dataset)

if not os.path.exists(dest):
    os.makedirs(dest, exist_ok=True)

for nt in range(n_test):
    indexes = [(k, i, j) for k, i, j in itertools.product(range(n_relation), range(n_entity), range(n_entity))]
    total = n_relation * n_entity * n_entity
    rnd.shuffle(indexes)

    validT = np.zeros_like(T)
    testT = np.zeros_like(T)
    for (k, i, j) in indexes[int(total * 0.5):int(total * 0.7)]:
        validT[k, k, j] = 1
    for (k, i, j) in indexes[int(total * 0.7):]:
        testT[k, i, j] = 1

    test_file = os.path.join(dest, 'test.pkl')
    pickle.dump(testT, open(test_file, 'wb'))

    for p in ps:
        train = np.zeros_like(T)
        for (k, i, j) in indexes[:int(p * len(indexes))]:
            train[k, i, j] = 1

        train_file = os.path.join(dest, '%.2f_train.pkl' % p)
        pickle.dump(train, open(train_file, 'wb'))

        print(nt, 'Train Sum', np.sum(T[train == 1]), np.sum(train))
        for model in models:
            # destination folder where model and log files are saved
            if model == 'bcomp_add' or model == 'bcomp_mul':
                output_file = os.path.join(dest, '%s_%.2f_%d_%.2f_%.2f_training_error.pkl' % (
                    model, p, n_dim, var_x, var_comp))
            else:
                output_file = os.path.join(dest, '%s_%.2f_%d_training_error.pkl' % (model, p, n_dim))

            if not os.path.exists(output_file):
                if model == 'brescal':
                    model_file = os.path.join(dest, 'sRESCAL_varx_%.2f_dim_%d_par_%d_test_%d.pkl' % (
                        var_x, n_dim, n_particle, nt))
                elif model == 'logit':
                    model_file = os.path.join(dest, 'sRESCAL_dim_%d_par_%d_test_%d.pkl' % (
                        n_dim, n_particle, nt))
                else:
                    model_file = os.path.join(dest, 'sRESCAL_var_%.2f_%.2f_dim_%d_par_%d_test_%d.pkl' % (
                        var_x, var_comp, n_dim, n_particle, nt))

                if model == 'brescal':
                    _model = PFBayesianRescal(n_dim, var_x=var_x, n_particles=n_particle, compute_score=False,
                                              parallel=False, sample_all=True)
                elif model == 'bcomp_add':
                    _model = PFBayesianCompRescal(n_dim, compositionality='additive', var_x=var_x, var_comp=var_comp,
                                                  n_particles=n_particle, compute_score=False)
                elif model == 'bcomp_mul':
                    _model = PFBayesianCompRescal(n_dim, compositionality='multiplicative', var_x=var_x,
                                                  var_comp=var_comp,
                                                  n_particles=n_particle, compute_score=False)
                elif model == 'logit':
                    _model = PFBayesianLogitRescal(n_dim, compute_score=False, n_particles=n_particle)
                elif model == 'logit_mul':
                    _model = PFBayesianCompRescal(n_dim, compositionality='logit_mul',
                                                  n_particles=n_particle, compute_score=False)
                else:
                    raise Exception('No such model exists %s' % model)

                seq = _model.fit(T, obs_mask=train.copy(), max_iter=max_iter)
                particle = _model.p_weights.argmax()
                _X = _model._reconstruct(_model.E[particle], _model.R[particle])
                val_auc = roc_auc_score(T[validT == 1].flatten(), _X[validT == 1].flatten())
                auc = roc_auc_score(T[testT == 1].flatten(), _X[testT == 1].flatten())

                if (model, p) not in result:
                    result[(model, p)] = list()
                result[(model, p)].append([val_auc, auc])
                print(model, p, val_auc, auc)
                pickle.dump(result, open(output_file, 'wb'))
                pickle.dump(_model, open(os.path.join(dest, '%s_%.2f.pkl' % (model, p)), 'wb'))
                pickle.dump(_model.E[particle], open(os.path.join(dest, 'E_%s_%.2f.pkl' % (model, p)), 'wb'))
                pickle.dump(_model.R[particle], open(os.path.join(dest, 'R_%s_%.2f.pkl' % (model, p)), 'wb'))

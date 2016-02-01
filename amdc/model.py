__author__ = 'Dongwoo Kim'

import numpy as np
import logging
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

heaviside = lambda x: 1 if x >= 0 else 0
MIN_VAL = np.iinfo(np.int32).min


class AMDC:
    """
    Implementation of Active Multi-relational Data Construction (AMDC) method.

    Reference:
    Kajino, H., Kishimoto, A., Botea, A., Daly, E., & Kotoulas, S. (2015). Active Learning for Multi-relational Data
    Construction. WWW 2015
    """

    def __init__(self, n_dim, alpha_0=0.1, gamma=0.3, gamma_p=0.9, c_e=5., c_n=1., population=False):
        """

        @param n_dim: latent dimension of entity
        @param alpha_0: initial learning rate
        @param gamma: hyperparameter
        @param gamma_p: hyperparameter
        @param c_e: hyperparameter, impose score of positive triple to be greater than 0,
                    and negative triple to be less than 0
        @param c_n: hyperparameter, importance of negative samples
        @return:
        """
        self.n_dim = n_dim
        self.alpha_0 = alpha_0
        self.gamma = gamma
        self.gamma_p = gamma_p
        self.c_e = c_e
        self.c_n = c_n
        self.population = population

    def fit(self, T, p_idx, n_idx, max_iter=100, e_gap=100, A=None, R=None, obs_only=False):
        """
        Stochastic gradient descent optimization for AMDC

        @param T: [n_entity x n_entity x n_relation] multi-dimensional array,
                tensor representation of knowledge graph
                n_entity = number of entities
                n_relation = number of relationships
        @param p_idx: index of observed positive triples, all indices are raveled by np.ravel_multi_index
        @param n_idx: index of observed negative triples
        @param max_iter: maximum number of iterations
        @param e_gap: evaluation gap
        @param obs_only: When this parameter is True, the stochastic gradient step uses the
                observed positive and negative triples only.
        @return: A, R, r_error
                A: [n_entity x n_dim] latent feature vector of entities
                R: [n_dim x n_dim x n_relation] rotation matrix for each entity
                n_dim = size of latent dimension
                r_error: list of reconstruction errors at each evaluation point
        """
        n_entity, n_relation = T.shape[0], T.shape[2]

        np_idx = np.setdiff1d(range(np.prod(T.shape)), p_idx)  # not positive index
        nn_idx = np.setdiff1d(range(np.prod(T.shape)), n_idx)  # not negative index

        if isinstance(A, type(None)):
            A = np.random.random([n_entity, self.n_dim]) - 0.5
            A /= np.linalg.norm(A, ord=2, axis=1)[:, np.newaxis]
            R = np.zeros([self.n_dim, self.n_dim, n_relation])  # rotation matrices
            for k in range(n_relation):
                R[:, :, k] = np.identity(self.n_dim)

        r_error = list()

        if len(p_idx) == 0 and len(n_idx) == 0:
            return A, R, r_error

        if obs_only and (len(p_idx) == 0 or len(n_idx) == 0):
            return A, R, r_error

        it = 0
        converged = False
        learning_rate = self.alpha_0
        while not converged:
            if not obs_only:
                selector = np.random.randint(100) % 2
                if len(p_idx) == 0:
                    selector = 0
                elif len(n_idx) == 0:
                    selector = 1

                if selector:
                    next_idx = np.random.randint(len(p_idx))
                    next_np_idx = np.random.randint(len(np_idx))

                    i, j, k = np.unravel_index(p_idx[next_idx], T.shape)
                    i_bar, j_bar, k_bar = np.unravel_index(np_idx[next_np_idx], T.shape)

                    I_1 = heaviside(self.gamma - np.dot(np.dot(A[i], R[:, :, k]), A[j])
                                    + np.dot(np.dot(A[i_bar], R[:, :, k_bar]), A[j_bar]))
                    I_2 = heaviside(self.gamma_p - np.dot(np.dot(A[i], R[:, :, k]), A[j]))

                    # updating parameters
                    if I_1 != 0 or I_2 != 0:
                        a_i, a_j, r_k = A[i].copy(), A[j].copy(), R[:, :, k].copy()
                        A[i] -= learning_rate * (-(I_1 + I_2 * self.c_e) * np.dot(r_k, a_j))
                        A[j] -= learning_rate * (-(I_1 + I_2 * self.c_e) * np.dot(r_k.T, a_i))
                        R[:, :, k] -= learning_rate * (-(I_1 + I_2 * self.c_e) * np.outer(a_i, a_j))

                    if I_1 != 0:
                        a_i_bar, a_j_bar, r_k_bar = A[i_bar].copy(), A[j_bar].copy(), R[:, :, k_bar].copy()
                        A[i_bar] -= learning_rate * (I_1 * np.dot(r_k_bar, a_j_bar))
                        A[j_bar] -= learning_rate * (I_1 * np.dot(r_k_bar.T, a_i_bar))
                        R[:, :, k_bar] -= learning_rate * (I_1 * np.outer(a_i_bar, a_j_bar))

                else:
                    next_idx = np.random.randint(len(n_idx))
                    next_nn_idx = np.random.randint(len(nn_idx))

                    i, j, k = np.unravel_index(n_idx[next_idx], T.shape)
                    i_bar, j_bar, k_bar = np.unravel_index(nn_idx[next_nn_idx], T.shape)

                    I_3 = heaviside(self.gamma + np.dot(np.dot(A[i], R[:, :, k]), A[j])
                                    - np.dot(np.dot(A[i_bar], R[:, :, k_bar]), A[j_bar]))
                    I_4 = heaviside(self.gamma_p + np.dot(np.dot(A[i], R[:, :, k]), A[j]))

                    if I_3 != 0 or I_4 != 0:
                        a_i, a_j, r_k = A[i].copy(), A[j].copy(), R[:, :, k].copy()
                        A[i] -= learning_rate * ((I_3 * self.c_n + I_4 * self.c_e) * np.dot(r_k, a_j))
                        A[j] -= learning_rate * ((I_3 * self.c_n + I_4 * self.c_e) * np.dot(r_k.T, a_i))
                        R[:, :, k] -= learning_rate * ((I_3 * self.c_n + I_4 * self.c_e) * np.outer(a_i, a_j))

                    if I_3 != 0:
                        a_i_bar, a_j_bar, r_k_bar = A[i_bar].copy(), A[j_bar].copy(), R[:, :, k_bar].copy()
                        A[i_bar] -= learning_rate * (I_3 * self.c_n * np.dot(r_k_bar, a_j_bar))
                        A[j_bar] -= learning_rate * (I_3 * self.c_n * np.dot(r_k_bar.T, a_i_bar))
                        R[:, :, k_bar] -= learning_rate * (I_3 * self.c_n * np.outer(a_i_bar, a_j_bar))

            elif obs_only:
                next_p_idx = np.random.randint(len(p_idx))
                next_n_idx = np.random.randint(len(n_idx))

                i, j, k = np.unravel_index(p_idx[next_p_idx], T.shape)
                i_bar, j_bar, k_bar = np.unravel_index(np_idx[next_n_idx], T.shape)

                I_1 = heaviside(self.gamma - np.dot(np.dot(A[i], R[:, :, k]), A[j])
                                + np.dot(np.dot(A[i_bar], R[:, :, k_bar]), A[j_bar]))
                I_2 = heaviside(self.gamma_p - np.dot(np.dot(A[i], R[:, :, k]), A[j]))

                I_4 = heaviside(self.gamma_p + np.dot(np.dot(A[i_bar], R[:, :, k_bar]), A[j_bar]))

                if I_1 != 0 or I_2 != 0:
                    a_i, a_j, r_k = A[i].copy(), A[j].copy(), R[:, :, k].copy()
                    A[i] -= learning_rate * (-(I_1 + I_2 * self.c_e) * np.dot(r_k, a_j))
                    A[j] -= learning_rate * (-(I_1 + I_2 * self.c_e) * np.dot(r_k.T, a_i))
                    R[:, :, k] -= learning_rate * (-(I_1 + I_2 * self.c_e) * np.outer(a_i, a_j))

                if I_1 != 0 or I_4 != 0:
                    a_i, a_j, r_k = A[i_bar].copy(), A[j_bar].copy(), R[:, :, k_bar].copy()
                    A[i] -= learning_rate * ((I_1 + I_4 * self.c_e) * np.dot(r_k, a_j))
                    A[j] -= learning_rate * ((I_1 + I_4 * self.c_e) * np.dot(r_k.T, a_i))
                    R[:, :, k] -= learning_rate * ((I_1 + I_4 * self.c_e) * np.outer(a_i, a_j))

            # unit vector projection (this could be improved by using l1 projection alg.)
            # converting learned matrix to rotational matrix
            A[i] /= np.linalg.norm(A[i], ord=2)
            A[j] /= np.linalg.norm(A[j], ord=2)
            U, sigma, V = np.linalg.svd(R[:, :, k])
            R[:, :, k] = np.dot(U, V)

            A[i_bar] /= np.linalg.norm(A[i_bar], ord=2)
            A[j_bar] /= np.linalg.norm(A[j_bar], ord=2)
            U, sigma, V = np.linalg.svd(R[:, :, k_bar])
            R[:, :, k_bar] = np.dot(U, V)

            if it >= max_iter:
                converged = True

            if it % e_gap == 0 and it != 0:
                T_bar = self.reconstruct(A, R)
                _T = T.copy()
                _T[_T == -1] = 0
                T_bar = (T_bar + 1.) / 2.

                err = 0.
                for k in range(n_relation):
                    err += roc_auc_score(_T[:, :, k].flatten(), T_bar[:, :, k].flatten())
                err /= float(n_relation)

                obj = self.evaluate_objfn(A, R, p_idx, n_idx)

                r_error.append((obj, err))

                log.debug('Iter %d, ObjectiveFn: %.5f, ROC-AUC: %.5f' % (it, obj, err))
            else:
                log.debug('Iter %d' % (it))

            it += 1
            learning_rate = self.alpha_0 / np.sqrt(it)

        return A, R, r_error

    def reconstruct(self, A, R):
        """
        Reconstruct knowledge graph from latent representations of entities and rotation matrices

        @param A: [E x D] multi-dimensional array, latent representation of entity
        @param R: [D x D x K] multi-dimensional array, rotation matrix for each relation
        @return: [E x E x K] reconstructed knowledge graph
        """
        T = np.zeros((A.shape[0], A.shape[0], R.shape[2]))

        for i in range(R.shape[2]):
            T[:, :, i] = np.dot(np.dot(A, R[:, :, i]), A.T)

        return T

    def get_next_sample(self, A, R, mask):
        T = self.reconstruct(A, R)
        T[mask == 1] = MIN_VAL
        if self.population:
            idx = T.argmax()
            return np.unravel_index(idx, T.shape), idx
        else:
            idx = np.abs(T).argmin()
            return np.unravel_index(idx, T.shape), idx

    def evaluate_objfn(self, A, R, p_idx, n_idx):
        """
        compute objective function of AMDC model

        @param A: [E x D] multi-dimensional array, latent representation of entity
        @param R: [D x D x K] multi-dimensional array, rotation matrix for each relation
        @param p_idx: index of observed positive triples, all indices are raveled by np.ravel_multi_index
        @param n_idx: index of observed negative triples
        @return: objective function of AMDC model
                Equation (4) in the original paper
        """
        obj = 0
        total = A.shape[0] * A.shape[0] * R.shape[2]
        np_idx = np.setdiff1d(range(total), p_idx)  # not positive index
        nn_idx = np.setdiff1d(range(total), n_idx)  # not negative index

        scores = self.reconstruct(A, R)
        scores = scores.flatten()

        # this approach requires too much memory
        # first = self.gamma - scores[p_idx] + scores[np_idx][:,np.newaxis]
        # first = np.sum(first[first>0])

        # alternative (takes too much time...)
        first, third = 0, 0
        for i in p_idx:
            tmp = self.gamma - scores[i] + scores[np_idx]
            first += np.sum(tmp[tmp > 0])

        second = self.c_e * (self.gamma_p - scores[p_idx])
        second = np.sum(second[second > 0])

        # third = self.c_n * (self.gamma - scores[nn_idx] + scores[n_idx][:,np.newaxis])
        # third = np.sum(third[third>0])
        for i in nn_idx:
            tmp = self.c_n * (self.gamma - scores[i] + scores[n_idx])
            third += np.sum(tmp[tmp > 0])

        fourth = self.c_e * (self.gamma_p + scores[n_idx])
        fourth = np.sum(fourth[fourth > 0])

        obj += (first + second) / float(len(p_idx) * len(np_idx))
        obj += (third + fourth) / float(len(n_idx) * len(nn_idx))

        return obj

    def do_active_learning(self, T, mask, max_iter, test_t, query_log='', eval_log='', obs_only=False):
        T[T == 0] = -1
        cur_obs = np.zeros_like(T)
        cur_obs[mask == 1] = T[mask == 1]

        p_idx = np.ravel_multi_index((cur_obs == 1).nonzero(), T.shape)  # raveled positive index
        n_idx = np.ravel_multi_index((cur_obs == -1).nonzero(), T.shape)  # raveled negative index

        pop = 0
        pull_size = 1

        E, K = T.shape[0], T.shape[2]
        A = np.random.random([E, self.n_dim]) - 0.5
        A /= np.linalg.norm(A, ord=2, axis=1)[:, np.newaxis]
        R = np.zeros([self.n_dim, self.n_dim, K])  # rotation matrices
        for k in range(K):
            R[:, :, k] = np.identity(self.n_dim)

        seq = list()
        auc_scores = list()

        for iter in range(max_iter):
            A, R, _ = self.fit(T, p_idx, n_idx, max_iter=1000, e_gap=1001, A=A, R=R, obs_only=obs_only)
            _T = self.reconstruct(A, R)
            _T[mask == 1] = MIN_VAL
            _T[test_t == 1] = MIN_VAL
            for pull_no in range(pull_size):
                if self.population:
                    idx = _T.argmax()
                    next_idx = np.unravel_index(idx, T.shape)
                else:
                    idx = np.abs(_T).argmin()
                    next_idx = np.unravel_index(idx, T.shape)

                if len(query_log) > 0:
                    with open(query_log, 'a') as f:
                        f.write('%d,%d,%d\n' % (next_idx[0], next_idx[1], next_idx[2]))

                seq.append(next_idx)

                _T[next_idx] = MIN_VAL
                mask[next_idx] = 1
                cur_obs[next_idx] = T[next_idx]
                if cur_obs[next_idx] == 1:
                    pop += 1

                if T[next_idx] == 1:
                    p_idx = np.concatenate((p_idx, (idx,)))
                else:
                    n_idx = np.concatenate((n_idx, (idx,)))

                log.debug('[NEXT IDX] %s, %d', next_idx, cur_obs[next_idx])

            _T = self.reconstruct(A, R)
            auc_roc = roc_auc_score(T[test_t == 1], _T[test_t == 1])
            log.info('[ITER %d] %d/%d, %.2f', iter, pop, (iter + 1) * pull_size,
                     auc_roc)

            if len(eval_log) > 0:
                with open(eval_log, 'a') as f:
                    f.write('%f\n' % auc_roc)

            auc_scores.append(auc_roc)

        return seq, auc_scores


def test():
    """
    Test with Kinship dataset
    Use all positive triples and negative triples as a training set
    See how the reconstruction error is reduced during training
    """
    from scipy.io.matlab import loadmat
    mat = loadmat('../data/kinship/alyawarradata.mat')
    T = np.array(mat['Rs'], np.float32)
    T[T == 0] = -1  # set negative value to -1
    E, K = T.shape[0], T.shape[2]
    max_iter = E * E * K * 10

    n_dim = 10

    # p_idx = np.ravel_multi_index((T == 1).nonzero(), T.shape)  # raveled positive index
    # n_idx = np.ravel_multi_index((T == -1).nonzero(), T.shape)  # raveled negative index
    # model.fit(T, p_idx, n_idx, max_iter, e_gap=10000)

    training = np.random.binomial(1., 0.01, T.shape)
    testing = np.random.binomial(1., 0.5, T.shape)
    testing[training == 1] = 0

    model = AMDC(n_dim)
    model.population = True
    model.do_active_learning(T, training, 15000, testing)


if __name__ == '__main__':
    test()

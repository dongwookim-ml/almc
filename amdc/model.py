__author__ = 'Dongwoo Kim'

import numpy as np
from sklearn.metrics import roc_auc_score

from ..utils.formatted_logger import formatted_logger

log = formatted_logger('AMDC', 'info')

heaviside = lambda x: 1 if x >= 0 else 0

class AMDC:
    """
    Implementation of Active Multi-relational Data Construction (AMDC) method.

    Reference:
    Kajino, H., Kishimoto, A., Botea, A., Daly, E., & Kotoulas, S. (2015). Active Learning for Multi-relational Data
    Construction. WWW 2015
    """

    def __init__(self, D, alpha_0=0.1, gamma=0.3, gamma_p=0.9, c_e=5., c_n=1.):
        """

        @param D: latent dimension of entity
        @param alpha_0: initial learning rate
        @param gamma: hyperparameter
        @param gamma_p: hyperparameter
        @param c_e: hyperparameter, impose score of positive triple to be greater than 0,
                    and negative triple to be less than 0
        @param c_n: hyperparameter, importance of negative samples
        @return:
        """
        self.D = D
        self.alpha_0 = alpha_0
        self.gamma = gamma
        self.gamma_p = gamma_p
        self.c_e = c_e
        self.c_n = c_n

    def learn(self, T, p_idx, n_idx, max_iter, e_gap=1):
        """
        Stochastic gradient descent optimization for AMDC

        @param T: [E x E x K] multi-dimensional array,
                tensor representation of knowledge graph
                E = number of entities
                K = number of relationships
        @param p_idx: index of observed positive triples, all indices are raveled by np.ravel_multi_index
        @param n_idx: index of observed negative triples
        @param max_iter: maximum number of iterations
        @param e_gap: evaluation gap
        @return: A, R, r_error
                A: [E x D] latent feature vector of entities
                R: [D x D x K] rotation matrix for each entity
                D = size of latent dimension
                r_error: list of reconstruction errors at each evaluation point
        """
        E, K = T.shape[0], T.shape[2]

        np_idx = np.setdiff1d(range(np.prod(T.shape)), p_idx)  # not positive index
        nn_idx = np.setdiff1d(range(np.prod(T.shape)), n_idx)  # not negative index

        A = np.random.random([E, self.D]) - 0.5
        A /= np.linalg.norm(A, ord=2, axis=1)[:, np.newaxis]
        R = np.zeros([self.D, self.D, K])  # rotation matrices
        for k in range(K):
            R[:, :, k] = np.identity(self.D)

        r_error = list()

        it = 0
        converged = False
        learning_rate = self.alpha_0
        while not converged:

            if np.random.randint(100) % 2 == 0:
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

            if it % e_gap == 0:
                T_bar = self.reconstruct(A, R)
                _T = T.copy()
                _T[_T==-1] = 0
                T_bar = (T_bar + 1.)/2.

                err = 0.
                for k in range(K):
                    err += roc_auc_score(_T[:,:,k].flatten(),T_bar[:,:,k].flatten())
                err /= float(K)

                obj = self.evaluate_objfn(A, R, p_idx, n_idx)

                r_error.append((obj, err))

                log.info('Iter %d, ObjectiveFn: %.5f, ROC-AUC: %.5f' % (it, obj, err))
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


def test():
    """
    Test with Kinship dataset
    Use all positive triples and negative triples as a training set
    See how the reconstruction error is reduced during training
    """
    from scipy.io.matlab import loadmat
    mat = loadmat('../data/alyawarradata.mat')
    T = np.array(mat['Rs'], np.float32)
    T[T == 0] = -1  # set negative value to -1
    E, K = T.shape[0], T.shape[2]
    max_iter = E * E * K * 10

    latent_dimension = 10

    p_idx = np.ravel_multi_index((T == 1).nonzero(), T.shape)  # raveled positive index
    n_idx = np.ravel_multi_index((T == -1).nonzero(), T.shape)  # raveled negative index

    model = AMDC(latent_dimension)
    model.learn(T, p_idx, n_idx, max_iter, e_gap=10000)


if __name__ == '__main__':
    test()

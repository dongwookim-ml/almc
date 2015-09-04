__author__ = 'Dongwoo Kim'

import numpy as np
from scipy.io.matlab import loadmat
from sklearn.metrics import precision_recall_curve, auc

heaviside = lambda x: 1 if x >= 0 else 0


class AMDC:
    """
    Implementation of Active Multi-relational Data Construction (AMDC) method.
    In this implementation, I did not consider the existence of negative examples suggested in the original paper.

    Reference:
    Kajino, H., Kishimoto, A., Botea, A., Daly, E., & Kotoulas, S. (2015). Active Learning for Multi-relational Data
    Construction. In WWW. International World Wide Web Conferences Steering Committee.
    """

    def __init__(self, D, alpha_0=0.1, gamma=0.3, gamma_p=0.9, c_e=5., c_n=1.):
        self.D = D
        self.alpha_0 = alpha_0
        self.gamma = gamma
        self.gamma_p = gamma_p
        self.c_e = c_e
        self.c_n = c_n

    def learn(self, T, p_idx, n_idx, max_iter, e_gap=1000):
        E, K = T.shape[0], T.shape[2]

        np_idx = np.setdiff1d(range(np.prod(T.shape)), p_idx)
        nn_idx = np.setdiff1d(range(np.prod(T.shape)), n_idx)

        A = np.random.random([E, self.D])
        A /= np.linalg.norm(A, axis=1)[:, np.newaxis]
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
                T_bar[T_bar <= 0] = -1
                T_bar[T_bar > 0] = 1
                err = np.sum(np.abs(T - T_bar))
                r_error.append(err)
                print('Iter %d, Reconstruction error: %d' % (it, err))

            it += 1
            learning_rate = self.alpha_0 / np.sqrt(it)

        return A, R, r_error

    def reconstruct(self, A, R):
        T = np.zeros((A.shape[0], A.shape[0], R.shape[2]))

        for i in range(R.shape[2]):
            T[:, :, i] = np.dot(A, np.dot(R[:, :, i], A.T))

        return T


def test():
    mat = loadmat('../data/alyawarradata.mat')
    T = np.array(mat['Rs'], np.float32)
    T[T == 0] = -1  # set negative value to -1
    E, K = T.shape[0], T.shape[2]
    max_iter = E * E * K

    latent_dimension = 10

    p_idx = np.ravel_multi_index((T == 1).nonzero(), T.shape)  # raveled positive index
    n_idx = np.ravel_multi_index((T == -1).nonzero(), T.shape)  #raveled negative index

    model = AMDC(latent_dimension)
    model.learn(T, p_idx, n_idx, max_iter)  #use all positive and negative samples as training data

if __name__ == '__main__':
    test()

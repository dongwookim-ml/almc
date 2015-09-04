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

    def __init__(self, D, alpha_0=0.1, gamma=0.3, gamma_p=0.9, c_e=5):
        self.D = D
        self.alpha_0 = alpha_0
        self.gamma = gamma
        self.gamma_p = gamma_p
        self.c_e = c_e

    def learn(self, T, max_iter):
        E, K = T.shape[0], T.shape[2]

        known_index = np.transpose(np.nonzero(T))  # indicies of nonzero element in T
        unknown_index = np.transpose(np.nonzero(T == 0))  # indicies of zero element in T

        A = np.random.random([E, self.D])
        A /= np.linalg.norm(A, axis=1)[:, np.newaxis]
        assert np.isclose(np.linalg.norm(A[0]), 1)
        R = np.zeros([self.D, self.D, K])  # rotation matrices
        for k in range(K):
            R[:, :, k] = np.identity(self.D)

        it = 0
        converged = False
        learning_rate = self.alpha_0

        while not converged:
            next_idx = np.random.randint(known_index.shape[0])
            next_unknown_idx = np.random.randint(unknown_index.shape[0])

            i, j, k = known_index[next_idx]
            i_bar, j_bar, k_bar = unknown_index[next_unknown_idx]

            I_1 = heaviside(self.gamma - np.dot(np.dot(A[i], R[:, :, k]), A[j])
                            + np.dot(np.dot(A[i_bar], R[:, :, k_bar]), A[j_bar]))
            I_2 = heaviside(self.gamma_p - np.dot(np.dot(A[i], R[:, :, k]), A[j]))

            # updating parameters
            A[i] -= learning_rate * (-(I_1 + I_2 * self.c_e) * np.dot(R[:, :, k], A[j]))
            A[j] -= learning_rate * (-(I_1 + I_2 * self.c_e) * np.dot(R[:, :, k].T, A[i]))
            R[:, :, k] -= learning_rate * (-(I_1 + I_2 * self.c_e) * np.outer(A[i], A[j]))

            A[i_bar] -= learning_rate * (I_1 * np.dot(R[:, :, k_bar], A[j_bar]))
            A[j_bar] -= learning_rate * (I_1 * np.dot(R[:, :, k_bar].T, A[i_bar]))
            R[:, :, k_bar] -= learning_rate * (I_1 * np.outer(A[i_bar], A[j_bar]))

            # unit vector projection (this could be improved by using l1 projection alg.)
            # converting learned matrix to rotational matrix
            if I_1 != 0 or I_2 != 0:
                A[i] /= np.linalg.norm(A[i], ord=2)
                A[j] /= np.linalg.norm(A[j], ord=2)
                U, sigma, V = np.linalg.svd(R[:, :, k])
                R[:, :, k] = np.dot(U, V.T)

            if I_1 != 0:
                A[i_bar] /= np.linalg.norm(A[i_bar], ord=2)
                A[j_bar] /= np.linalg.norm(A[j_bar], ord=2)
                U, sigma, V = np.linalg.svd(R[:, :, k_bar])
                R[:, :, k_bar] = np.dot(U, V.T)

            it += 1
            learning_rate = self.alpha_0 / np.sqrt(it)

            if it >= max_iter:
                converged = True

            if it % 100 == 0:
                print(it, learning_rate, np.sum(np.abs(T - self.reconstruct(A, R))))

        return A, R

    def reconstruct(self, A, R):
        T = np.zeros((A.shape[0], A.shape[0], R.shape[2]))

        for i in range(R.shape[2]):
            T[:, :, i] = np.dot(A, np.dot(R[:, :, i], A.T))

        return T


def test():
    mat = loadmat('../data/alyawarradata.mat')
    T = np.array(mat['Rs'], np.float32)
    E, K = T.shape[0], T.shape[2]
    max_iter = E * E * K

    latent_dimension = 5

    model = AMDC(latent_dimension)
    model.learn(T, max_iter)


if __name__ == '__main__':
    test()

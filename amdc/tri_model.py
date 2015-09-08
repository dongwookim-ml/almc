__author__ = 'Dongwoo Kim'

import numpy as np
from sklearn.metrics import precision_recall_curve, auc
import path_tool

heaviside = lambda x: 1 if x >= 0 else 0


class TAMDC:
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
        @param p_idx: observed index of positive triples, all indices are raveled by np.ravel_multi_index
        @param n_idx: observed index of negative triples
        @param max_iter: maximum number of iterations
        @param e_gap: evaluation gap
        @return: A, R, r_error
                A: [E x D] latent feature vector of entities
                R: [D x D x K] rotation matrix for each entity
                r_error: list of reconstruction errors at each evaluation point
        """
        E, K = T.shape[0], T.shape[2]

        np_idx = np.setdiff1d(range(np.prod(T.shape)), p_idx)  # not positive index
        nn_idx = np.setdiff1d(range(np.prod(T.shape)), n_idx)  # not negative index

        A = np.random.random([E, self.D]) - 0.5
        A /= np.linalg.norm(A, axis=1)[:, np.newaxis]
        R = np.zeros([self.D, self.D, K])  # rotation matrices
        for k in range(K):
            R[:, :, k] = np.identity(self.D)

        tri_indices = path_tool.tri_index(T)

        r_error = list()

        it = 0
        converged = False
        learning_rate = self.alpha_0
        while not converged:

            if np.random.randint(100) % 2 == 0:
                # next_idx = np.random.randint(len(p_idx))
                # next_np_idx = np.random.randint(len(np_idx))
                #
                # i, j, k = np.unravel_index(p_idx[next_idx], T.shape)
                # i_bar, j_bar, k_bar = np.unravel_index(np_idx[next_np_idx], T.shape)

                next_idx = np.random.randint(len(tri_indices))
                (i,j,a), (j,k,b), (i,k,c) = tri_indices[next_idx]
                (i_bar,j_bar,a_bar), (j_bar, k_bar, b_bar), (i_bar, k_bar, c_bar) = path_tool.sample_broken_tri(T)

                I_1 = heaviside(self.gamma - np.dot(np.dot(np.dot(A[i], R[:,:,b]), R[:,:,a]), A[k])
                                + np.dot(np.dot(np.dot(A[i_bar], R[:,:,b_bar]), R[:,:,a_bar]), A[k_bar]))
                I_2 = heaviside(self.gamma_p - np.dot(np.dot(np.dot(A[i], R[:,:,b]), R[:,:,a]), A[k]))
                # I_1 = heaviside(self.gamma - np.dot(np.dot(A[i], R[:, :, k]), A[j])
                #                 + np.dot(np.dot(A[i_bar], R[:, :, k_bar]), A[j_bar]))
                # I_2 = heaviside(self.gamma_p - np.dot(np.dot(A[i], R[:, :, k]), A[j]))

                # updating parameters
                if I_1 != 0 or I_2 != 0:
                    a_i, a_k, r_a, r_b = A[i].copy(), A[k].copy(), R[:, :, a].copy(), R[:,:,b].copy()
                    A[i] -= learning_rate * (-(I_1 + I_2 * self.c_e) * np.dot(np.dot(r_b,r_a), a_k))
                    A[k] -= learning_rate * (-(I_1 + I_2 * self.c_e) * np.dot(np.dot(r_b,r_a).T, a_i))
                    R[:, :, a] -= learning_rate * (-(I_1 + I_2 * self.c_e) * np.outer(a_i, np.dot(r_b, a_k)))
                    R[:, :, b] -= learning_rate * (-(I_1 + I_2 * self.c_e) * np.outer(np.dot(r_a.T, a_i), a_k))

                    A[i] /= np.linalg.norm(A[i], ord=2)
                    A[k] /= np.linalg.norm(A[k], ord=2)
                    U, sigma, V = np.linalg.svd(R[:, :, a])
                    R[:, :, a] = np.dot(U, V)
                    U, sigma, V = np.linalg.svd(R[:, :, b])
                    R[:, :, b] = np.dot(U, V)

                if I_1 != 0:
                    a_i_bar, a_k_bar, r_a_bar, r_b_bar = A[i_bar].copy(), A[k_bar].copy(), R[:, :, a_bar].copy(), R[:,:,b_bar].copy()
                    A[i_bar] -= learning_rate * (I_1 * np.dot(np.dot(r_b_bar,r_a_bar), a_k_bar))
                    A[k_bar] -= learning_rate * (I_1 * np.dot(np.dot(r_b_bar,r_a_bar).T, a_i_bar))
                    R[:, :, a_bar] -= learning_rate * (I_1 * np.outer(a_i_bar, np.dot(r_b_bar,a_k_bar)))
                    R[:, :, b_bar] -= learning_rate * (I_1 * np.outer(np.dot(r_a_bar.T,a_i_bar),a_k_bar))

                    A[i_bar] /= np.linalg.norm(A[i_bar], ord=2)
                    A[k_bar] /= np.linalg.norm(A[k_bar], ord=2)
                    U, sigma, V = np.linalg.svd(R[:, :, a_bar])
                    R[:, :, a_bar] = np.dot(U, V)
                    U, sigma, V = np.linalg.svd(R[:, :, b_bar])
                    R[:, :, b_bar] = np.dot(U, V)

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

                    A[i] /= np.linalg.norm(A[i], ord=2)
                    A[j] /= np.linalg.norm(A[j], ord=2)
                    U, sigma, V = np.linalg.svd(R[:, :, k])
                    R[:, :, k] = np.dot(U, V)

                if I_3 != 0:
                    a_i_bar, a_j_bar, r_k_bar = A[i_bar].copy(), A[j_bar].copy(), R[:, :, k_bar].copy()
                    A[i_bar] -= learning_rate * (I_3 * self.c_n * np.dot(r_k_bar, a_j_bar))
                    A[j_bar] -= learning_rate * (I_3 * self.c_n * np.dot(r_k_bar.T, a_i_bar))
                    R[:, :, k_bar] -= learning_rate * (I_3 * self.c_n * np.outer(a_i_bar, a_j_bar))

                    A[i_bar] /= np.linalg.norm(A[i_bar], ord=2)
                    A[j_bar] /= np.linalg.norm(A[j_bar], ord=2)
                    U, sigma, V = np.linalg.svd(R[:, :, k_bar])
                    R[:, :, k_bar] = np.dot(U, V)

            if it >= max_iter:
                converged = True

            if it % e_gap == 0:
                T_bar = self.reconstruct(A, R)
                _T = T.copy()
                _T[_T == -1] = 0
                T_bar = (T_bar + 1.) / 2.

                from sklearn.metrics import roc_auc_score
                err = 0.
                for k in range(K):
                    err += roc_auc_score(_T[:, :, k].flatten(), T_bar[:, :, k].flatten())
                err /= float(K)

                print('Iter %d, ROC-AUC: %.5f' % (it, err))

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

    def test(self, T, A, R, test_idx):
        T_bar = self.reconstruct(A, R)
        T_bar[T_bar > 0] = 1
        T_bar[T_bar <= 0] = -1

        idx = np.unravel_index(test_idx, T.shape)

        prec, recall, _ = precision_recall_curve(T[idx], T_bar[idx])
        return auc(recall, prec)


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

    model = TAMDC(latent_dimension)
    model.learn(T, p_idx, n_idx, max_iter, e_gap=10000)


if __name__ == '__main__':
    test()

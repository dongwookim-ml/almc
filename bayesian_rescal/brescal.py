import numpy as np
import itertools
from numpy.random import multivariate_normal

class BayesianRescal:
    def __init__(self, n_dim, sigma_e=1., sigma_x=1., sigma_r=1., compute_score=True):
        self.n_dim = n_dim
        self.sigma_e = sigma_e
        self.sigma_x = sigma_x
        self.sigma_r = sigma_r
        self.compute_score = compute_score

    def fit(self, X, max_iter=100):
        self.n_relations = X.shape[0]
        self.n_entities = X.shape[1]

        self.E = np.zeros([self.n_entities, self.n_dim])
        self.R = np.zeros([self.n_relations, self.n_dim, self.n_dim])

        self._gibbs(X, max_iter)

    def _gibbs(self, X, max_iter):
        for i in range(max_iter):
            self._sample_entities(X)
            self._sample_relations(X)

            if self.compute_score:
                _score = self.score(X)
                print('Iter %d: Score %.3f' % (i, _score))

    def _sample_entities(self, X):
        for i in range(self.n_entities):
            _lambda = np.identity(self.n_dim)
            xi = np.zeros(self.n_dim)

            for k in range(self.n_relations):
                tmp = np.dot(self.R[k], self.E.T)       # D x E
                tmp2 = np.dot(tmp, tmp.T)
                tmp3 = np.dot(self.R[k].T, self.E.T)
                tmp4 = np.dot(tmp3, tmp3.T)
                _lambda += tmp2 + tmp4

                xi += np.sum(X[k, i, :] * tmp, 1) + np.sum(X[k,:,i] * tmp3, 1)

            _lambda *= 1./self.sigma_x
            inv_lambda = np.linalg.inv(_lambda)
            mu = 1./self.sigma_x * np.dot(inv_lambda, xi)
            self.E[i] = multivariate_normal(mu, inv_lambda)

    def _sample_relations(self, X):
        for k in range(self.n_relations):
            _lambda = np.identity(self.n_dim * self.n_dim)
            xi = np.zeros(self.n_dim * self.n_dim)
            for (i,j) in itertools.permutations(range(self.n_entities), 2):
                tmp = np.kron(self.E[i], self.E[j])
                _lambda += np.outer(tmp,tmp)
                xi += tmp*X[k, i, j]

            _lambda *= 1./self.sigma_x
            inv_lambda = np.linalg.inv(_lambda)
            mu = 1./self.sigma_x * np.dot(inv_lambda, xi)
            self.R[k] = multivariate_normal(mu, inv_lambda).reshape([10,10])

    def _reconstruct(self):
        _X = np.zeros([self.n_relations, self.n_entities, self.n_entities])

        for k in range(self.n_relations):
            _X[k] = np.dot(np.dot(self.E.T, self.R), self.E)

        return _X

    def score(self, X):
        """

        Returns the log-likelihood of the model
        -------

        """
        score = 0
        for k in range(self.n_relations):
            mean = np.dot(np.dot(self.E, self.R[k]), self.E.T)
            score += np.sum((X[k] - mean)**2. / (2.*self.sigma_x))

        return score


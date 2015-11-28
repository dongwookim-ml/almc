import itertools
import logging

import numpy as np
from numpy.random import multivariate_normal, gamma

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class BayesianRescal:
    def __init__(self, n_dim, var_e=1., var_x=1., var_r=1., compute_score=True, sample_prior=True, e_alpha=1.,
                 e_beta=1., x_alpha=1., x_beta=1., r_alpha=1., r_beta=1., prior_sample_gap=10):
        self.n_dim = n_dim
        self.var_e = var_e
        self.var_x = var_x
        self.var_r = var_r
        self.compute_score = compute_score

        self.sample_prior = sample_prior
        self.e_alpha = e_alpha
        self.e_beta = e_beta
        self.r_alpha = r_alpha
        self.r_beta = r_beta
        self.prior_sample_gap = prior_sample_gap

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

            if self.sample_prior and (i + 1) % self.prior_sample_gap == 0:
                self._sample_prior()

            if self.compute_score:
                _score = self.score(X)
                logger.info("Iter %d: Score %.3f", i, _score)

    def _sample_prior(self):
        self._sample_var_r()
        self._sample_var_e()

    def _sample_var_r(self):
        self.var_r = 1. / gamma(0.5 * self.n_relations * self.n_dim * self.n_dim + self.r_alpha,
                                1. / (0.5 * np.sum(self.R ** 2) + self.r_beta))
        logger.debug("Sampled var_r %.3f", self.var_r)

    def _sample_var_e(self):
        self.var_e = 1. / gamma(0.5 * self.n_entities * self.n_dim + self.e_alpha,
                                1. / (0.5 * np.sum(self.E ** 2) + self.e_beta))
        logger.debug("Sampled var_e %.3f", self.var_e)

    def _sample_entities(self, X):
        for i in range(self.n_entities):
            _lambda = np.identity(self.n_dim)
            xi = np.zeros(self.n_dim)

            for k in range(self.n_relations):
                tmp = np.dot(self.R[k], self.E.T)  # D x E
                tmp2 = np.dot(tmp, tmp.T)
                tmp3 = np.dot(self.R[k].T, self.E.T)
                tmp4 = np.dot(tmp3, tmp3.T)
                _lambda += tmp2 + tmp4

                xi += np.sum(X[k, i, :] * tmp, 1) + np.sum(X[k, :, i] * tmp3, 1)

            _lambda *= 1. / self.var_e
            inv_lambda = np.linalg.inv(_lambda)
            mu = 1. / self.var_x * np.dot(inv_lambda, xi)
            self.E[i] = multivariate_normal(mu, inv_lambda)

    def _sample_relations(self, X):
        for k in range(self.n_relations):
            _lambda = np.identity(self.n_dim ** 2)
            xi = np.zeros(self.n_dim * self.n_dim)
            for (i, j) in itertools.permutations(range(self.n_entities), 2):
                tmp = np.kron(self.E[i], self.E[j])
                _lambda += np.outer(tmp, tmp)
                xi += tmp * X[k, i, j]

            _lambda *= 1. / self.var_r
            inv_lambda = np.linalg.inv(_lambda)
            mu = 1. / self.var_x * np.dot(inv_lambda, xi)
            self.R[k] = multivariate_normal(mu, inv_lambda).reshape([10, 10])

    def _reconstruct(self):
        _X = np.zeros([self.n_relations, self.n_entities, self.n_entities])

        for k in range(self.n_relations):
            _X[k] = np.dot(np.dot(self.E.T, self.R), self.E)

        return _X

    # noinspection PyTypeChecker
    def score(self, X):
        """

        Compute the log-likelihood of the model
        -------

        Parameters
        ----------
        X

        """
        score = 0
        for k in range(self.n_relations):
            mean = np.dot(np.dot(self.E, self.R[k]), self.E.T)
            score -= np.sum((X[k] - mean) ** 2. / (2. * self.var_x))  # p(x|E,R)
            score -= np.sum(self.R[k] ** 2 / (2. * self.var_r))  # p(R)

        for i in range(self.n_entities):
            score -= np.sum(self.E[i] ** 2 / (2. * self.var_e))  # p(E)

        if self.sample_prior:
            score += (self.e_alpha - 1.) * np.log(self.var_e) - self.e_beta * self.var_e
            score += (self.r_alpha - 1.) * np.log(self.var_r) - self.r_beta * self.var_r

        return score

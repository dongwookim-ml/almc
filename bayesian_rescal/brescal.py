import itertools
import logging
import time
import numpy as np
from numpy.random import multivariate_normal, gamma

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BayesianRescal:
    def __init__(self, n_dim, var_e=1., var_x=1., var_r=1., compute_score=True, sample_prior=False, e_alpha=1.,
                 e_beta=1., x_alpha=1., x_beta=1., r_alpha=1., r_beta=1., prior_sample_gap=5, controlled_var=False,
                 obs_var=1., unobs_var=10.):
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

        self.controlled_var = controlled_var
        self.obs_var = obs_var
        self.unobs_var = unobs_var

    def fit(self, X, max_iter=100):
        self.n_relations = X.shape[0]
        self.n_entities = X.shape[1]

        #self.E = np.zeros([self.n_entities, self.n_dim])
        self.E = np.random.random([self.n_entities,self.n_dim])
        #self.R = np.zeros([self.n_relations, self.n_dim, self.n_dim])
        self.R = np.random.random([self.n_relations, self.n_dim, self.n_dim])

        # for controlled variance
        if self.controlled_var:
            self.var_X = np.ones_like(X) * self.obs_var
            self.var_X[X == 0] = self.unobs_var

        self._gibbs(X, max_iter)

    def _gibbs(self, X, max_iter):
        logger.info("[INIT] LL: %.3f | fit: %0.5f", self.score(X), self._compute_fit(X)[0])

        for i in range(max_iter):
            tic = time.time()
            self._sample_entities(X)
            self._sample_relations(X)

            if self.sample_prior and (i + 1) % self.prior_sample_gap == 0:
                self._sample_prior()
            toc = time.time()

            if self.compute_score:
                _score = self.score(X)
                _fit, _ = self._compute_fit(X)
                logger.info("[%3d] LL: %.3f | fit: %0.5f |  sec: %.3f", i, _score, _fit, (toc - tic))
            else:
                logger.info("[%3d] sec: %.3f", i, (toc - tic))

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
            self.E[i] *= 0
            _lambda = np.zeros([self.n_dim, self.n_dim])
            xi = np.zeros(self.n_dim)

            if self.controlled_var:
                for k in range(self.n_relations):
                    tmp = np.dot(self.R[k], self.E.T) * (1. / self.var_X[k, i, :])  # D x E
                    tmp2 = np.dot(self.R[k].T, self.E.T) * (1. / self.var_X[k, :, i])
                    _lambda += np.dot(tmp, tmp.T) + np.dot(tmp2, tmp2.T)
                    xi += np.sum((1. / self.var_X[k, i, :]) * X[k, i, :] * tmp, 1) \
                          + np.sum((1. / self.var_X[k, :, i]) * X[k, :, i] * tmp2, 1)

                _lambda += (1. / self.var_e) * np.identity(self.n_dim)
                inv_lambda = np.linalg.inv(_lambda)
                mu = np.dot(inv_lambda, xi)

            else:
                for k in range(self.n_relations):
                    tmp = np.dot(self.R[k], self.E.T)  # D x E
                    tmp2 = np.dot(self.R[k].T, self.E.T)
                    _lambda += np.dot(tmp, tmp.T) + np.dot(tmp2, tmp2.T)
                    xi += np.sum(X[k, i, :] * tmp, 1) + np.sum(X[k, :, i] * tmp2, 1)

                _lambda *= 1. / self.var_x
                _lambda += (1. / self.var_e) * np.identity(self.n_dim)
                inv_lambda = np.linalg.inv(_lambda)
                mu = (1. / self.var_x) * np.dot(inv_lambda, xi)

            self.E[i] = multivariate_normal(mu, inv_lambda)

    def _sample_relations(self, X):

        if self.controlled_var:
            EXE = np.kron(self.E, self.E)

            for k in range(self.n_relations):
                _EXE = EXE * (1. / self.var_X[k, :, :].flatten()[:, np.newaxis])
                _lambda = np.dot(EXE.T, EXE)  # D^2 x D^2
                _lambda += (1. / self.var_r) * np.identity(self.n_dim ** 2)
                inv_lambda = np.linalg.inv(_lambda)

                xi = np.sum(_EXE * X[k].flatten()[:, np.newaxis] * (1. / self.var_X[k].flatten()[:, np.newaxis]), 0)
                mu = np.dot(inv_lambda, xi)
                self.R[k] = multivariate_normal(mu, inv_lambda).reshape([self.n_dim, self.n_dim])

        else:
            EXE = np.kron(self.E, self.E)
            _lambda = np.dot(EXE.T, EXE)  # D^2 x D^2
            _lambda *= (1. / self.var_x)
            _lambda += (1. / self.var_r) * np.identity(self.n_dim ** 2)
            inv_lambda = np.linalg.inv(_lambda)

            for k in range(self.n_relations):
                xi = np.sum(EXE * X[k].flatten()[:, np.newaxis], 0)
                mu = (1. / self.var_x) * np.dot(inv_lambda, xi)
                self.R[k] = multivariate_normal(mu, inv_lambda).reshape([self.n_dim, self.n_dim])

    def _reconstruct(self):
        _X = np.zeros([self.n_relations, self.n_entities, self.n_entities])

        for k in range(self.n_relations):
            _X[k] = np.dot(np.dot(self.E, self.R[k]), self.E.T)

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
        from scipy.stats import norm, multivariate_normal

        if not hasattr(self, 'n_relations'):
            self.n_entities, self.n_relations, _ = X.shape

        score = 0
        for k in range(self.n_relations):
            mean = np.dot(np.dot(self.E, self.R[k]), self.E.T)
            #score -= np.sum((X[k] - mean) ** 2. / (2. * self.var_x))  # p(x|E,R)
            #score -= np.sum(self.R[k] ** 2 / (2. * self.var_r))  # p(R)
            #for i,j in itertools.product(range(self.n_entities), repeat=2):
            score += np.sum(norm.logpdf(X[k].flatten(), mean.flatten(), np.sqrt(self.var_x)))
            score += np.sum(norm.logpdf(self.R[k].flatten(), 0, np.sqrt(self.var_r)))

        for i in range(self.n_entities):
            #score -= np.sum(self.E[i] ** 2 / (2. * self.var_e))  # p(E)
            a = multivariate_normal.logpdf(self.E[i], np.zeros(self.n_dim), np.identity(self.n_dim)*self.var_e)
            b = np.sum(norm.logpdf(self.E[i], 0, np.sqrt(self.var_e)))

            assert np.all(np.isclose(a,b))
            score += multivariate_normal.logpdf(self.E[i], np.zeros(self.n_dim), np.identity(self.n_dim)*self.var_e)

        if self.sample_prior:
            score += (self.e_alpha - 1.) * np.log(self.var_e) - self.e_beta * self.var_e
            score += (self.r_alpha - 1.) * np.log(self.var_r) - self.r_beta * self.var_r

        return score

    # metric used in rescal
    def _compute_fit(self, X):
        from numpy.linalg import norm

        f = 0
        r_error = 0
        sumNorm = np.sum(X ** 2)
        for k in range(self.n_relations):
            mean = np.dot(np.dot(self.E, self.R[k]), self.E.T)
            _diff = X[k] - mean
            f += norm(_diff) ** 2
            r_error += np.sum(np.abs(_diff))

        return (1. - f / sumNorm), r_error

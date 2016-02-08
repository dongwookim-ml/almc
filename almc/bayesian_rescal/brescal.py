import logging
import time
import itertools
import numpy as np
import concurrent.futures
from numpy.random import multivariate_normal, gamma
from sklearn.metrics import mean_squared_error

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

_E_ALPHA = 1.
_E_BETA = 1.
_R_ALPHA = 1.
_R_BETA = 1.
_P_SAMPLE_GAP = 5
_P_SAMPLE = False
_PARALLEL = True
_MAX_THREAD = 4

_VAR_E = 1.
_VAR_R = 1.
_VAR_X = 0.01

_DEST = ''


def gen_random_tensor(n_dim, n_entity, n_relation, var_e=1., var_r=1, var_x=0.01):
    """
    Generate a random tensor following the generative process of BRESCAL

    Parameters
    ----------
    n_dim: int
        latent dimensionaitly of entities and relations
    n_entity: int
        number of entities to be generated
    n_relation: int
        number of relations to be generated
    var_e: float
        isotropic variance of entity vector
    var_r: float
        isotropic variance of relation matrix
    var_x: float
        variance of triple

    Returns
    -------
        array, shape (n_relation, n_entity, n_entity)

    """
    e_mean = np.zeros(n_dim)
    r_mean = np.zeros(n_dim ** 2)
    E = np.random.multivariate_normal(e_mean, np.identity(n_dim) * var_e, size=n_entity)
    R = np.zeros([n_relation, n_dim, n_dim])
    T = np.zeros([n_relation, n_entity, n_entity])
    for k in range(n_relation):
        R[k] = np.random.multivariate_normal(r_mean, np.identity(n_dim ** 2) * var_r).reshape(n_dim, n_dim)
    for k in range(n_relation):
        ERET = np.dot(np.dot(E, R[k]), E.T)
        for i, j in itertools.product(range(n_entity), repeat=2):
            T[k, i, j] = np.random.normal(ERET[i, j], var_x)
    return T


class BayesianRescal:
    """
    Bayesian RESCAL
    generalisation of RESCAL (Nickel et.al., ICML 2011) with prior over latent features
    """

    def __init__(self, n_dim, compute_score=True, controlled_var=False, obs_var=0.01, unobs_var=10.,
                 eval_fn=mean_squared_error, **kwargs):
        """

        Parameters
        ----------
        n_dim: int
            latent dimensionality of the model
        compute_score: boolean, optional
            Compute the joint likelihood of the model
        controlled_var: boolean, optional
            Controlled variance approach. Place different variances on observed and unobserved triples
        obs_var: float, optional
            variance of observed triples
        unobs_var: float, optional
            variance of unobserved triples, only applied when `controlled_var` is True
        eval_fn: `sklearn.metric.mean_squred_error`
            Evaluate the current fit of the model based on `eval_fn`. By default, the model compute the
            mean squared error between current observation and reconstructed tensor.
        sample_prior: boolean, optional
            If True, sample priors of the variance `var_e` and `var_r`
        sample_prior_gap: int
            If `sample_prior` is True, then sample priors every `sample_prior_gap` iterations
        e_alpha: float
            shape parameter of the prior of `var_e`
        e_beta: float
            scale parameter of the prior of `var_e`
        r_alpha: float
            shape parameter of the prior of `var_r`
        r_beta: float
            scale parameter of the prior of `var_r`
        parallel: boolean
            If True, use multi-thread to sample relations
        max_thread: int
            maximum number of threads
        dest: str
            destination path of the output model file

        Returns
        -------

        """
        self.n_dim = n_dim

        self.var_e = kwargs.pop('var_e', _VAR_E)
        self.var_r = kwargs.pop('var_r', _VAR_R)
        self.var_x = kwargs.pop('var_x', _VAR_X)

        self.compute_score = compute_score

        self.sample_prior = kwargs.pop('sample_prior', _P_SAMPLE)
        self.prior_sample_gap = kwargs.pop('prior_sample_gap', _P_SAMPLE_GAP)

        self.e_alpha = kwargs.pop('e_alpha', _E_ALPHA)
        self.e_beta = kwargs.pop('e_beta', _E_BETA)
        self.r_alpha = kwargs.pop('r_alpha', _R_ALPHA)
        self.r_beta = kwargs.pop('r_beta', _R_BETA)

        self.parallelize = kwargs.pop('parallel', _PARALLEL)
        self.max_thread = kwargs.pop('max_thread', _MAX_THREAD)

        self.dest = kwargs.pop('dest', _DEST)

        self.controlled_var = controlled_var
        self.obs_var = obs_var
        self.unobs_var = unobs_var

        self.eval_fn = eval_fn

    def __getstate__(self):
        """Remove `var_X` when save model
        """
        d = dict(self.__dict__)
        if self.controlled_var:
            del d['var_X']
        return d

    def fit(self, X, max_iter=100):
        """Infer vector and matrix embedding of entities and relations, respectively

        Parameters
        ----------
        X: ndarray, shape (n_relations, n_entities, n_entities)
            Full tensor to be factorised.
        max_iter: int
            maximum number of Gibbs sampling iterations

        """
        self.n_relations = X.shape[0]
        self.n_entities = X.shape[1]

        self.E = np.random.random([self.n_entities, self.n_dim])
        self.R = np.random.random([self.n_relations, self.n_dim, self.n_dim])

        # for controlled variance
        if self.controlled_var:
            self.var_X = np.ones_like(X) * self.obs_var
            self.var_X[X == 0] = self.unobs_var

        self._gibbs(X, max_iter)

        if len(self.dest) > 0:
            self._save_model()

    def _gibbs(self, X, max_iter):
        """Gibbs sampling for entities and relations

        Parameters
        ----------
        X: ndarray, shape (n_relations, n_entities, n_entities)
            Full tensor to be fatorised
        max_iter:
            maximum number of Gibbs sampling iterations
        """
        logger.info("[INIT] LL: %.3f | fit: %0.5f", self.score(X), self._compute_fit(X))

        for i in range(max_iter):
            tic = time.time()
            self._sample_entities(X, self.E, self.R)
            self._sample_relations(X, self.E, self.R)

            if self.sample_prior and (i + 1) % self.prior_sample_gap == 0:
                self._sample_prior()

            toc = time.time()

            if self.compute_score:
                _score = self.score(X)
                _fit = self._compute_fit(X)
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

    def _sample_entities(self, X, E, R):
        """Sample entity vectors

        Parameters
        ----------
        X: ndarray, shape (n_relations, n_entities, n_entities)
            full tensor
        E: ndarray, shape (n_entities, n_dim)
        R: ndarray, shape (n_relations, n_dim, n_dim)

        Returns
        -------
        E: ndarray, shape (n_entities, n_dim)
            returns newly sampled entities
        """
        for i in range(self.n_entities):
            self._sample_entity(X, E, R, i)
        return E

    def _sample_entity(self, X, E, R, i):
        E[i] *= 0
        _lambda = np.zeros([self.n_dim, self.n_dim])
        xi = np.zeros(self.n_dim)

        if self.controlled_var:
            for k in range(self.n_relations):
                tmp = np.dot(R[k], E.T)  # D x E
                tmp2 = np.dot(R[k].T, E.T)
                _lambda += np.dot(tmp * (1. / self.var_X[k, i, :]), tmp.T) \
                           + np.dot(tmp2 * (1. / self.var_X[k, :, i]), tmp2.T)

                xi += np.sum((1. / self.var_X[k, i, :]) * X[k, i, :] * tmp, 1) \
                      + np.sum((1. / self.var_X[k, :, i]) * X[k, :, i] * tmp2, 1)

            _lambda += (1. / self.var_e) * np.identity(self.n_dim)
            inv_lambda = np.linalg.inv(_lambda)
            mu = np.dot(inv_lambda, xi)

        else:
            for k in range(self.n_relations):
                tmp = np.dot(R[k], E.T)  # D x E
                tmp2 = np.dot(R[k].T, E.T)
                _lambda += np.dot(tmp, tmp.T) + np.dot(tmp2, tmp2.T)
                xi += np.sum(X[k, i, :] * tmp, 1) + np.sum(X[k, :, i] * tmp2, 1)

            xi *= (1. / self.var_x)
            _lambda *= 1. / self.var_x
            _lambda += (1. / self.var_e) * np.identity(self.n_dim)
            inv_lambda = np.linalg.inv(_lambda)
            mu = np.dot(inv_lambda, xi)

        E[i] = multivariate_normal(mu, inv_lambda)
        return E[i]

    def _sample_relation(self, X, E, R, k, EXE, inv_lambda=None):
        """Sample relation matrix

        Parameters
        ----------
        X: ndarray, shape (n_relations, n_entities, n_entities)
        E: ndarray, shape (n_entities, n_dim)
        R: ndarray, shape (n_relations, n_dim, n_dim)
        k: int
            relation to be sampled
        EXE: ndarray, shape (n_entities * n_entities, n_dim * n_dim)
            kronecker product between entities
        inv_lambda:ndarray, shape (n_dim * n_dim, n_dim * n_dim)
            inverse of covariance matrix

        Returns
        -------
        R[k]: ndarray, shape (n_dim, n_dim)
            returns newly sampled relation matrix R
        k: int
            index of sampled relation
        """
        if not self.controlled_var:
            xi = np.sum(EXE * X[k].flatten()[:, np.newaxis], 0)
            mu = (1. / self.var_x) * np.dot(inv_lambda, xi)
            R[k] = multivariate_normal(mu, inv_lambda).reshape([self.n_dim, self.n_dim])

        else:
            tmp = EXE * (1. / self.var_X[k, :, :].flatten()[:, np.newaxis])
            _lambda = np.dot(tmp.T, EXE)
            _lambda += (1. / self.var_r) * np.identity(self.n_dim ** 2)
            inv_lambda = np.linalg.inv(_lambda)

            xi = np.sum(EXE * X[k].flatten()[:, np.newaxis] * (1. / self.var_X[k, :, :].flatten()[:, np.newaxis]), 0)
            mu = np.dot(inv_lambda, xi)
            R[k] = multivariate_normal(mu, inv_lambda).reshape([self.n_dim, self.n_dim])

        return R[k], k

    def _sample_relations(self, X, E, R):
        """Sample relation matrices

        Parameters
        ----------
        X: ndarray, shape (n_relations, n_entities, n_entities)
        E: ndarray, shape (n_entities, n_dim)
        R: ndarray, shape (n_relations, n_dim, n_dim)

        Returns
        -------
        R: ndarray, shape (n_relations, n_dim, n_dim)
            returns newly sampled relation matrices
        """
        EXE = np.kron(E, E)

        if not self.controlled_var:
            _lambda = np.dot(EXE.T, EXE)  # D^2 x D^2
            _lambda *= (1. / self.var_x)
            _lambda += (1. / self.var_r) * np.identity(self.n_dim ** 2)
            inv_lambda = np.linalg.inv(_lambda)

            if self.parallelize:
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_thread) as executor:
                    fs = [executor.submit(self._sample_relation, X, E, R, k, EXE, inv_lambda) for k in
                          range(self.n_relations)]
                concurrent.futures.wait(fs)
            else:
                [self._sample_relation(X, E, R, k, EXE, inv_lambda) for k in range(self.n_relations)]

        else:
            if self.parallelize:
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_thread) as executor:
                    fs = [executor.submit(self._sample_relation, X, E, R, k, EXE) for k in range(self.n_relations)]
                concurrent.futures.wait(fs)

            else:
                [self._sample_relation(X, E, R, k, EXE) for k in range(self.n_relations)]

        return R

    def _reconstruct(self):
        """ Reconstruct tensor by the current model parameters

        Returns
        -------
        _X: ndarray, shape (n_relations, n_entities, n_entities)
            Reconstructed tensor
        """
        _X = np.zeros([self.n_relations, self.n_entities, self.n_entities])

        for k in range(self.n_relations):
            _X[k] = np.dot(np.dot(self.E, self.R[k]), self.E.T)

        return _X

    def score(self, X):
        from scipy.stats import norm, multivariate_normal

        if not hasattr(self, 'n_relations'):
            self.n_entities, self.n_relations, _ = X.shape

        score = 0
        for k in range(self.n_relations):
            mean = np.dot(np.dot(self.E, self.R[k]), self.E.T)
            if self.controlled_var:
                score += np.sum(norm.logpdf(X[k].flatten(), mean.flatten(), np.sqrt(self.var_X[k].flatten())))
            else:
                score += np.sum(norm.logpdf(X[k].flatten(), mean.flatten(), np.sqrt(self.var_x)))

            score += np.sum(norm.logpdf(self.R[k].flatten(), 0, np.sqrt(self.var_r)))

        for i in range(self.n_entities):
            score += multivariate_normal.logpdf(self.E[i], np.zeros(self.n_dim), np.identity(self.n_dim) * self.var_e)

        if self.sample_prior:
            score += (self.e_alpha - 1.) * np.log(self.var_e) - self.e_beta * self.var_e
            score += (self.r_alpha - 1.) * np.log(self.var_r) - self.r_beta * self.var_r

        return score

    def _compute_fit(self, X):
        """Compute reconstruction error

        Parameters
        ----------
        X: ndarray, shape (n_relations, n_entities, n_entities)

        Returns
        -------
        _fit: float
            returns the evaluation error between reconstructed tensor and `X`
            based on `eval_fn`
        """
        _X = self._reconstruct()
        _fit = self.eval_fn(X.flatten(), _X.flatten())
        return _fit

    def _save_model(self):
        """Save the current parameters of the model
        """
        import pickle

        with open(self.dest, 'wb') as f:
            pickle.dump(self, f)

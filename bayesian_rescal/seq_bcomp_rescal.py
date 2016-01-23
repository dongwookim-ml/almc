import logging
import time
import itertools
import numpy as np
from numpy.random import multivariate_normal, gamma, multinomial
from sklearn.metrics import mean_squared_error

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

_E_ALPHA = 1.
_E_BETA = 1.
_R_ALPHA = 1.
_R_BETA = 1.
_P_SAMPLE_GAP = 5
_P_SAMPLE = False
_PARALLEL = False
_MAX_THREAD = 4
_POS_VAL = 1
_MC_MOVE = 1
_GIBBS_INIT = True
_GIBBS_ITER = 10
_PULL_SIZE = 1
_SAMPLE_ALL = True
_COMPUTE_SCORE = True

_VAR_E = 1.
_VAR_R = 1.
_VAR_X = 0.1
_VAR_COMP = 10.

_DEST = ''
_LOG = ''

ADDITIVE = 'additive'
MULTIPLICATIVE = 'multiplicative'

MIN_VAL = np.iinfo(np.int32).min


class PFBayesianCompRescal:
    def __init__(self, n_dim, compositionality='additive', n_particles=5,
                 eval_fn=mean_squared_error, **kwargs):
        """

        Parameters
        ----------
        n_dim: int
            Number of latent features (size of latent dimension)

        compute_score: boolean, default=True
            Compute log-likelihood of the model given training data. Also
            compute the evaluation function. Default evaluation function
            is the mean squared error

        sample_prior: boolean default=False
            If True, sample variances of entities and relations every ``prior_sample_gap``

        n_particles: int, default=5
            Number of particles for sequential Monte Carlo sampler

        eval_fn: callable
            Evaluate scoring function between training data and reconstructed data

        kwargs

        Returns
        -------

        """
        self.n_dim = n_dim
        self.compositionality = compositionality

        self._var_e = kwargs.pop('var_e', _VAR_E)
        self._var_r = kwargs.pop('var_r', _VAR_R)
        self.var_x = kwargs.pop('var_x', _VAR_X)
        self.var_comp = kwargs.pop('var_comp', _VAR_COMP)

        self.gibbs_init = kwargs.pop('gibbs_init', _GIBBS_INIT)
        self.gibbs_iter = kwargs.pop('gibbs_iter', _GIBBS_ITER)
        self.sample_all = kwargs.pop('sample_all', _SAMPLE_ALL)
        self.compute_score = kwargs.pop('compute_score', _COMPUTE_SCORE)
        self.sample_prior = kwargs.pop('sample_prior', _P_SAMPLE)
        self.prior_sample_gap = kwargs.pop('prior_sample_gap', _P_SAMPLE_GAP)
        self.e_alpha = kwargs.pop('e_alpha', _E_ALPHA)
        self.e_beta = kwargs.pop('e_beta', _E_BETA)
        self.r_alpha = kwargs.pop('r_alpha', _R_ALPHA)
        self.r_beta = kwargs.pop('r_beta', _R_BETA)
        self.parallelize = kwargs.pop('parallel', _PARALLEL)
        self.max_thread = kwargs.pop('max_thread', _MAX_THREAD)
        self.mc_move = kwargs.pop('mc_move', _MC_MOVE)
        self.pos_val = kwargs.pop('pos_val', _POS_VAL)
        self.dest = kwargs.pop('dest', _DEST)
        self.log = kwargs.pop('log', _LOG)

        if not len(kwargs) == 0:
            raise ValueError('Unknown keywords (%s)' % (kwargs.keys()))

        self.n_particles = n_particles
        self.p_weights = np.ones(n_particles) / n_particles
        self.eval_fn = eval_fn

        self.var_e = np.ones(self.n_particles) * self._var_e
        self.var_r = np.ones(self.n_particles) * self._var_r

    def __getstate__(self):
        d = dict(self.__dict__)
        return d

    def fit(self, X, obs_mask=None, max_iter=0):
        """
        Running the particle Thompson sampling with predefined parameters.

        Parameters
        ----------
        X : numpy.ndarray
            Fully observed tensor with shape (n_relations, n_entities, n_entities)
        obs_mask : numpy.ndarray, default=None
            Mask tensor of observed triples
        max_iter : int, default=0
            Maximum number of iterations for particle Thompson sampling
        Returns
        -------
        seq : numpy.ndarray
            Returns a sequence of selected triples over iterations.
        """
        self.n_entities = X.shape[1]
        self.n_relations = X.shape[0] + X.shape[0] ** 2
        self.n_pure_relations = X.shape[0]
        expand_X = np.zeros([self.n_relations, self.n_entities, self.n_entities])
        expand_X[:self.n_pure_relations] = X
        for k, (k1, k2) in enumerate(itertools.product(range(self.n_pure_relations), repeat=2)):
            expand_X[self.n_pure_relations + k] = np.dot(X[k1], X[k2])

        logger.info('Original size: %d', np.sum(X))
        logger.info('Expanded size %d', np.sum(expand_X))
        logger.info('Original shape: %s', X.shape)
        logger.info('Expanded shape: %s', expand_X.shape)

        self.E = np.zeros([self.n_particles, self.n_entities, self.n_dim])
        self.R = np.zeros([self.n_particles, self.n_pure_relations, self.n_dim, self.n_dim])

        if type(obs_mask) == type(None):
            # observation mask
            obs_mask = np.zeros_like(expand_X)

        cur_obs = np.zeros_like(expand_X)
        for k in range(self.n_pure_relations):
            cur_obs[k][obs_mask[k] == 1] = X[k][obs_mask[k] == 1]

        self.expand_obsmask(obs_mask, cur_obs)

        # sum of the observed values for each relation
        self.obs_sum = np.sum(np.sum(obs_mask, 1), 1)
        # list of relations of which one observation is founded at least
        self.valid_relations = np.nonzero(np.sum(np.sum(expand_X, 1), 1))[0]

        del expand_X

        max_possible_iter = int(np.prod([self.n_pure_relations, self.n_entities, self.n_entities]) - np.sum(
            obs_mask[:self.n_pure_relations]))
        if max_iter == 0 or max_iter > max_possible_iter:
            max_iter = max_possible_iter

        cur_obs[cur_obs.nonzero()] = 1
        if self.gibbs_init and np.sum(self.obs_sum) != 0:
            # initialize latent variables with gibbs sampling
            E = np.random.random([self.n_entities, self.n_dim])
            R = np.random.random([self.n_pure_relations, self.n_dim, self.n_dim])

            for gi in range(self.gibbs_iter):
                tic = time.time()
                self._sample_entities(cur_obs, obs_mask, E, R, self._var_e)
                self._sample_relations(cur_obs, obs_mask, E, R, self._var_r)
                logger.info("Gibbs Init %d: %f", gi, time.time() - tic)

            for p in range(self.n_particles):
                self.E[p] = E.copy()
                self.R[p] = R.copy()
        else:
            # random initialization
            for p in range(self.n_particles):
                self.E[p] = np.random.random([self.n_entities, self.n_dim])
                self.R[p] = np.random.random([self.n_pure_relations, self.n_dim, self.n_dim])

        if len(self.log) > 0:
            seq = list()
            for idx in self.particle_filter(X, cur_obs, obs_mask, max_iter):
                with open(self.log, 'a') as f:
                    f.write('%d,%d,%d\n' % (idx[0], idx[1], idx[2]))
                seq.append(idx)
        else:
            seq = [idx for idx in self.particle_filter(X, cur_obs, obs_mask, max_iter)]

        if len(self.dest) > 0:
            self._save_model(seq)

        return seq

    def particle_filter(self, X, cur_obs, mask, max_iter):

        pop = 0
        for i in range(max_iter):
            tic = time.time()

            next_idx = self.get_next_sample(mask)
            yield next_idx
            cur_obs[next_idx] = X[next_idx]
            mask[next_idx] = 1
            if cur_obs[next_idx] == self.pos_val:
                self.expand_obsmask(mask, cur_obs, next_idx[0])
            if X[next_idx] == self.pos_val:
                pop += 1

            cur_obs[cur_obs.nonzero()] = 1

            logger.info('[NEXT] %s: %.3f, population: %d/%d', str(next_idx), X[next_idx], pop, i)

            self.p_weights *= self.compute_particle_weight(next_idx, cur_obs, mask)
            self.p_weights /= np.sum(self.p_weights)

            cur_obs[cur_obs.nonzero()] = 1
            self.obs_sum = np.sum(np.sum(mask, 1), 1)

            logger.debug('[Additional Points] %d', np.sum(self.obs_sum[self.n_pure_relations:]))

            # effective sample size(ESS). When ESS is less than n_particles/2, resample particles
            ESS = 1. / np.sum((self.p_weights ** 2))

            if ESS < self.n_particles / 2.:
                self.resample()

            for m in range(self.mc_move):
                for p in range(self.n_particles):
                    self._sample_relations(cur_obs, mask, self.E[p], self.R[p], self.var_r[p])
                    self._sample_entities(cur_obs, mask, self.E[p], self.R[p], self.var_e[p])

            if self.sample_prior and i != 0 and i % self.prior_sample_gap == 0:
                self._sample_prior()

            toc = time.time()
            if self.compute_score:
                # compute training log-likelihood and error on observed data points
                _score = self.score(cur_obs, mask)
                _fit = self._compute_fit(cur_obs[:self.n_pure_relations], mask)
                logger.info("[%3d] LL: %.3f | fit(%s): %0.5f |  sec: %.3f", i, _score, self.eval_fn.__name__, _fit,
                            (toc - tic))
            else:
                logger.info("[%3d] sec: %.3f", i, (toc - tic))

    def expand_obsmask(self, mask, cur_obs, next_idx=-1):
        for k, (k1, k2) in enumerate(itertools.product(range(self.n_pure_relations), repeat=2)):
            if next_idx != -1 and (next_idx == k1 or next_idx == k2):
                cur_obs[self.n_pure_relations + k] = np.dot(cur_obs[k1], cur_obs[k2])
                mask[self.n_pure_relations + k][cur_obs[self.n_pure_relations + k] != 0] = 1
            elif next_idx == -1:
                cur_obs[self.n_pure_relations + k] = np.dot(cur_obs[k1], cur_obs[k2])
                mask[self.n_pure_relations + k][cur_obs[self.n_pure_relations + k] != 0] = 1

    def compute_particle_weight(self, next_idx, X, mask):
        from scipy.stats import norm
        r_k, e_i, e_j = next_idx

        log_weight = np.zeros(self.n_particles)
        for p in range(self.n_particles):
            mean = np.dot(np.dot(self.E[p][e_i], self.R[p][r_k]), self.E[p][e_j])
            log_weight[p] = norm.logpdf(X[next_idx], mean, self.var_x)

        log_weight -= np.max(log_weight)
        weight = np.exp(log_weight)
        weight += 1e-10
        return weight / np.sum(weight)

    def resample(self):
        count = multinomial(self.n_particles, self.p_weights)

        logger.debug("[RESAMPLE] %s", str(count))

        new_E = np.zeros_like(self.E)
        new_R = np.zeros_like(self.R)

        cnt = 0
        for p in range(self.n_particles):
            for i in range(count[p]):
                new_E[cnt] = self.E[p]
                new_R[cnt] = self.R[p]
                cnt += 1

        self.E = new_E
        self.R = new_R
        self.p_weights = np.ones(self.n_particles) / self.n_particles

    def get_next_sample(self, mask):
        p = multinomial(1, self.p_weights).argmax()
        _X = self._reconstruct(self.E[p], self.R[p], False)
        _X[mask[:self.n_pure_relations] == 1] = MIN_VAL
        return np.unravel_index(_X.argmax(), _X.shape)

    def _sample_prior(self):
        self._sample_var_r()
        self._sample_var_e()

    def _sample_var_r(self):
        for p in range(self.n_particles):
            self.var_r[p] = 1. / gamma(0.5 * self.n_relations * self.n_dim * self.n_dim + self.r_alpha,
                                       1. / (0.5 * np.sum(self.R[p] ** 2) + self.r_beta))
        logger.debug("Sampled var_r %.3f", np.mean(self.var_r))

    def _sample_var_e(self):
        for p in range(self.n_particles):
            self.var_e[p] = 1. / gamma(0.5 * self.n_entities * self.n_dim + self.e_alpha,
                                       1. / (0.5 * np.sum(self.E[p] ** 2) + self.e_beta))
        logger.debug("Sampled var_e %.3f", np.mean(self.var_e))

    def _sample_entities(self, X, mask, E, R, var_e):
        _R = np.zeros([self.n_relations, self.n_dim, self.n_dim])
        _R[:self.n_pure_relations] = R
        for k, (k1, k2) in enumerate(itertools.product(range(self.n_pure_relations), repeat=2)):
            _R[self.n_pure_relations + k] = np.dot(R[k1], R[k2])

        RE = np.zeros([self.n_relations, self.n_entities, self.n_dim])
        RTE = np.zeros([self.n_relations, self.n_entities, self.n_dim])
        for k in range(self.n_relations):
            RE[k] = np.dot(_R[k], E.T).T
            RTE[k] = np.dot(_R[k].T, E.T).T

        for i in range(self.n_entities):
            self._sample_entity(X, mask, E, i, var_e, RE, RTE)
            for k in range(self.n_relations):
                RE[k][i] = np.dot(_R[k], E[i])
                RTE[k][i] = np.dot(_R[k].T, E[i])

    def _sample_entity(self, X, mask, E, i, var_e, RE=None, RTE=None):
        _lambda = np.identity(self.n_dim) / var_e
        xi = np.zeros(self.n_dim)

        for k in self.obs_sum.nonzero()[0]:
            RE[k][i] *= 0
            RTE[k][i] *= 0
            tmp = RE[k][mask[k, i, :] == 1]  # ExD
            tmp2 = RTE[k][mask[k, :, i] == 1]
            if tmp.shape[0] != 0:
                if k < self.n_pure_relations:
                    xi += np.sum(X[k, i, mask[k, i, :] == 1] * tmp.T, 1) / self.var_x
                    _lambda += np.dot(tmp.T, tmp) / self.var_x
                else:
                    xi += np.sum(X[k, i, mask[k, i, :] == 1] * tmp.T, 1) / self.var_comp
                    _lambda += np.dot(tmp.T, tmp) / self.var_comp
            if tmp2.shape[0] != 0:
                if k < self.n_pure_relations:
                    xi += np.sum(X[k, mask[k, :, i] == 1, i] * tmp2.T, 1) / self.var_x
                    _lambda += np.dot(tmp2.T, tmp2) / self.var_x
                else:
                    xi += np.sum(X[k, mask[k, :, i] == 1, i] * tmp2.T, 1) / self.var_comp
                    _lambda += np.dot(tmp2.T, tmp2) / self.var_comp

        inv_lambda = np.linalg.inv(_lambda)
        mu = np.dot(inv_lambda, xi)

        try:
            E[i] = multivariate_normal(mu, inv_lambda)
        except:
            logger.debug('Sample E error', i)

    def _sample_relations(self, X, mask, E, R, var_r):
        EXE = np.kron(E, E)

        for k in range(self.n_pure_relations):
            if self.obs_sum[k] != 0:
                if self.compositionality == ADDITIVE:
                    self._sample_additive_relation(X, mask, R, k, EXE, var_r)
                elif self.compositionality == MULTIPLICATIVE:
                    self._sample_multiplicative_relation(X, mask, R, E, k, EXE, var_r)
            else:
                R[k] = np.random.normal(0, var_r, size=[self.n_dim, self.n_dim])

    def _sample_multiplicative_relation(self, X, mask, R, E, k, EXE, var_r):
        _lambda = np.identity(self.n_dim ** 2) / var_r
        xi = np.zeros(self.n_dim ** 2)

        kron = EXE[mask[k].flatten() == 1]
        if kron.shape[0] != 0:
            _lambda += np.dot(kron.T, kron) / self.var_x
            xi += np.sum(X[k, mask[k] == 1].flatten() * kron.T, 1) / self.var_x

        for _k, (k1, k2) in enumerate(itertools.product(range(self.n_pure_relations), repeat=2)):
            cur_idx = self.n_pure_relations + _k
            if self.obs_sum[cur_idx] != 0:
                if k1 == k:
                    exre = np.kron(E, np.dot(R[k2], E.T).T)
                    kron2 = exre[mask[cur_idx].T.flatten() == 1]
                    _lambda += np.dot(kron2.T, kron2) / self.var_comp
                    xi += np.sum(X[cur_idx, mask[cur_idx] == 1].flatten() * kron2.T, 1) / self.var_comp
                elif k2 == k:
                    rexe = np.kron(np.dot(E, R[k1]), E)
                    kron2 = rexe[mask[cur_idx].T.flatten() == 1]
                    _lambda += np.dot(kron2.T, kron2) / self.var_comp
                    xi += np.sum(X[cur_idx, mask[cur_idx] == 1].flatten() * kron2.T, 1) / self.var_comp

        inv_lambda = np.linalg.inv(_lambda)
        mu = np.dot(inv_lambda, xi)

        try:
            R[k] = multivariate_normal(mu, inv_lambda).reshape([self.n_dim, self.n_dim])
        except:
            logger.debug('Sample R error', k)


    def _sample_additive_relation(self, X, mask, R, k, EXE, var_r):
        _lambda = np.identity(self.n_dim ** 2) / var_r
        xi = np.zeros(self.n_dim ** 2)

        kron = EXE[mask[k].T.flatten() == 1]
        if kron.shape[0] != 0:
            _lambda += np.dot(kron.T, kron) / self.var_x
            xi += np.sum(X[k, mask[k] == 1].flatten() * kron.T, 1) / self.var_x

        tmp = np.zeros(self.n_dim ** 2)
        for _k, (k1, k2) in enumerate(itertools.product(range(self.n_pure_relations), repeat=2)):
            cur_idx = self.n_pure_relations + _k
            if self.obs_sum[cur_idx] != 0:
                kron = EXE[mask[cur_idx].T.flatten() == 1]
                if k1 == k:
                    _lambda += np.dot(kron.T, kron) / (self.var_comp * 4.)
                    tmp += np.sum(X[cur_idx, mask[cur_idx] == 1].flatten() * kron.T, 1)
                    tmp -= 0.5 * np.sum(np.dot(kron, R[k2].flatten()) * kron.T, 1)
                elif k2 == k:
                    _lambda += np.dot(kron.T, kron) / (self.var_comp * 4.)
                    tmp += np.sum(X[cur_idx, mask[cur_idx] == 1].flatten() * kron.T, 1)
                    tmp -= 0.5 * np.sum(np.dot(kron, R[k1].flatten()) * kron.T, 1)

        tmp /= 2. * self.var_comp
        xi += tmp

        inv_lambda = np.linalg.inv(_lambda)
        mu = np.dot(inv_lambda, xi)

        try:
            R[k] = multivariate_normal(mu, inv_lambda).reshape([self.n_dim, self.n_dim])
        except:
            logger.debug('Sample R error', k)

    def _reconstruct(self, E, R, include_comp=False):
        if include_comp:
            _X = np.zeros([self.n_relations, self.n_entities, self.n_entities])
            for k in range(self.n_relations):
                _X[k] = np.dot(np.dot(E, R[k]), E.T)
        else:
            _X = np.zeros([self.n_pure_relations, self.n_entities, self.n_entities])
            for k in range(self.n_pure_relations):
                _X[k] = np.dot(np.dot(E, R[k]), E.T)

        return _X

    def score(self, X, mask):
        from scipy.stats import norm, multivariate_normal, gamma

        if not hasattr(self, 'n_relations'):
            self.n_entities, self.n_relations, _ = X.shape

        score = 0.
        p = self.p_weights.argmax()

        for k in range(self.n_relations):
            mean = np.dot(np.dot(self.E[p], self.R[p][k]), self.E[p].T)
            score += np.sum(norm.logpdf(X[k].flatten(), mean.flatten(), np.sqrt(self.var_x)) * mask[k].flatten())
            score += np.sum(norm.logpdf(self.R[p][k].flatten(), 0, np.sqrt(self.var_r[p])))

        for i in range(self.n_entities):
            score += multivariate_normal.logpdf(self.E[p][i], np.zeros(self.n_dim),
                                                np.identity(self.n_dim) * self.var_e[p])

        if self.sample_prior:
            score += gamma.logpdf(self.var_e[p], loc=self.e_alpha, shape=self.e_beta)
            score += gamma.logpdf(self.var_r[p], loc=self.r_alpha, shape=self.r_beta)

        return score

    def _compute_fit(self, X, mask):
        p = self.p_weights.argmax()
        _X = self._reconstruct(self.E[p], self.R[p], False)
        return self.eval_fn(X[mask[:self.n_pure_relations] == 1].flatten(),
                            _X[mask[:self.n_pure_relations] == 1].flatten())

    def _save_model(self, seq):
        import pickle

        with open(self.dest, 'wb') as f:
            pickle.dump([self, seq], f)

import time
import numpy as np
from numpy.random import multivariate_normal, gamma, multinomial
from sklearn.metrics import mean_squared_error, roc_auc_score
from ..utils.formatted_logger import formatted_logger

logger = formatted_logger(__name__)

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
_SGLD = False
_NMINI = 1
_GIBBS_INIT = True
_SAMPLE_ALL = True

_VAR_E = 1.
_VAR_R = 1.
_VAR_X = 0.01

_DEST = ''
_LOG = ''

a = 0.001
b = 0.01
tau = -0.55

MIN_VAL = np.iinfo(np.int32).min


def inverse(M):
    """Inverse of a symmetric matrix.
    Using cProfile, this seems slower than numpy.linalg.inv
    """
    w, v = np.linalg.eigh(M)
    inv_w = np.diag(1. / w)
    inv = np.dot(v.T, np.dot(inv_w, v))
    return inv


def rademacher(n):
    """Returns a Rademacher random variable of length n"""
    r = np.random.rand(n)
    r = np.round(r)
    r[r == 0.] = -1.
    return r


def normal(mean, prec):
    """Multivariate normal with precision (inverse covariance)
    as a parameter.

    Returns a single sample.

    Copying
    https://github.com/numpy/numpy/blob/master/numpy/random/mtrand/mtrand.pyx
    """
    # Check preconditions on arguments
    mean = np.array(mean)
    prec = np.array(prec)

    if len(mean.shape) != 1:
        raise ValueError("mean must be 1 dimensional")
    if (len(prec.shape) != 2) or (prec.shape[0] != prec.shape[1]):
        raise ValueError("cov must be 2 dimensional and square")
    if mean.shape[0] != prec.shape[0]:
        raise ValueError("mean and cov must have same length")

    n = mean.shape[0]
    x = np.random.randn(n)
    # numpy uses svd, we use eigh
    (s, v) = np.linalg.eigh(prec)
    # s = s[::-1]
    # v = np.flipud((rademacher(mean.shape[0])*v).T)
    # (u, _s, _v) = np.linalg.svd(prec)
    # if not np.allclose(s, _s):
    #    print(s)
    #    print(_s)
    # if not np.allclose(np.abs(v), np.abs(_v)):
    #    print(v)
    #    print(_v)
    x = np.dot(x, np.sqrt(1. / s)[:, None] * (rademacher(n) * v).T)
    x += mean
    return x


def compute_regret(T, seq):
    mask = np.ones_like(T)
    regret = list()
    for s in seq:
        best = np.max(T[mask == 1])
        regret.append(best - T[s])
        mask[s] = 0
    return regret


def load_and_run(path, T, max_iter):
    import pickle
    import os
    with open(path, 'rb') as f:
        model, seq = pickle.load(f)
        log = os.path.splitext(path)[0] + ".txt"
        obs_mask = np.zeros_like(T)
        for s in seq:
            obs_mask[s] = 1

        if os.path.exists(log):
            model.log = log
            for idx in model.particle_filter(T, obs_mask, max_iter):
                with open(model.log, 'a') as f:
                    f.write('%d,%d,%d\n' % (idx[0], idx[1], idx[2]))
                seq.append(idx)
        else:
            _seq = [idx for idx in model.particle_filter(T, obs_mask, max_iter)]
            for s in _seq:
                seq.append(s)

        if len(model.dest) > 0:
            model._save_model(seq)

        return seq


class PFBayesianRescal:
    def __init__(self, n_dim, compute_score=True, sample_prior=False, rbp=False,
                 obs_var=.01, unobs_var=10., n_particles=5, selection='Thompson',
                 eval_fn=roc_auc_score, **kwargs):
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

        rbp: boolean, default=False
            If True, Rao-Blackwellized particle sampling algorithm will be
            applied instead basic particle sampling.

        obs_var: float, default=0.01
            The variance of the observed triple.

        unobs_var: float, default=10.0
            The variance of the unobserved triple.

        n_particles: int, default=5
            Number of particles for sequential Monte Carlo sampler

        selection: string {'Thompson', 'random'}
            Query selection method.

        eval_fn: callable
            Evaluate scoring function between training data and reconstructed data


        Returns
        -------

        """
        self.n_dim = n_dim

        self._var_e = kwargs.pop('var_e', _VAR_E)
        self._var_r = kwargs.pop('var_r', _VAR_R)
        self.var_x = kwargs.pop('var_x', _VAR_X)

        self.rbp = rbp
        self.gibbs_init = kwargs.pop('gibbs_init', _GIBBS_INIT)
        self.sample_all = kwargs.pop('sample_all', _SAMPLE_ALL)
        self.compute_score = compute_score
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
        self.eval_log = kwargs.pop('eval_log', _LOG)

        if not len(kwargs) == 0:
            raise ValueError('Unknown keywords (%s)' % (kwargs.keys()))

        self.n_particles = n_particles
        self.p_weights = np.ones(n_particles) / n_particles
        self.selection = selection
        self.eval_fn = eval_fn

        self.obs_var = obs_var
        self.unobs_var = unobs_var

        self.var_e = np.ones(self.n_particles) * self._var_e
        self.var_r = np.ones(self.n_particles) * self._var_r

    def __getstate__(self):
        """
        Remove attributes that are used to speed up the computations
        """
        d = dict(self.__dict__)
        del d['features']
        del d['xi']
        del d['RE']
        del d['RTE']
        return d

    def fit(self, X, obs_mask=None, max_iter=100, test_mask=None, givenR=None):
        """
        Running the particle Thompson sampling with predefined parameters.

        Parameters
        ----------
        X : numpy.ndarray
            Fully observed tensor with shape (n_relations, n_entities, n_entities)
        obs_mask : numpy.ndarray, default=None
            Mask tensor of observed triples
        max_iter : int, default=100
            Maximum number of iterations for particle Thompson sampling
        Returns
        -------
        seq : numpy.ndarray
            Returns a sequence of selected triples over iterations.
        """
        self.n_relations = X.shape[0]
        self.n_entities = X.shape[1]
        self.E = list()
        self.R = list()
        self.RE = np.zeros([self.n_relations, self.n_entities, self.n_dim])
        self.RTE = np.zeros([self.n_relations, self.n_entities, self.n_dim])

        if isinstance(obs_mask, type(None)):
            obs_mask = np.zeros_like(X)
        else:
            logger.info("Initial Total, Positive, Negative Observation: %d / %d / %d", np.sum(obs_mask),
                        np.sum(X[obs_mask == 1]), np.sum(obs_mask) - np.sum(X[obs_mask == 1]))

        cur_obs = np.zeros_like(X)
        for k in range(self.n_relations):
            cur_obs[k][obs_mask[k] == 1] = X[k][obs_mask[k] == 1]

        self.obs_sum = np.sum(np.sum(obs_mask, 1), 1)
        self.valid_relations = np.nonzero(np.sum(np.sum(X, 1), 1))[0]

        self.features = np.zeros([2 * self.n_entities * self.n_relations, self.n_dim])
        self.xi = np.zeros([2 * self.n_entities * self.n_relations])

        # cur_obs[cur_obs.nonzero()] = 1
        if self.gibbs_init and np.sum(self.obs_sum) != 0:
            # initialize latent variables with gibbs sampling
            E = np.random.random([self.n_entities, self.n_dim])
            R = np.random.random([self.n_relations, self.n_dim, self.n_dim])

            for gi in range(20):
                tic = time.time()
                if isinstance(givenR, type(None)):
                    self._sample_relations(cur_obs, obs_mask, E, R, self._var_r)
                else:
                    self._sample_entities(cur_obs, obs_mask, E, R, self._var_e)
                logger.info("Gibbs Init %d: %f", gi, time.time() - tic)

            for p in range(self.n_particles):
                self.E.append(E.copy())
                self.R.append(R.copy())
        else:
            # random initialization
            for p in range(self.n_particles):
                self.E.append(np.random.random([self.n_entities, self.n_dim]))
                self.R.append(np.random.random([self.n_relations, self.n_dim, self.n_dim]))

        if len(self.log) > 0:
            seq = list()
            for idx in self.particle_filter(X, cur_obs, obs_mask, max_iter, test_mask):
                with open(self.log, 'a') as f:
                    f.write('%d,%d,%d\n' % (idx[0], idx[1], idx[2]))
                seq.append(idx)
                if len(self.eval_log) > 0:
                    p = multinomial(1, self.p_weights).argmax()
                    test_error = self.eval_fn(X[test_mask == 1],
                                              self._reconstruct(self.E[p], self.R[p])[test_mask == 1])
                    with open(self.eval_log, 'a') as f:
                        f.write('%f\n' % (test_error))
                    logger.info('[TEST_ERROR] %.2f', test_error)
        else:
            seq = [idx for idx in self.particle_filter(X, cur_obs, obs_mask, max_iter)]

        if len(self.dest) > 0:
            self._save_model(seq)

        return seq

    def particle_filter(self, X, cur_obs, mask, max_iter, test_mask=None):
        """
        Running a particle Thompson sampling

        Parameters
        ----------
        X : numpy.ndarray
            Fully observed tensor with shape (n_relations, n_entities, n_entities)
        cur_obs : numpy.ndarray
            Initial observation of tensor `X`
        mask : numpy.ndarray, default=None
            Mask tensor of observed triples
        max_iter: int
            maximum number of particle Thompson sampling

        test_mask

        Returns
        -------

        """

        pop = 0
        for i in range(max_iter):
            tic = time.time()

            next_idx = self.get_next_sample(mask, test_mask)
            yield next_idx
            cur_obs[next_idx] = X[next_idx]
            mask[next_idx] = 1
            if X[next_idx] == self.pos_val:
                pop += 1

            # cur_obs[cur_obs.nonzero()] = 1

            logger.info('[NEXT] %s: %.3f, population: %d/%d', str(next_idx), X[next_idx], pop,
                        (i + 1))

            self.p_weights *= self.compute_particle_weight(next_idx, cur_obs, mask)
            self.p_weights /= np.sum(self.p_weights)

            # cur_obs[cur_obs.nonzero()] = 1
            self.obs_sum = np.sum(np.sum(mask, 1), 1)

            ESS = 1. / np.sum((self.p_weights ** 2))

            if ESS < self.n_particles / 2.:
                self.resample()

            for m in range(self.mc_move):
                for p in range(self.n_particles):
                    if self.sample_all:
                        self._sample_relations(cur_obs, mask, self.E[p], self.R[p], self.var_r[p])
                        self._sample_entities(cur_obs, mask, self.E[p], self.R[p], self.var_e[p])
                    else:
                        self._sample_relations(cur_obs, mask, self.E[p], self.R[p], self.var_r[p])
                        self._sample_entities(cur_obs, mask, self.E[p], self.R[p], self.var_e[p],
                                              [next_idx[0], next_idx[1]])

                    if self.rbp:
                        self._sample_relations(cur_obs, mask, self.E[p], self.R[p], self.var_r[p])

            if self.sample_prior and i != 0 and i % self.prior_sample_gap == 0:
                self._sample_prior()

            toc = time.time()
            if self.compute_score:
                # compute training log-likelihood and error on observed data points
                _score = self.score(cur_obs, mask)
                _fit = self._compute_fit(cur_obs, mask)
                logger.info("[%3d] LL: %.3f | fit(%s): %0.5f |  sec: %.3f", i, _score, self.eval_fn.__name__, _fit,
                            (toc - tic))
            else:
                logger.info("[%3d] sec: %.3f", i, (toc - tic))

    def compute_particle_weight(self, next_idx, X, mask):
        from scipy.stats import norm
        r_k, e_i, e_j = next_idx

        log_weight = np.zeros(self.n_particles)
        for p in range(self.n_particles):
            if self.rbp:
                EXE = np.kron(self.E[p], self.E[p])
                _lambda = np.identity(self.n_dim ** 2) / self.var_r[p]
                xi = np.zeros(self.n_dim ** 2)

                kron = EXE[mask[r_k].flatten() == 1]
                if kron.shape[0] != 0:
                    _lambda += np.dot(kron.T, kron)
                    xi += np.sum(X[r_k, mask[r_k] == 1].flatten()[:, np.newaxis] * kron, 0)

                _lambda /= self.var_x
                mu = np.linalg.solve(_lambda, xi) / self.var_x
                ###################
                # EXE = np.kron(self.E[p], self.E[p])
                # _lambda = np.dot(EXE.T, EXE)  # D^2 x D^2
                # _lambda /= self.var_x
                # _lambda += (1. / self.var_r[p]) * np.identity(self.n_dim ** 2)
                # inv_lambda = np.linalg.inv(_lambda)
                # xi = np.sum(EXE * X[r_k].flatten()[:, np.newaxis], 0)
                # mu = (1. / self.var_x) * np.dot(inv_lambda, xi)
                ###################

                exe = np.kron(self.E[p][e_i], self.E[p][e_j])
                log_weight[p] = norm.logpdf(X[next_idx], np.dot(exe, mu), np.dot(np.dot(exe.T, _lambda), exe))
            else:
                mean = np.dot(np.dot(self.E[p][e_i], self.R[p][r_k]), self.E[p][e_j])
                log_weight[p] = norm.logpdf(X[next_idx], mean, self.var_x)

        log_weight -= np.max(log_weight)
        weight = np.exp(log_weight)
        weight += 1e-10
        return weight / np.sum(weight)

    def resample(self):
        count = multinomial(self.n_particles, self.p_weights)

        logger.debug("[RESAMPLE] %s", str(count))

        new_E = list()
        new_R = list()

        for p in range(self.n_particles):
            for i in range(count[p]):
                new_E.append(self.E[p].copy())
                new_R.append(self.R[p].copy())

        self.E = new_E
        self.R = new_R
        self.p_weights = np.ones(self.n_particles) / self.n_particles

    def get_next_sample(self, mask, test_mask=None):
        if self.selection == 'Thompson':
            p = multinomial(1, self.p_weights).argmax()
            _X = self._reconstruct(self.E[p], self.R[p])
            _X[mask[:self.n_relations] == 1] = MIN_VAL
            if not isinstance(test_mask, type(None)):
                _X[test_mask == 1] = MIN_VAL
            return np.unravel_index(_X.argmax(), _X.shape)

        elif self.selection == 'Random':
            correct = False

            while not correct:
                sample = (np.random.randint(self.n_relations), np.random.randint(self.n_entities),
                          np.random.randint(self.n_entities))
                if mask[sample] == 0:
                    correct = True
            return sample

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

    def _sample_entities(self, X, mask, E, R, var_e, sample_idx=None):
        RE = self.RE
        RTE = self.RTE
        for k in range(self.n_relations):
            RE[k] = np.dot(R[k], E.T).T
            RTE[k] = np.dot(R[k].T, E.T).T

        if isinstance(sample_idx, type(None)):
            sample_idx = range(self.n_entities)

        for i in sample_idx:
            self._sample_entity(X, mask, E, R, i, var_e, RE, RTE)
            for k in range(self.n_relations):
                RE[k][i] = np.dot(R[k], E[i])
                RTE[k][i] = np.dot(R[k].T, E[i])

    def _sample_entity(self, X, mask, E, R, i, var_e, RE=None, RTE=None):
        nz_r = mask[:, i, :].nonzero()
        nz_c = mask[:, :, i].nonzero()
        nnz_r = nz_r[0].size
        nnz_c = nz_c[0].size
        nnz_all = nnz_r + nnz_c

        self.features[:nnz_r] = RE[nz_r]
        self.features[nnz_r:nnz_all] = RTE[nz_c]
        self.xi[:nnz_r] = X[:, i, :][nz_r]
        self.xi[nnz_r:nnz_all] = X[:, :, i][nz_c]
        _xi = self.xi[:nnz_all] * self.features[:nnz_all].T
        xi = np.sum(_xi, 1) / self.var_x

        _lambda = np.identity(self.n_dim) / var_e
        _lambda += np.dot(self.features[:nnz_all].T, self.features[:nnz_all]) / self.var_x

        # mu = np.linalg.solve(_lambda, xi)
        # E[i] = normal(mu, _lambda)

        inv_lambda = np.linalg.inv(_lambda)
        mu = np.dot(inv_lambda, xi)
        E[i] = multivariate_normal(mu, inv_lambda)

        mean_var = np.mean(np.diag(inv_lambda))
        logger.info('Mean variance E, %d, %f', i, mean_var)


    def _sample_relations(self, X, mask, E, R, var_r):
        EXE = np.kron(E, E)

        for k in self.valid_relations:
            if self.obs_sum[k] != 0:
                self._sample_relation(X, mask, E, R, k, EXE, var_r)
            else:
                R[k] = np.random.normal(0, var_r, size=[self.n_dim, self.n_dim])

    def _sample_relation(self, X, mask, E, R, k, EXE, var_r):
        _lambda = np.identity(self.n_dim ** 2) / var_r
        xi = np.zeros(self.n_dim ** 2)

        kron = EXE[mask[k].flatten() == 1]

        if kron.shape[0] != 0:
            _lambda += np.dot(kron.T, kron)
            xi += np.sum(X[k, mask[k] == 1].flatten() * kron.T, 1)

        _lambda /= self.var_x
        # mu = np.linalg.solve(_lambda, xi) / self.var_x

        inv_lambda = np.linalg.inv(_lambda)
        mu = np.dot(inv_lambda, xi) / self.var_x
        try:
            # R[k] = normal(mu, _lambda).reshape([self.n_dim, self.n_dim])
            R[k] = multivariate_normal(mu, inv_lambda).reshape([self.n_dim, self.n_dim])
            mean_var = np.mean(np.diag(inv_lambda))
            logger.info('Mean variance R, %d, %f', k, mean_var)
        except:
            pass

    def _reconstruct(self, E, R):
        _X = np.zeros([self.n_relations, self.n_entities, self.n_entities])

        for k in range(self.n_relations):
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
        _X = self._reconstruct(self.E[p], self.R[p])
        return self.eval_fn(X[mask == 1].flatten(), _X[mask == 1].flatten())

    def _save_model(self, seq):
        import pickle

        with open(self.dest, 'wb') as f:
            pickle.dump([self, seq], f)

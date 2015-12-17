import itertools
import logging
import time
import numpy as np
import concurrent.futures
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
_PARALLEL = True
_MAX_THREAD = 4
_POS_VAL = 1
_MC_MOVE = 1

_VAR_E = 1.
_VAR_R = 1.
_VAR_X = 0.01

class PFBayesianRescal:
    def __init__(self, n_dim, compute_score=True, sample_prior=False, rbp=False,
                 controlled_var=False, obs_var=.01, unobs_var=10., n_particles=30, selection='Thompson',
                 eval_fn=mean_squared_error, **kwargs):
        self.n_dim = n_dim

        self._var_e = kwargs.pop('var_e', _VAR_E)
        self._var_r = kwargs.pop('var_r', _VAR_R)
        self.var_x = kwargs.pop('var_x', _VAR_X)

        self.rbp = rbp

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

        self.controlled_var = controlled_var
        self.obs_var = obs_var
        self.unobs_var = unobs_var

        self.n_particles = n_particles
        self.p_weights = np.ones(n_particles)/n_particles
        self.selection = selection
        self.eval_fn = eval_fn

    def fit(self, X, obs_mask=None, max_iter=None):
        self.n_relations = X.shape[0]
        self.n_entities = X.shape[1]

        self.E = list()
        self.R = list()

        self.var_e = np.ones(self.n_particles) * self._var_e
        self.var_r = np.ones(self.n_particles) * self._var_r

        for p in range(self.n_particles):
            self.E.append(np.random.random([self.n_entities, self.n_dim]))
            self.R.append(np.random.random([self.n_relations, self.n_dim, self.n_dim]))

        if type(obs_mask) == type(None):
            obs_mask = np.zeros_like(X)

        if max_iter == None:
            max_iter = np.prod([self.n_relations, self.n_entities, self.n_entities]) - np.sum(obs_mask)

        # for controlled variance
        if self.controlled_var:
            self.var_X = np.ones_like(X) * self.unobs_var
            self.var_X[obs_mask == 1] = self.obs_var

        seq = self.particle_filter(X, obs_mask, max_iter)

        return seq

    def particle_filter(self, X, mask, max_iter):
        cur_obs = np.zeros_like(X)
        cur_obs[mask==1] = X[mask==1]

        seq = list()

        pop = 0

        for i in range(max_iter):
            tic = time.time()

            next_idx = self.get_next_sample(mask, cur_obs)
            cur_obs[next_idx] = X[next_idx]
            mask[next_idx] = 1

            if X[next_idx] == self.pos_val:
                pop += 1

            if self.controlled_var:
                self.var_X[next_idx] = self.obs_var

            if self.sample_prior and i!=0 and i%self.prior_sample_gap==0:
                self._sample_prior()

            seq.append(next_idx)

            logger.info('[NEXT] %s: %.3f, population: %d/%d', str(next_idx), X[next_idx], pop, (i+1))

            self.p_weights *= self.compute_particle_weight(next_idx, cur_obs)
            self.p_weights /= np.sum(self.p_weights)

            ESS = 1./np.sum((self.p_weights**2))

            if ESS < self.n_particles/2.:
                self.resample()

            for p in range(self.n_particles):
                # self._sample_entity(cur_obs,self.E[p],self.R[p],next_idx[1], self.var_e[p])
                # self._sample_entity(cur_obs,self.E[p],self.R[p],next_idx[2], self.var_e[p])
                for m in range(self.mc_move):
                    self._sample_entities(cur_obs, self.E[p], self.R[p], self.var_e[p])
                    self._sample_relations(cur_obs, self.E[p], self.R[p], self.var_r[p])

            toc = time.time()
            if self.compute_score:
                #compute training log-likelihood and erorr
                _score = self.score(cur_obs)
                _fit = self._compute_fit(cur_obs)
                logger.info("[%3d] LL: %.3f | fit(%s): %0.5f |  sec: %.3f", i, _score, self.eval_fn.__name__,  _fit, (toc - tic))
            else:
                logger.info("[%3d] sec: %.3f", i, (toc - tic))

        return seq


    def compute_particle_weight(self, next_idx, X):
        from scipy.stats import norm
        r_k, e_i, e_j = next_idx

        log_weight = np.zeros(self.n_particles)
        for p in range(self.n_particles):
            if self.rbp:
                if not self.controlled_var:
                    EXE = np.kron(self.E[p], self.E[p])
                    _lambda = np.dot(EXE.T, EXE)  # D^2 x D^2
                    _lambda *= (1. / self.var_x)
                    _lambda += (1. / self.var_r[p]) * np.identity(self.n_dim ** 2)
                    inv_lambda = np.linalg.inv(_lambda)
                    xi = np.sum(EXE * X[r_k].flatten()[:, np.newaxis], 0)
                    mu = (1. / self.var_x) * np.dot(inv_lambda, xi)
                    exe = np.kron(self.E[p][e_i], self.E[p][e_j])
                    log_weight[p] = norm.logpdf(X[next_idx], np.dot(exe , mu), np.dot(np.dot(exe.T, _lambda), exe))
                else:
                    raise Exception('Rao-blackwellized PF for controlled var is implemented!')

            else :
                mean = np.dot(np.dot(self.E[p][e_i], self.R[p][r_k]), self.E[p][e_j])
                if self.controlled_var:
                    log_weight[p] = norm.logpdf(X[next_idx], mean, self.unobs_var)
                else:
                    log_weight[p] = norm.logpdf(X[next_idx], mean, self.var_x)
                # score = 0
                # for k in range(self.n_relations):
                #     mean = np.dot(np.dot(self.E[p], self.R[p][k]), self.E[p].T)
                #     if self.controlled_var:
                #         score += np.sum(norm.logpdf(X[k].flatten(), mean.flatten(), np.sqrt(self.var_X[k].flatten())))
                #     else:
                #         score += np.sum(norm.logpdf(X[k].flatten(), mean.flatten(), np.sqrt(self.var_x)))
                # log_weight[p] = score


        log_weight -= np.max(log_weight)
        weight = np.exp(log_weight)
        weight += 1e-10
        return weight/np.sum(weight)

    def resample(self):
        count = multinomial(self.n_particles, self.p_weights)

        logger.debug("[RESAMPLE] %s: %s", self.p_weights, str(count))

        new_E = list()
        new_R = list()

        for p in range(self.n_particles):
            for i in range(count[p]):
                new_E.append(self.E[p].copy())
                new_R.append(self.R[p].copy())

        self.E = new_E
        self.R = new_R
        self.p_weights = np.ones(self.n_particles)/self.n_particles

    def get_next_sample(self, mask, X=None):
        if self.selection == 'Thompson':
            p = multinomial(1, self.p_weights).argmax()
            _X = self._reconstruct(self.E[p], self.R[p], self.var_r[p], X)
            _X *= (1-mask)
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

    def _sample_entity(self, X, E, R, i, var_e):
        E[i] *= 0
        _lambda = np.zeros([self.n_dim, self.n_dim])
        xi = np.zeros(self.n_dim)

        if self.controlled_var:
            for k in range(self.n_relations):
                tmp = np.dot(R[k], E.T)  # D x E
                tmp2 = np.dot(R[k].T, E.T)
                _lambda += np.dot(tmp * (1./self.var_X[k,i,:]), tmp.T) + np.dot(tmp2 * (1./self.var_X[k,:,i]), tmp2.T)

                xi += np.sum((1. / self.var_X[k, i, :]) * X[k, i, :] * tmp, 1) \
                      + np.sum((1. / self.var_X[k, :, i]) * X[k, :, i] * tmp2, 1)

            _lambda += (1. / var_e) * np.identity(self.n_dim)
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
            _lambda += (1. / var_e) * np.identity(self.n_dim)
            inv_lambda = np.linalg.inv(_lambda)
            mu = np.dot(inv_lambda, xi)

        E[i] = multivariate_normal(mu, inv_lambda)
        return E[i]

    def _sample_entities(self, X, E, R, var_e):
        for i in range(self.n_entities):
            self._sample_entity(X, E, R, i, var_e)
        return E

    def _sample_relation(self, X, E, R, k, EXE, var_r, inv_lambda=None):
        if not self.controlled_var:
            if type(inv_lambda) == type(None):
                _lambda = np.dot(EXE.T, EXE)  # D^2 x D^2
                _lambda *= (1. / self.var_x)
                _lambda += (1. / var_r) * np.identity(self.n_dim ** 2)
                inv_lambda = np.linalg.inv(_lambda)

            xi = np.sum(EXE * X[k].flatten()[:, np.newaxis], 0)
            mu = (1. / self.var_x) * np.dot(inv_lambda, xi)
            R[k] = multivariate_normal(mu, inv_lambda).reshape([self.n_dim, self.n_dim])

        else:
            tmp = EXE * (1./self.var_X[k, :, :].flatten()[:,np.newaxis])
            _lambda = np.dot(tmp.T, EXE)
            _lambda += (1. / var_r) * np.identity(self.n_dim ** 2)
            inv_lambda = np.linalg.inv(_lambda)

            xi = np.sum(EXE * X[k].flatten()[:, np.newaxis] * (1. / self.var_X[k, :, :].flatten()[:, np.newaxis]), 0)
            mu = np.dot(inv_lambda, xi)
            R[k] = multivariate_normal(mu, inv_lambda).reshape([self.n_dim, self.n_dim])

        return R[k]

    def _sample_relations(self, X, E, R, var_r):
        EXE = np.kron(E, E)

        if not self.controlled_var:
            _lambda = np.dot(EXE.T, EXE)  # D^2 x D^2
            _lambda *= (1. / self.var_x)
            _lambda += (1. / var_r) * np.identity(self.n_dim ** 2)
            inv_lambda = np.linalg.inv(_lambda)

            if self.parallelize:
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_thread) as executor:
                    fs = [executor.submit(self._sample_relation, X, E, R, k, EXE, var_r, inv_lambda) for k in range(self.n_relations)]
                concurrent.futures.wait(fs)
            else:
                [self._sample_relation(X, E, R, k, EXE, var_r, inv_lambda) for k in range(self.n_relations)]

        else:
            if self.parallelize:
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_thread) as executor:
                    fs = [executor.submit(self._sample_relation, X, E, R, k, EXE, var_r) for k in range(self.n_relations)]
                concurrent.futures.wait(fs)

            else:
                [self._sample_relation(X, E, R, k, EXE, var_r) for k in range(self.n_relations)]

        return R

    def _reconstruct(self, E, R, var_r, X=None):
        _X = np.zeros([self.n_relations, self.n_entities, self.n_entities])

        if self.rbp:
            EXE = np.kron(E, E)
            _lambda = np.dot(EXE.T, EXE)  # D^2 x D^2
            _lambda *= (1. / self.var_x)
            _lambda += (1. / var_r) * np.identity(self.n_dim ** 2)
            inv_lambda = np.linalg.inv(_lambda)
            for k in range(self.n_relations):
                xi = np.sum(EXE * X[k].flatten()[:, np.newaxis], 0)
                mu = (1. / self.var_x) * np.dot(inv_lambda, xi)
                _X[k] = np.dot(EXE, mu).reshape([self.n_entities, self.n_entities])
        else:
            for k in range(self.n_relations):
                _X[k] = np.dot(np.dot(E, R[k]), E.T)

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
        from scipy.stats import norm, multivariate_normal, gamma

        if not hasattr(self, 'n_relations'):
            self.n_entities, self.n_relations, _ = X.shape

        score = 0.
        p = self.p_weights.argmax()

        for k in range(self.n_relations):
            mean = np.dot(np.dot(self.E[p], self.R[p][k]), self.E[p].T)
            if self.controlled_var:
                score += np.sum(norm.logpdf(X[k].flatten(), mean.flatten(), np.sqrt(self.var_X[k].flatten())))
            else:
                score += np.sum(norm.logpdf(X[k].flatten(), mean.flatten(), np.sqrt(self.var_x)))
            score += np.sum(norm.logpdf(self.R[p][k].flatten(), 0, np.sqrt(self.var_r[p])))

        for i in range(self.n_entities):
            score += multivariate_normal.logpdf(self.E[p][i], np.zeros(self.n_dim), np.identity(self.n_dim)*self.var_e[p])

        if self.sample_prior:
            score += gamma.logpdf(self.var_e[p], loc=self.e_alpha, shape=self.e_beta)
            score += gamma.logpdf(self.var_r[p], loc=self.r_alpha, shape=self.r_beta)

        return score/self.n_particles

    def _compute_fit(self, X):
        p = self.p_weights.argmax()
        _X = self._reconstruct(self.E[p], self.R[p], self.var_r[p], X)
        _fit = self.eval_fn(X.flatten(), _X.flatten())

        return _fit

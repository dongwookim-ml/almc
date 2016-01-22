import logging
import time
import itertools
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
_PARALLEL = False
_MAX_THREAD = 4
_POS_VAL = 1
_MC_MOVE = 1
_SGLD = False
_NMINI = 1
_GIBBS_INIT = True
_PULL_SIZE = 1
_COMP = False
_SAMPLE_ALL = True

_VAR_E = 1.
_VAR_R = 1.
_VAR_X = 0.1

_DEST = ''

a = 0.001
b = 0.01
tau = -0.55

MIN_VAL = np.iinfo(np.int32).min


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

        if model.controlled_var:
            model.var_X = np.ones_like(T) * model.unobs_var
            model.var_X[obs_mask == 1] = model.obs_var

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
                 controlled_var=False, obs_var=.01, unobs_var=10., n_particles=5, selection='Thompson',
                 eval_fn=mean_squared_error, log="", **kwargs):
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

        controlled_var: boolean, default=False
            Apply controlled variance approach where we place different
            variances on observed triple and unobserved triple. A variance
            of observed triple is set to be ```obs_var``` and a variance of
            unobserved triple is set to be ```unobs_var```

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

        kwargs

        Returns
        -------

        """
        self.n_dim = n_dim

        self._var_e = kwargs.pop('var_e', _VAR_E)
        self._var_r = kwargs.pop('var_r', _VAR_R)
        self.var_x = kwargs.pop('var_x', _VAR_X)

        self.is_sgld = kwargs.pop('sgld', _SGLD)
        self.n_minibatch = kwargs.pop('n_mini', _NMINI)
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
        self.pull_size = kwargs.pop('pull_size', _PULL_SIZE)
        self.compositional = kwargs.pop('compositional', _COMP)

        if not len(kwargs) == 0:
            raise ValueError('Unknown keywords (%s)' % (kwargs.keys()))

        self.n_particles = n_particles
        self.p_weights = np.ones(n_particles) / n_particles
        self.selection = selection
        self.eval_fn = eval_fn

        self.controlled_var = controlled_var
        self.obs_var = obs_var
        self.unobs_var = unobs_var

        self.var_e = np.ones(self.n_particles) * self._var_e
        self.var_r = np.ones(self.n_particles) * self._var_r

        self.var_x_expanded = 10.

        self.log = log

    def __getstate__(self):
        d = dict(self.__dict__)
        if self.controlled_var:
            del d['var_X']
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
        self.n_relations = X.shape[0]
        self.n_pure_relations = X.shape[0]
        self.n_entities = X.shape[1]
        if self.compositional:
            self.n_relations = X.shape[0] + X.shape[0] ** 2
            self.n_pure_relations = X.shape[0]
            tmp = np.zeros([self.n_relations, self.n_entities, self.n_entities])
            X = self.expand_tensor(X, tmp)

            logger.info('Original size: %d', np.sum(X[:self.n_pure_relations]))
            logger.info('Expanded size %d', np.sum(X[self.n_pure_relations:]))
            logger.info('Expanded shape: %s', X.shape)

        self.E = list()
        self.R = list()

        if type(obs_mask) == type(None):
            obs_mask = np.zeros_like(X)

        cur_obs = np.zeros_like(X)
        for k in range(self.n_pure_relations):
            cur_obs[k][obs_mask[k] == 1] = X[k][obs_mask[k] == 1]

        if self.compositional:
            tmp = np.zeros_like(X)
            tmp[:self.n_pure_relations] == obs_mask
            obs_mask = self.expand_obsmask(tmp, cur_obs)

        self.obs_sum = np.sum(np.sum(obs_mask, 1), 1)
        self.valid_relations = np.nonzero(np.sum(np.sum(X, 1), 1))[0]

        if max_iter == 0:
            max_iter = int(np.prod([self.n_relations, self.n_entities, self.n_entities]) - np.sum(obs_mask))

        # for controlled variance
        if self.controlled_var:
            self.var_X = np.ones_like(X) * self.unobs_var
            self.var_X[obs_mask == 1] = self.obs_var

        cur_obs[cur_obs.nonzero()] = 1
        if self.gibbs_init and np.sum(self.obs_sum) != 0:
            # initialize latent variables with gibbs sampling
            E = np.random.random([self.n_entities, self.n_dim])
            R = np.random.random([self.n_relations, self.n_dim, self.n_dim])

            for gi in range(20):
                tic = time.time()
                self._sample_entities(cur_obs, obs_mask, E, R, self._var_e)
                self._sample_relations(cur_obs, obs_mask, E, R, self._var_r)
                logger.info("Gibbs Init %d: %f", gi, time.time()-tic)

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

            for pn in range(self.pull_size):
                next_idx = self.get_next_sample(mask)
                yield next_idx
                cur_obs[next_idx] = X[next_idx]
                mask[next_idx] = 1
                if self.compositional and cur_obs[next_idx] == self.pos_val:
                    mask = self.expand_obsmask(mask, cur_obs, next_idx[0])
                if X[next_idx] == self.pos_val:
                    pop += 1

                cur_obs[cur_obs.nonzero()] = 1

                if self.controlled_var:
                    self.var_X[next_idx] = self.obs_var

                logger.info('[NEXT] %s: %.3f, population: %d/%d', str(next_idx), X[next_idx], pop,
                            (i * self.pull_size + pn))

                self.p_weights *= self.compute_particle_weight(next_idx, cur_obs, mask)
                self.p_weights /= np.sum(self.p_weights)

            cur_obs[cur_obs.nonzero()] = 1
            self.obs_sum = np.sum(np.sum(mask, 1), 1)

            if self.compositional:
                logger.debug('[Additional Points] %d', np.sum(self.obs_sum[self.n_pure_relations:]))

            ESS = 1. / np.sum((self.p_weights ** 2))

            if ESS < self.n_particles / 2.:
                self.resample()

            if self.is_sgld:
                epsilon = a * (b + i) ** tau

                if self.parallelize:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_thread) as executor:
                        fs = [executor.submit(self.stochastic_gradient, cur_obs, self.E[p], self.R[p], self.n_minibatch,
                                              epsilon, self.var_e[p], self.var_r[p]) for p
                              in range(self.n_particles)]
                    concurrent.futures.wait(fs)
                else:
                    for p in range(self.n_particles):
                        self.stochastic_gradient(cur_obs, self.E[p], self.R[p], self.n_minibatch, epsilon,
                                                 self.var_e[p], self.var_r[p])

            else:
                for m in range(self.mc_move):
                    for p in range(self.n_particles):
                        if self.sample_all:
                            self._sample_relations(cur_obs, mask, self.E[p], self.R[p], self.var_r[p])
                            self._sample_entities(cur_obs, mask, self.E[p], self.R[p], self.var_e[p])
                        else:
                            self._sample_relations(cur_obs, mask, self.E[p], self.R[p], self.var_r[p])

                            RE = list()
                            RTE = list()
                            for k in range(self.n_relations):
                                RE.append(np.dot(self.R[p][k], self.E[p].T).T)  #
                                RTE.append(np.dot(self.R[p][k].T, self.E[p].T).T)

                            for ni in [next_idx[1], next_idx[2]]:
                                self._sample_entity(X, mask, self.E[p], self.R[p], ni, self.var_e[p], RE, RTE)
                                for k in range(self.n_relations):
                                    RE[k][ni] = np.dot(self.R[p][k], self.E[p][ni])
                                    RTE[k][ni] = np.dot(self.R[p][k].T, self.E[p][ni])

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

    def expand_tensor(self, T, T_expanded):
        """

        Parameters
        ----------
        T : numpy.ndarray
            Tensor with size of (n_relations, n_entities, n_entities)
        T_expanded : numpy.ndarray
            Tensor with size of (n_relations+n_relations**2, n_entities, n_entities)

        Returns
        -------
        T_expanded : numpy.ndarray
            returns two-step expanded tensor of T where each entry count the number of path from entity to entity
            with combination of two relations
        """
        T_expanded[:self.n_pure_relations] = T
        for k, (k1, k2) in enumerate(itertools.product(range(self.n_pure_relations), repeat=2)):
            T_expanded[self.n_pure_relations + k] = np.dot(T[k1], T[k2])
        return T_expanded

    def expand_obsmask(self, mask, cur_obs, next_idx=-1):
        for k, (k1, k2) in enumerate(itertools.product(range(self.n_pure_relations), repeat=2)):
            if next_idx != -1 and (next_idx == k1 or next_idx == k2):
                cur_obs[self.n_pure_relations + k] = np.dot(cur_obs[k1], cur_obs[k2])
                mask[self.n_pure_relations + k][cur_obs[self.n_pure_relations + k] != 0] = 1
            elif next_idx == -1:
                cur_obs[self.n_pure_relations + k] = np.dot(cur_obs[k1], cur_obs[k2])
                mask[self.n_pure_relations + k][cur_obs[self.n_pure_relations + k] != 0] = 1
        return mask

    def compute_particle_weight(self, next_idx, X, mask):
        from scipy.stats import norm
        r_k, e_i, e_j = next_idx

        log_weight = np.zeros(self.n_particles)
        for p in range(self.n_particles):
            if self.rbp:
                if not self.controlled_var:
                    EXE = np.kron(self.E[p], self.E[p])
                    _lambda = np.identity(self.n_dim ** 2) / self.var_r[p]
                    xi = np.zeros(self.n_dim ** 2)

                    kron = EXE[mask[r_k].flatten() == 1]
                    if kron.shape[0] != 0:
                        _lambda += np.dot(kron.T, kron)
                        xi += np.sum(X[r_k, mask[r_k] == 1].flatten()[:, np.newaxis] * kron, 0)

                    _lambda /= self.var_x
                    inv_lambda = np.linalg.inv(_lambda)
                    mu = np.dot(inv_lambda, xi) / self.var_x

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
                    raise Exception('Rao-blackwellized PF for controlled var is implemented!')

            else:
                mean = np.dot(np.dot(self.E[p][e_i], self.R[p][r_k]), self.E[p][e_j])
                if self.controlled_var:
                    log_weight[p] = norm.logpdf(X[next_idx], mean, self.unobs_var)
                else:
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

    def get_next_sample(self, mask):
        if self.selection == 'Thompson':
            p = multinomial(1, self.p_weights).argmax()
            _X = self._reconstruct(self.E[p], self.R[p])
            _X[mask[:self.n_pure_relations] == 1] = MIN_VAL
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

    def _sample_entities(self, X, mask, E, R, var_e):
        if self.controlled_var:
            for i in range(self.n_entities):
                self._sample_entity(X, mask, E, R, i, var_e)
        else:
            RE = list()
            RTE = list()
            for k in range(self.n_relations):
                RE.append(np.dot(R[k], E.T).T)  #
                RTE.append(np.dot(R[k].T, E.T).T)

            for i in range(self.n_entities):
                self._sample_entity(X, mask, E, R, i, var_e, RE, RTE)
                for k in range(self.n_relations):
                    RE[k][i] = np.dot(R[k], E[i])
                    RTE[k][i] = np.dot(R[k].T, E[i])

    def _sample_entity(self, X, mask, E, R, i, var_e, RE=None, RTE=None):
        _lambda = np.identity(self.n_dim) / var_e
        xi = np.zeros(self.n_dim)

        E[i] *= 0

        if self.controlled_var:
            for k in range(self.n_relations):
                tmp = np.dot(R[k], E.T)  # D x E
                tmp2 = np.dot(R[k].T, E.T)
                _lambda += np.dot(tmp * (1. / self.var_X[k, i, :]), tmp.T)
                _lambda += np.dot(tmp2 * (1. / self.var_X[k, :, i]), tmp2.T)

                xi += np.sum((1. / self.var_X[k, i, :]) * X[k, i, :] * tmp, 1)
                xi += np.sum((1. / self.var_X[k, :, i]) * X[k, :, i] * tmp2, 1)

            inv_lambda = np.linalg.inv(_lambda)
            mu = np.dot(inv_lambda, xi)

        else:
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
                        xi += np.sum(X[k, i, mask[k, i, :] == 1] * tmp.T, 1) / self.var_x_expanded
                        _lambda += np.dot(tmp.T, tmp) / self.var_x_expanded
                if tmp2.shape[0] != 0:
                    if k < self.n_pure_relations:
                        xi += np.sum(X[k, mask[k, :, i] == 1, i] * tmp2.T, 1) / self.var_x
                        _lambda += np.dot(tmp2.T, tmp2) / self.var_x
                    else:
                        xi += np.sum(X[k, mask[k, :, i] == 1, i] * tmp2.T, 1) / self.var_x_expanded
                        _lambda += np.dot(tmp2.T, tmp2) / self.var_x_expanded

            # xi /= self.var_x
            # _lambda /= self.var_x

            inv_lambda = np.linalg.inv(_lambda)
            mu = np.dot(inv_lambda, xi)

            ################### sanity check
            # _lambda = np.zeros([self.n_dim, self.n_dim])
            # xi = np.zeros(self.n_dim)
            # for k in range(self.n_relations):
            #     tmp = np.dot(R[k], E.T)  # D x E
            #     tmp2 = np.dot(R[k].T, E.T)
            #     _lambda += np.dot(tmp, tmp.T) + np.dot(tmp2, tmp2.T)
            #     xi += np.sum(X[k, i, :] * tmp, 1) + np.sum(X[k, :, i] * tmp2, 1)
            #
            # xi *= (1. / self.var_x)
            # _lambda *= 1. / self.var_x
            # _lambda += (1. / self.var_e) * np.identity(self.n_dim)
            # _inv_lambda = np.linalg.inv(_lambda)
            # _mu = np.dot(_inv_lambda, xi)
            #
            # assert np.allclose(_inv_lambda, inv_lambda)
            # assert np.allclose(_mu, mu)

        E[i] = multivariate_normal(mu, inv_lambda)

    def _sample_relations(self, X, mask, E, R, var_r):
        EXE = np.kron(E, E)

        for k in self.valid_relations:
            if self.obs_sum[k] != 0:
                self._sample_relation(X, mask, E, R, k, EXE, var_r)
            else:
                R[k] = np.random.normal(0, var_r, size=[self.n_dim, self.n_dim])

    def _sample_relation(self, X, mask, E, R, k, EXE, var_r):
        if self.controlled_var:
            tmp = EXE / self.var_X[k, :, :].flatten()[:, np.newaxis]
            _lambda = np.dot(tmp.T, EXE)
            _lambda += np.identity(self.n_dim ** 2) / var_r
            inv_lambda = np.linalg.inv(_lambda)

            xi = np.sum(EXE * X[k].flatten()[:, np.newaxis] / self.var_X[k, :, :].flatten()[:, np.newaxis], 0)
            mu = np.dot(inv_lambda, xi)
        else:
            _lambda = np.identity(self.n_dim ** 2) / var_r
            xi = np.zeros(self.n_dim ** 2)

            kron = EXE[mask[k].flatten() == 1]
            if kron.shape[0] != 0:
                _lambda += np.dot(kron.T, kron)
                xi += np.sum(X[k, mask[k] == 1].flatten() * kron.T, 1)

            if k < self.n_pure_relations:
                _lambda /= self.var_x
            else:
                _lambda /= self.var_x_expanded
            inv_lambda = np.linalg.inv(_lambda)
            if k < self.n_pure_relations:
                mu = np.dot(inv_lambda, xi) / self.var_x
            else:
                mu = np.dot(inv_lambda, xi) / self.var_x_expanded

        try:
            R[k] = multivariate_normal(mu, inv_lambda).reshape([self.n_dim, self.n_dim])
        except:
            pass

    def stochastic_gradient(self, X, E, R, n_minibatch, epsilon, var_e, var_r):
        for i in range(self.n_entities):
            candid = np.setdiff1d(range(self.n_entities), i)
            np.random.shuffle(candid)
            random_entities = candid[:n_minibatch]
            for k in range(self.n_relations):
                x_bar = X[k, i, random_entities] - np.dot(np.dot(E[i], R[k]), E[random_entities].T)
                gradE = np.dot(E[random_entities], R[k]) * x_bar[:, np.newaxis] / self.var_x

                x_bar = X[k, random_entities, i] - np.dot(np.dot(E[random_entities], R[k].T), E[i].T)
                gradE += np.dot(E[random_entities], R[k].T) * x_bar[:, np.newaxis] / self.var_x

            gradE = np.sum(gradE, 0)
            gradE *= (self.n_entities - 1) * 2 / (n_minibatch * 2)

            gradE -= E[i] / var_e

            nu = np.random.normal(0, epsilon, size=self.n_dim)

            E[i] += 0.5 * epsilon * gradE + nu

        for k in range(self.n_relations):
            candid = list(range(self.n_entities))
            np.random.shuffle(candid)
            random_entities = candid[:n_minibatch]

            EXE = np.kron(E[random_entities], E[random_entities])
            x_bar = X[k, random_entities, random_entities] - np.dot(np.dot(E[random_entities], R[k]),
                                                                    E[random_entities].T)
            gradR = EXE * x_bar.flatten()[:, np.newaxis]
            gradR = np.sum(gradR, 0)

            gradR *= (self.n_entities ** 2) / (n_minibatch ** 2)
            gradR -= R[k].flatten() / var_r
            nu = np.random.normal(0, epsilon, size=self.n_dim ** 2)

            R[k] += (0.5 * epsilon * gradR + nu).reshape([self.n_dim, self.n_dim])

    def _reconstruct(self, E, R):
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
            if self.controlled_var:
                score += np.sum(norm.logpdf(X[k].flatten(), mean.flatten(), np.sqrt(self.var_X[k].flatten())))
            else:
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

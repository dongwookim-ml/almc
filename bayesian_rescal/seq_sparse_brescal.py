import logging
import time
import itertools
import numpy as np
import concurrent.futures
from numpy.random import multivariate_normal, gamma, multinomial
from sklearn.metrics import mean_squared_error
from scipy.sparse import csr_matrix as sparse_matrix

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
_NMINI = 1
_GIBBS_INIT = True
_PULL_SIZE = 1
_COMP = False

_VAR_E = 1.
_VAR_R = 1.
_VAR_X = 0.1

_DEST = ''

MIN_VAL = np.iinfo(np.int32).min


class PFSparseBayesianRescal:
    def __init__(self, n_dim, compute_score=True, sample_prior=False, rbp=False,
                 obs_var=.01, unobs_var=10., n_particles=5, selection='Thompson',
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

        self.rbp = rbp

        self.gibbs_init = kwargs.pop('gibbs_init', _GIBBS_INIT)

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

        self.var_e = np.ones(self.n_particles) * self._var_e
        self.var_r = np.ones(self.n_particles) * self._var_r

        self.var_x_expanded = 1.0

        self.log = log

    def __getstate__(self):
        d = dict(self.__dict__)
        return d

    def fit(self, X, obs_mask=None, max_iter=0):
        """
        Running the particle Thompson sampling with predefined parameters.

        Parameters
        ----------
        X : scipy.sparse
            List of sparse matrix with shape (n_relations) (n_entities, n_entities)
        obs_mask : numpy.ndarray, default=None
            Mask tensor of observed triples
        max_iter : int, default=0
            Maximum number of iterations for particle Thompson sampling
        Returns
        -------
        seq : numpy.ndarray
            Returns a sequence of selected triples over iterations.
        """

        self.n_relations = len(X)
        self.n_pure_relations = len(X)
        self.n_entities = X[0].shape[0]
        if self.compositional:
            self.n_relations = self.n_pure_relations + self.n_pure_relations ** 2
            self.expand_tensor(X)
            logger.info('Original size: %d', np.sum([X[i].sum() for i in range(self.n_pure_relations)]))
            logger.info('Expanded size %d',
                        np.sum([X[i].sum() for i in range(self.n_pure_relations, self.n_relations)]))

        self.E = list()
        self.R = list()

        if type(obs_mask) == type(None):
            obs_mask = [sparse_matrix((self.n_entities, self.n_entities)) for i in range(self.n_pure_relations)]

        if self.compositional:
            cur_obs = [sparse_matrix((self.n_entities, self.n_entities)) for i in range(self.n_relations)]
            for k in range(self.n_pure_relations):
                cur_obs[k][obs_mask[k].nonzero()] = X[k][obs_mask[k].nonzero()]
            for (k1, k2) in itertools.product(range(self.n_pure_relations), repeat=2):
                obs_mask.append(sparse_matrix((self.n_entities, self.n_entities)))
            self.expand_obsmask(obs_mask, cur_obs)

        self.obs_sum = np.array([obs_mask[i].sum() for i in range(self.n_relations)])

        if max_iter == 0:
            max_iter = int(np.prod([self.n_relations, self.n_entities, self.n_entities]) - np.sum(self.obs_sum))

        for p in range(self.n_particles):
            self.E.append(np.random.normal(0, 1, size=[self.n_entities, self.n_dim]))
            self.R.append(np.random.normal(0, 1, size=[self.n_relations, self.n_dim, self.n_dim]))

        if len(self.log) > 0:
            seq = list()
            for idx in self.particle_filter(X, obs_mask, max_iter):
                with open(self.log, 'a') as f:
                    f.write('%d,%d,%d\n' % (idx[0], idx[1], idx[2]))
                seq.append(idx)
        else:
            seq = [idx for idx in self.particle_filter(X, obs_mask, max_iter)]

        if len(self.dest) > 0:
            self._save_model(seq)

        return seq

    def particle_filter(self, X, mask, max_iter):
        cur_obs = [sparse_matrix((self.n_entities, self.n_entities)) for i in range(self.n_relations)]
        for k in range(self.n_relations):
            cur_obs[k][mask[k].nonzero()] = X[k][mask[k].nonzero()]

        pop = 0
        for i in range(max_iter):
            tic = time.time()

            n_i = np.random.randint(self.n_entities)
            n_k = np.random.randint(self.n_pure_relations)
            n_j = self.get_next_sample(mask, n_i, n_k)
            # n_k, n_i, n_j = next_idx
            next_idx = (n_k, n_i, n_j)
            yield next_idx

            if X[n_k][n_i, n_j] != 0:
                cur_obs[n_k][n_i, n_j] = X[n_k][n_i, n_j]
            mask[n_k][n_i, n_j] = 1
            if self.compositional and cur_obs[n_k][n_i, n_j] == self.pos_val:
                self.expand_obsmask(mask, cur_obs, n_k)
            if X[n_k][n_i, n_j] == self.pos_val:
                pop += 1

            logger.info('[NEXT] %s: %.3f, population: %d/%d', str((n_k, n_i, n_j)), X[n_k][n_i, n_j], pop, i)

            self.p_weights *= self.compute_particle_weight(next_idx, cur_obs, mask)
            self.p_weights /= np.sum(self.p_weights)

            self.obs_sum = np.array([mask[i].sum() for i in range(self.n_relations)])

            if self.compositional:
                logger.debug('[Additional Points] %d', np.sum(self.obs_sum[self.n_pure_relations:]))

            ESS = 1. / np.sum((self.p_weights ** 2))

            if ESS < self.n_particles / 2.:
                self.resample()

            for m in range(self.mc_move):
                for p in range(self.n_particles):
                    self._sample_relations(cur_obs, mask, self.E[p], self.R[p], self.var_r[p])
                    self._sample_entities(cur_obs, mask, self.E[p], self.R[p], self.var_e[p], n_i, n_j)
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

    def expand_tensor(self, T):
        """

        Parameters
        ----------
        T : list of scipy.sparse

        Returns
        -------
        T_expanded : numpy.ndarray
            returns two-step expanded tensor of T where each entry count the number of path from entity to entity
            with combination of two relations
        """

        for k, (k1, k2) in enumerate(itertools.product(range(self.n_pure_relations), repeat=2)):
            T.append(T[k1].dot(T[k2]))
        return T

    def expand_obsmask(self, mask, cur_obs, next_idx=-1):
        for k, (k1, k2) in enumerate(itertools.product(range(self.n_pure_relations), repeat=2)):
            if next_idx != -1 and (next_idx == k1 or next_idx == k2):
                cur_obs[self.n_pure_relations + k] = cur_obs[k1].dot(cur_obs[k2])
                mask[self.n_pure_relations + k][cur_obs[self.n_pure_relations + k] != 0] = 1
            elif next_idx == -1:
                cur_obs[self.n_pure_relations + k] = cur_obs[k1].dot(cur_obs[k2])
                mask[self.n_pure_relations + k][cur_obs[self.n_pure_relations + k] != 0] = 1
        return mask

    def compute_particle_weight(self, next_idx, X, mask):
        from scipy.stats import norm
        r_k, e_i, e_j = next_idx

        log_weight = np.zeros(self.n_particles)
        for p in range(self.n_particles):
            mean = np.dot(np.dot(self.E[p][e_i], self.R[p][r_k]), self.E[p][e_j])
            log_weight[p] = norm.logpdf(X[r_k][e_i, e_j], mean, self.var_x)

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

    def get_next_sample(self, mask, i, k):
        p = multinomial(1, self.p_weights).argmax()
        _X = np.dot(np.dot(self.E[p][i], self.R[p][k]), self.E[p].T)
        nz = mask[k].getrow(i).nonzero()
        if len(nz[0]) > 0:
            print(nz)
            print(len(nz))
            _X[mask[k].getrow(i).nonzero()] = MIN_VAL
        return _X.argmax()

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

    def _sample_entities(self, X, mask, E, R, var_e, n_i, n_j):
        sampled_entities = np.arange(self.n_entities)
        np.random.shuffle(sampled_entities)

        RE = list()
        RTE = list()
        for k in range(self.n_relations):
            RE.append(np.dot(R[k], E.T).T)  #
            RTE.append(np.dot(R[k].T, E.T).T)

        self._sample_entity(X, mask, E, R, n_i, var_e, RE, RTE)
        for k in range(self.n_relations):
            RE[k][n_i] = np.dot(R[k], E[n_i])
            RTE[k][n_i] = np.dot(R[k].T, E[n_i])

        self._sample_entity(X, mask, E, R, n_j, var_e, RE, RTE)
        for k in range(self.n_relations):
            RE[k][n_j] = np.dot(R[k], E[n_j])
            RTE[k][n_j] = np.dot(R[k].T, E[n_j])

        # for i in range(self.n_entities):
        for i in sampled_entities[:100]:
            self._sample_entity(X, mask, E, R, i, var_e, RE, RTE)
            for k in range(self.n_relations):
                RE[k][i] = np.dot(R[k], E[i])
                RTE[k][i] = np.dot(R[k].T, E[i])

    def _sample_entity(self, X, mask, E, R, i, var_e, RE=None, RTE=None):
        _lambda = np.identity(self.n_dim) / var_e
        xi = np.zeros(self.n_dim)

        E[i] *= 0

        for k in self.obs_sum.nonzero()[0]:
            RE[k][i] *= 0
            RTE[k][i] *= 0
            tmp = RE[k][mask[k].getrow(i).nonzero()[1]]  # ExD
            tmp2 = RTE[k][mask[k].getcol(i).nonzero()[0]]
            # tmp = np.dot(R[k], E[mask[k, i, :] == 1].T)  # D x E
            # tmp2 = np.dot(R[k].T, E[mask[k, :, i] == 1].T)
            if tmp.shape[0] != 0:
                if k < self.n_pure_relations:
                    # xi += np.sum(X[k][i, mask[k][i, :] == 1] * tmp.T, 1) / self.var_x
                    xi += np.sum(np.squeeze(np.array(X[k].getrow(i)[mask[k].getrow(i).nonzero()])) * tmp.T,
                                 1) / self.var_x
                    _lambda += np.dot(tmp.T, tmp) / self.var_x
                else:
                    # xi += np.sum(X[k][i, mask[k][i, :] == 1] * tmp.T, 1) / self.var_x_expanded
                    xi += np.sum(np.squeeze(np.array(X[k].getrow(i)[mask[k].getrow(i).nonzero()])) * tmp.T,
                                 1) / self.var_x_expanded
                    _lambda += np.dot(tmp.T, tmp) / self.var_x_expanded
            if tmp2.shape[0] != 0:
                if k < self.n_pure_relations:
                    xi += np.sum(np.squeeze(np.array(X[k].getcol(i)[mask[k].getcol(i).nonzero()])) * tmp2.T,
                                 1) / self.var_x
                    _lambda += np.dot(tmp2.T, tmp2) / self.var_x
                else:
                    xi += np.sum(np.squeeze(np.array(X[k].getcol(i)[mask[k].getcol(i).nonzero()])) * tmp2.T,
                                 1) / self.var_x_expanded
                    _lambda += np.dot(tmp2.T, tmp2) / self.var_x_expanded

        inv_lambda = np.linalg.inv(_lambda)
        mu = np.dot(inv_lambda, xi)

        E[i] = multivariate_normal(mu, inv_lambda)

    def _sample_relations(self, X, mask, E, R, var_r):
        for k in range(self.n_relations):
            if self.obs_sum[k] != 0:
                self._sample_relation(X, mask, E, R, k, var_r)
            else:
                R[k] = np.random.normal(0, var_r, size=[self.n_dim, self.n_dim])

    def _sample_relation(self, X, mask, E, R, k, var_r):

        _lambda = np.identity(self.n_dim ** 2) / var_r
        xi = np.zeros(self.n_dim ** 2)

        nz = mask[k].nonzero()
        anz = np.array(nz)
        # idx = [_idx[0] * self.n_entities + _idx[1] for _idx in nz]
        # kron = EXE[idx]
        kron = np.zeros([anz.shape[1], self.n_dim ** 2])
        for e in range(anz.shape[1]):
            kron[e] = np.kron(E[anz[0, e]], E[anz[1, e]])

        if kron.shape[0] != 0:
            _lambda += np.dot(kron.T, kron)
            _x = np.squeeze(np.array(X[k][nz]))
            xi += np.sum(_x * kron.T, 1)

        if k < self.n_pure_relations:
            _lambda /= self.var_x
        else:
            _lambda /= self.var_x_expanded
        inv_lambda = np.linalg.inv(_lambda)
        if k < self.n_pure_relations:
            mu = np.dot(inv_lambda, xi) / self.var_x
        else:
            mu = np.dot(inv_lambda, xi) / self.var_x_expanded

        R[k] = multivariate_normal(mu, inv_lambda).reshape([self.n_dim, self.n_dim])

    def _reconstruct(self, E, R, k):
        return np.dot(np.dot(E, R[k]), E.T)

    def score(self, X, mask):
        pass

    def _compute_fit(self, X, mask):
        pass

    def _save_model(self, seq):
        import pickle

        with open(self.dest, 'wb') as f:
            pickle.dump([self, seq], f)

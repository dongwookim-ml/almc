import time
import numpy as np
from numpy.random import multivariate_normal, multinomial
from sklearn.linear_model import LogisticRegression
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
_GIBBS_INIT = True
_GIBBS_ITER = 20
_SAMPLE_ALL = True
_DEST = ''
_APPROX_DIAG = True

_VAR_E = 1.
_VAR_R = 1.
_VAR_X = 0.01

a = 0.001
b = 0.01
tau = -0.55

MIN_VAL = np.iinfo(np.int32).min


def _sigmoid(x):
    return 1. / (1 + np.exp(-x))


class PFBayesianLogitRescal:
    def __init__(self, n_dim, compute_score=True, sample_prior=False,
                 n_particles=5, selection='Thompson',
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

        self.compute_score = compute_score

        self.approx_diag = kwargs.pop('approx_diag', _APPROX_DIAG)
        self.e_alpha = kwargs.pop('e_alpha', _E_ALPHA)
        self.e_beta = kwargs.pop('e_beta', _E_BETA)
        self.r_alpha = kwargs.pop('r_alpha', _R_ALPHA)
        self.r_beta = kwargs.pop('r_beta', _R_BETA)

        self.mc_move = kwargs.pop('mc_move', _MC_MOVE)

        self.pos_val = kwargs.pop('pos_val', _POS_VAL)
        self.dest = kwargs.pop('dest', _DEST)

        self.gibbs_init = kwargs.pop('gibbs_init', _GIBBS_INIT)
        self.gibbs_iter = kwargs.pop('gibbs_iter', _GIBBS_ITER)
        self.sample_all = kwargs.pop('sample_all', _SAMPLE_ALL)

        if not len(kwargs) == 0:
            raise ValueError('Unknown keywords (%s)' % (kwargs.keys()))

        self.n_particles = n_particles
        self.p_weights = np.ones(n_particles) / n_particles
        self.selection = selection
        self.eval_fn = eval_fn

        self.var_e = np.ones(self.n_particles) * self._var_e
        self.var_r = np.ones(self.n_particles) * self._var_r

        self.log = log

    def __getstate__(self):
        d = dict(self.__dict__)
        return d

    def fit(self, X, obs_mask=None, max_iter=0):
        self.n_relations = X.shape[0]
        self.n_entities = X.shape[1]

        self.E = np.zeros([self.n_particles, self.n_entities, self.n_dim])
        self.R = np.zeros([self.n_particles, self.n_relations, self.n_dim, self.n_dim])

        self.n_obs_entities = np.zeros(self.n_entities)

        if isinstance(obs_mask, type(None)):
            obs_mask = np.zeros_like(X)
            for i, k in itertools.product(range(self.n_entities), range(self.n_relations)):
                self.n_obs_entities[i] += np.sum(obs_mask[k, i, :]) + np.sum(obs_mask[k, :, i])
        else:
            logger.info("Initial Total, Positive, Negative Observation: %d / %d / %d", np.sum(obs_mask),
                        np.sum(X[obs_mask == 1]), np.sum(obs_mask) - np.sum(X[obs_mask == 1]))

        cur_obs = np.zeros_like(X)
        cur_obs[obs_mask == 1] = X[obs_mask == 1]
        self.obs_sum = np.sum(np.sum(obs_mask, 1), 1)

        self.features = np.zeros([2 * self.n_entities * self.n_relations, self.n_dim])
        self.Y = np.zeros([2 * self.n_entities * self.n_relations])

        if self.gibbs_init and np.sum(self.obs_sum) != 0:
            # initialize latent variables with gibbs sampling
            E = np.random.random([self.n_entities, self.n_dim])
            R = np.random.random([self.n_relations, self.n_dim, self.n_dim])

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
                self.R[p] = np.random.random([self.n_relations, self.n_dim, self.n_dim])

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

            next_idx = self.get_next_sample(mask, cur_obs)
            yield next_idx
            cur_obs[next_idx] = X[next_idx]
            mask[next_idx] = 1

            self.n_obs_entities[next_idx[1]] += 1
            self.n_obs_entities[next_idx[2]] += 1
            self.obs_sum = np.sum(np.sum(mask, 1), 1)

            if X[next_idx] == self.pos_val:
                pop += 1

            logger.info('[NEXT] %s: %.3f, population: %d/%d', str(next_idx), X[next_idx], pop,
                        (i + 1))

            self.p_weights *= self.compute_particle_weight(next_idx, cur_obs, mask)
            self.p_weights /= np.sum(self.p_weights)

            ESS = 1. / np.sum((self.p_weights ** 2))

            if ESS < self.n_particles / 2.:
                self.resample()

            for m in range(self.mc_move):
                for p in range(self.n_particles):
                    self._sample_relations(cur_obs, mask, self.E[p], self.R[p], self.var_r[p])
                    if self.sample_all:
                        self._sample_entities(cur_obs, mask, self.E[p], self.R[p], self.var_e[p])
                    else:
                        self._sample_entities(cur_obs, mask, self.E[p], self.R[p], self.var_e[p],
                                              [next_idx[1], next_idx[2]])

            toc = time.time()
            if self.compute_score:
                # compute training log-likelihood and error on observed data points
                _score = self.score(X, mask)
                # _fit = self._compute_fit(X, mask)
                _fit = self._compute_fit(X, np.ones_like(X))  # compute fit with respect to the whole dataset
                logger.info("[%3d] LL: %.3f | fit(%s): %0.5f |  sec: %.3f", i, _score, self.eval_fn.__name__, _fit,
                            (toc - tic))
            else:
                logger.info("[%3d] sec: %.3f", i, (toc - tic))

    def compute_particle_weight(self, next_idx, X, mask):
        r_k, e_i, e_j = next_idx

        weight = np.zeros(self.n_particles)
        for p in range(self.n_particles):
            mean = np.dot(np.dot(self.E[p][e_i], self.R[p][r_k]), self.E[p][e_j])
            weight[p] = _sigmoid(mean)

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

    def get_next_sample(self, mask, X=None):
        if self.selection == 'Thompson':
            p = multinomial(1, self.p_weights).argmax()
            _X = self._reconstruct(self.E[p], self.R[p])
            _X[mask == 1] = MIN_VAL
            return np.unravel_index(_X.argmax(), _X.shape)

        elif self.selection == 'Random':
            correct = False

            while not correct:
                sample = (np.random.randint(self.n_relations), np.random.randint(self.n_entities),
                          np.random.randint(self.n_entities))
                if mask[sample] == 0:
                    correct = True
            return sample

    def _sample_entities(self, X, mask, E, R, var_e, sample_idx=None):
        RE = np.zeros([self.n_relations, self.n_entities, self.n_dim])
        RTE = np.zeros([self.n_relations, self.n_entities, self.n_dim])
        for k in range(self.n_relations):
            RE[k] = np.dot(R[k], E.T).T
            RTE[k] = np.dot(R[k].T, E.T).T

        if isinstance(sample_idx, type(None)):
            sample_idx = range(self.n_entities)

        for i in sample_idx:
            if self.approx_diag:
                self._sample_entity_diag(X, mask, E, R, i, var_e, RE, RTE)
            else:
                self._sample_entity(X, mask, E, R, i, var_e, RE, RTE)
            for k in range(self.n_relations):
                RE[k][i] = np.dot(R[k], E[i])
                RTE[k][i] = np.dot(R[k].T, E[i])

    def _sample_entity_diag(self, X, mask, E, R, i, var_e, RE, RTE):
        _lambda = np.ones(self.n_dim) / var_e

        nz_r = mask[:, i, :].nonzero()
        nz_c = mask[:, :, i].nonzero()
        nnz_r = nz_r[0].size
        nnz_c = nz_c[0].size
        nnz_all = nnz_r + nnz_c
        self.features[:nnz_r] = RE[nz_r]
        self.features[nnz_r:nnz_all] = RTE[nz_c]
        self.Y[:nnz_r] = X[:, i, :][nz_r]
        self.Y[nnz_r:nnz_all] = X[:, :, i][nz_c]

        features = self.features[:nnz_all]
        Y = self.Y[:nnz_all]
        try:
            logit = LogisticRegression(penalty='l2', C=1.0 / var_e, fit_intercept=False)
            logit.fit(features, Y)
            mu = logit.coef_[0]
            prd = logit.predict_proba(features)
            _lambda += np.sum(features.T ** 2 * prd[:, 0] * prd[:, 1], 1)
        except:
            mu = np.zeros(self.n_dim)

        inv_lambda = 1. / _lambda
        E[i] = np.random.normal(mu, inv_lambda)

    def _sample_entity(self, X, mask, E, R, i, var_e, RE, RTE):
        _lambda = np.identity(self.n_dim) / var_e

        nz_r = mask[:, i, :].nonzero()
        nz_c = mask[:, :, i].nonzero()
        nnz_r = nz_r[0].size
        nnz_c = nz_c[0].size
        nnz_all = nnz_r + nnz_c
        self.features[:nnz_r] = RE[nz_r]
        self.features[nnz_r:nnz_all] = RTE[nz_c]
        self.Y[:nnz_r] = X[:, i, :][nz_r]
        self.Y[nnz_r:nnz_all] = X[:, :, i][nz_c]

        features = self.features[:nnz_all]
        Y = self.Y[:nnz_all]
        try:
            logit = LogisticRegression(penalty='l2', C=1.0 / var_e, fit_intercept=False)
            logit.fit(features, Y)
            mu = logit.coef_[0]
            prd = logit.predict_proba(features)
            _lambda += np.dot(features.T * (prd[:, 0] * prd[:, 1]), features)
        except:
            mu = np.zeros(self.n_dim)

        inv_lambda = np.linalg.inv(_lambda)
        E[i] = multivariate_normal(mu, inv_lambda)

    def _sample_relations(self, X, mask, E, R, var_r):
        EXE = np.kron(E, E)

        for k in range(self.n_relations):
            self._sample_relation(X, mask, E, R, k, EXE, var_r)

    def _sample_relation(self, X, mask, E, R, k, EXE, var_r):
        if self.approx_diag:
            _lambda = np.ones(self.n_dim ** 2) / var_r
        else:
            _lambda = np.identity(self.n_dim ** 2) / var_r

        kron = EXE[mask[k].flatten() == 1]
        Y = X[k][mask[k] == 1].flatten()

        if len(np.unique(Y)) == 2:
            logit = LogisticRegression(penalty='l2', C=1.0 / var_r, fit_intercept=False)
            logit.fit(kron, Y)
            mu = logit.coef_[0]
            prd = logit.predict_proba(kron)

            if self.approx_diag:
                _lambda += np.sum(kron.T ** 2 * prd[:, 0] * prd[:, 1], 1)
            else:
                _lambda += np.dot(kron.T * (prd[:, 0] * prd[:, 1]), kron)
        else:
            mu = np.zeros(self.n_dim ** 2)

        if self.approx_diag:
            inv_lambda = 1. / _lambda
            R[k] = np.random.normal(mu, inv_lambda).reshape(R[k].shape)
        else:
            inv_lambda = np.linalg.inv(_lambda)
            R[k] = multivariate_normal(mu, inv_lambda).reshape(R[k].shape)

    def _reconstruct(self, E, R):
        _X = np.zeros([self.n_relations, self.n_entities, self.n_entities])

        for k in range(self.n_relations):
            _X[k] = np.dot(np.dot(E, R[k]), E.T)

        return _X

    def score(self, X, mask):
        return 0

    def _compute_fit(self, X, mask):
        p = self.p_weights.argmax()
        _X = self._reconstruct(self.E[p], self.R[p])
        _X = _sigmoid(_X)
        return self.eval_fn(X[mask == 1].flatten(), _X[mask == 1].flatten())

    def _save_model(self, seq):
        import pickle

        with open(self.dest, 'wb') as f:
            pickle.dump([self, seq], f)


if __name__ == '__main__':
    import itertools
    from sklearn.metrics import mean_squared_error, roc_auc_score

    n_dim = 5
    n_relation = 5
    n_entity = 10
    n_particle = 10

    E = np.random.normal(0, 1.0, size=[n_entity, n_dim])
    R = np.random.normal(0, 1.0, size=[n_relation, n_dim, n_dim])

    X = np.zeros([n_relation, n_entity, n_entity])
    for k, i, j in itertools.product(range(n_relation), range(n_entity), range(n_entity)):
        x = np.dot(np.dot(E[i].T, R[k]), E[j])
        p = _sigmoid(x)
        X[k, i, j] = np.random.binomial(1, p)

    model = PFBayesianLogitRescal(n_dim, n_particles=n_particle, eval_fn=roc_auc_score)
    # seq = model.fit(X, np.zeros_like(X))
    seq = model.fit(X, np.random.binomial(1, 0.1, X.shape))  # test with initial observations

import logging
import time
import numpy as np
from numpy.random import multivariate_normal, multinomial
from sklearn.linear_model import LogisticRegression
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
_PULL_SIZE = 1
_DEST = ''

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

        self.rbp = rbp
        self.compute_score = compute_score

        self.sample_prior = kwargs.pop('sample_prior', _P_SAMPLE)
        self.prior_sample_gap = kwargs.pop('prior_sample_gap', _P_SAMPLE_GAP)
        self.e_alpha = kwargs.pop('e_alpha', _E_ALPHA)
        self.e_beta = kwargs.pop('e_beta', _E_BETA)
        self.r_alpha = kwargs.pop('r_alpha', _R_ALPHA)
        self.r_beta = kwargs.pop('r_beta', _R_BETA)

        self.mc_move = kwargs.pop('mc_move', _MC_MOVE)

        self.pos_val = kwargs.pop('pos_val', _POS_VAL)
        self.dest = kwargs.pop('dest', _DEST)

        self.pull_size = kwargs.pop('pull_size', _PULL_SIZE)

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

        self.log = log

    def __getstate__(self):
        d = dict(self.__dict__)
        if self.controlled_var:
            del d['var_X']
        return d

    def fit(self, X, obs_mask=None, max_iter=0):
        self.n_relations = X.shape[0]
        self.n_entities = X.shape[1]

        self.E = list()
        self.R = list()

        if type(obs_mask) == type(None):
            obs_mask = np.zeros_like(X)

        if max_iter == 0:
            max_iter = int(np.prod([self.n_relations, self.n_entities, self.n_entities]) - np.sum(obs_mask))

        # for controlled variance
        if self.controlled_var:
            self.var_X = np.ones_like(X) * self.unobs_var
            self.var_X[obs_mask == 1] = self.obs_var

        for p in range(self.n_particles):
            self.E.append(np.random.random([self.n_entities, self.n_dim]))
            self.R.append(np.random.random([self.n_relations, self.n_dim, self.n_dim]))

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
        cur_obs = np.zeros_like(X)
        cur_obs[mask == 1] = X[mask == 1]

        pop = 0

        for i in range(max_iter):
            tic = time.time()

            for pn in range(self.pull_size):
                next_idx = self.get_next_sample(mask, cur_obs)
                yield next_idx
                cur_obs[next_idx] = X[next_idx]
                mask[next_idx] = 1

                if X[next_idx] == self.pos_val:
                    pop += 1

                if self.controlled_var:
                    self.var_X[next_idx] = self.obs_var

                logger.info('[NEXT] %s: %.3f, population: %d/%d', str(next_idx), X[next_idx], pop,
                            (i * self.pull_size + pn))

                self.p_weights *= self.compute_particle_weight(next_idx, cur_obs, mask)
                self.p_weights /= np.sum(self.p_weights)

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
                _fit = self._compute_fit(cur_obs, mask)
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

    def _sample_prior(self):
        self._sample_var_r()
        self._sample_var_e()

    def _sample_var_r(self):
        pass

    def _sample_var_e(self):
        pass

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
        features = np.zeros([2 * self.n_entities * self.n_relations, self.n_dim])
        Y = np.zeros([2 * self.n_entities * self.n_relations])

        idx = 0
        for k in range(self.n_relations):
            tmp = RE[k][mask[k, i, :] == 1]  # ExD
            tmp2 = RTE[k][mask[k, :, i] == 1]

            if tmp.shape[0] != 0:
                xTmp = X[k, i, mask[k, i, :] == 1]
                next_idx = idx + np.sum(mask[k, i, :] == 1)
                features[idx:next_idx] = RE[k][mask[k, i, :] == 1]
                Y[idx:next_idx] = xTmp
                idx = next_idx

            if tmp2.shape[0] != 0:
                xTmp2 = X[k, mask[k, :, i] == 1, i]
                next_idx = idx + np.sum(mask[k, :, i] == 1)
                features[idx:next_idx] = RTE[k][mask[k, :, i] == 1]
                Y[idx:next_idx] = xTmp2
                idx = next_idx

        features = features[:idx]
        Y = Y[:idx]
        if len(np.unique(Y[:idx])) == 2:
            logit = LogisticRegression(penalty='l2', C=1. / var_e, fit_intercept=False)
            logit.fit(features, Y)
            mu = logit.coef_[0]
            prd = logit.predict_proba(features)
            _lambda += np.dot(features.T * (prd[:, 0] * prd[:, 1]), features)
        else:
            mu = np.zeros(self.n_dim)

        E[i] = multivariate_normal(mu, np.linalg.inv(_lambda))

    def _sample_relations(self, X, mask, E, R, var_r):
        EXE = np.kron(E, E)

        for k in range(self.n_relations):
            self._sample_relation(X, mask, E, R, k, EXE, var_r)

    def _sample_relation(self, X, mask, E, R, k, EXE, var_r):
        _lambda = np.identity(self.n_dim ** 2) / var_r

        kron = EXE[mask[k].flatten() == 1]

        Y = X[k][mask[k] == 1].flatten()

        if len(np.unique(Y)) == 2:
            logit = LogisticRegression(penalty='l2', C=1. / var_r, fit_intercept=False)
            logit.fit(kron, Y)
            mu = logit.coef_[0]
            prd = logit.predict_proba(kron)
            _lambda += np.dot(kron.T * (prd[:, 0] * prd[:, 1]), kron)
        else:
            mu = np.zeros(self.n_dim ** 2)

        inv_lambda = np.linalg.inv(_lambda)
        R[k] = multivariate_normal(mu, inv_lambda).reshape([self.n_dim, self.n_dim])

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

    n_dim = 5
    n_relation = 10
    n_entity = 10
    n_particle = 10

    E = np.random.normal(0, 1.0, size=[n_entity, n_dim])
    R = np.random.normal(0, 1.0, size=[n_relation, n_dim, n_dim])

    X = np.zeros([n_relation, n_entity, n_entity])
    for k, i, j in itertools.product(range(n_relation), range(n_entity), range(n_entity)):
        x = np.dot(np.dot(E[i].T, R[k]), E[j])
        p = 1. / (1. + np.exp(-x))
        X[k, i, j] = np.random.binomial(1, p)

    model = PFBayesianLogitRescal(n_dim, controlled_var=False, n_particles=n_particle)
    seq = model.fit(X, np.zeros_like(X))

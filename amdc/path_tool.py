__author__ = 'Dongwoo Kim'

import numpy as np
import itertools
from collections import defaultdict


def sample_broken_tri(T, link_val=1):
    """
    find three nodes which do not have triangular path (i->j->k<-i) and corresponding relations a, b, c
    @param T: graph tensor matrix
    @return: tuple (a, b, c) where a, b, c is an index of link (i->j), (j->k), (i->k)
    """
    find = False

    while not find:
        i, j, k = np.random.permutation(range(T.shape[0]))[:3]
        a, b, c = np.random.randint(T.shape[2], size=3)

        if not (T[i, j, a] == link_val and T[j, k, b] == link_val and T[i, k, c] == link_val):
            find = True

    return ((i, j, a), (j, k, b), (i, k, c))


def tri_index(T, link_val=1):
    """
    extract indices of every possible triangular path in the graph
    especially for the following path structure
    i -> j
    j -> k
    i -> k

    @param T: [E x E x K] tensor graph where T[i,j,k] = 1 when there is type k link between node i and j
    @return: list of tuples consist of (a, b, c) where a, b, c is an index of link (i->j), (j->k), (i->k)
    """
    T = T.copy()
    T[T!=link_val] = 0

    e, k = T.shape[0], T.shape[2]
    T_squeeze = np.sum(T, 2)
    indices = list()

    link_types = defaultdict(list)
    for i, j in itertools.permutations(range(e), 2):
        _tmp = np.nonzero(T[i, j, :])[0]
        if len(_tmp) != 0:
            link_types[(i, j)] = np.nonzero(T[i, j, :])[0]

    for i in range(e):
        out_links = np.setdiff1d(np.nonzero(T_squeeze[i, :])[0], i)

        for j, k in itertools.permutations(out_links, 2):
            if T_squeeze[j, k] != 0:  # at least one edge from j to k exists
                type_1, type_2, type_3 = link_types[(i, j)], link_types[(j, k)], link_types[(i, k)]

                for types in itertools.product(type_1, type_2, type_3):
                    a = (i, j, types[0])
                    b = (j, k, types[1])
                    c = (i, k, types[2])
                    indices.append((a, b, c))

    return indices


def test():
    from scipy.io.matlab import loadmat
    mat = loadmat('../data/alyawarradata.mat')
    T = np.array(mat['Rs'], np.float32)

    indices = tri_index(T)

    print(len(indices))
    for ix in range(10):
        a, b, c = indices[ix]
        i, j, t1 = a
        j, k, t2 = b
        i, k, t3 = c
        print('a path %d->%d->%d by type %d/%d and a link %d->%d by type %d' % (i, j, k, t1, t2, i, k, t3))

    for ix in range(len(indices)):
        assert T[i, j, t1] and T[j, k, t2] and T[i, k, t3]


if __name__ == '__main__':
    test()

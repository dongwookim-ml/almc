"""
Implementation of Order&Extend model
Reference: [Matrix Completion with Queries, Ruchansky et al, KDD 2015]
"""
import numpy as np
from collections import defaultdict
import numpy.linalg as linalg

x_dim = 0
y_dim = 1

inv_dim = lambda x: x^1

class Node:
    """ For the bipartite graph """
    def __init__(self, dim, idx, degree):
        self.dim = dim
        self.idx = idx
        self.degree = degree
        self.adj_degree = degree    # used for graph degeneracy

    def __repr__(self):
        return '%d:%d:%d:%d' % (self.dim, self.idx, self.degree, self.adj_degree)

    def decrease_adj_degree(self):
        self.adj_degree -= 1

class OrderExtend:
    """ model class """
    def __init__(self, T, sigma, r):
        self.T = T  # original (true) matrix
        self.sigma = sigma
        self.T_sigma = self.T * self.sigma # observed (+queried) matrix
        self.nx, self.ny = self.T.shape
        self.nr = r

        self.queried = list()

        self.budget = self.nx * self.ny
        self.budget_used = 0

        self.x = np.zeros((self.nx, self.nr))
        self.y = np.zeros((self.ny, self.nr))

        self.x_computed = np.zeros(self.nx)
        self.y_computed = np.zeros(self.ny)

    def check_local_condition(self):
        return

    def stabilize(self):
        return
        
    def find_ordering(self):
        """
        Find initial ordering with graph degeneracy algorithm
        Reference: https://en.wikipedia.org/wiki/Degeneracy_(graph_theory)
        Repositioning algorithm describe in section 4.2 of the original paper should be added
        """

        pi = list()
        self.node_dict = dict()
        counter = defaultdict(list)
        mask = self.sigma.copy()
        max_degree = 0

        #compute degree of x and y
        for xi in range(self.nx):
            node = Node(x_dim, xi, np.count_nonzero(self.sigma[xi, :]))
            self.node_dict[(x_dim, xi)] = node

            counter[node.degree].append(node)
            if node.degree > max_degree:
                max_degree = node.degree

        for yi in range(self.ny):
            node = Node(y_dim, yi, np.count_nonzero(self.sigma[:, yi]))
            self.node_dict[(y_dim, yi)] = node

            counter[node.degree].append(node)
            if node.degree > max_degree:
                max_degree = node.degree

        inserted = 0
        while inserted < self.nx+self.ny:
            for di in range(max_degree):
                if len(counter[di]) > 0:
                    node = counter[di].pop(0)
                    pi.append(node)
                    inserted += 1

                    if node.dim == x_dim:
                        adj_array = np.nonzero(mask[node.idx, :])[0]
                        for adj in adj_array:
                            tmp = self.node_dict[(y_dim, adj)]
                            counter[tmp.adj_degree].remove(tmp)
                            tmp.decrease_adj_degree()
                            counter[tmp.adj_degree].append(tmp)
                        mask[node.idx,:] = 0
                    else:
                        adj_array = np.nonzero(mask[:, node.idx])[0]
                        for adj in adj_array:
                            tmp = self.node_dict[(x_dim, adj)]
                            counter[tmp.adj_degree].remove(tmp)
                            tmp.decrease_adj_degree()
                            counter[tmp.adj_degree].append(tmp)
                        mask[:,node.idx] = 0

                    break

        pi.reverse()

        return pi

    def construct_init_matrix(self, pi):
        """
        construct initial nr x nr matrix using SVD
        """
        x_list = list()
        y_list = list()

        x_cnt = 0
        y_cnt = 0

        idx = 0
        while x_cnt < self.nr or y_cnt < self.nr:
            dim = pi[idx].dim

            if dim == x_dim and x_cnt < self.nr:
                x_list.append(pi.pop(idx).idx)
                x_cnt += 1
            elif dim == y_dim and y_cnt < self.nr:
                y_list.append(pi.pop(idx).idx)
                y_cnt += 1
            else:
                idx += 1

        for xi in x_list:
            for yi in y_list:
                if self.sigma[xi, yi] == 0:
                    self.query(xi, yi)

        U, s, V = linalg.svd(self.T_sigma[np.ix_(x_list, y_list)], full_matrices = True)

        self.x[x_list,:] = np.dot(U, np.diag(s))
        self.y[y_list,:] = V.T

        self.x_computed[x_list] = 1
        self.y_computed[y_list] = 1

    def query(self, x_idx, y_idx):
        """
        query the value of T[x_idx,y_idx] to oracle
        """
        if self.budget > 0 and self.sigma[x_idx, y_idx] == 0:
            print('\tNew Query on (%d,%d)' % (x_idx, y_idx))
            self.sigma[x_idx, y_idx] = 1
            self.T_sigma[x_idx, y_idx] = self.T[x_idx, y_idx]
            self.queried.append((x_idx,y_idx))
            self.budget -= 1
            self.budget_used += 1
            return self.T_sigma[x_idx, y_idx]
        else:
            return None

    def reconstruct(self):
        """ matrix reconstruction """
        return np.dot(self.x, self.y.T)

    def compute_reconstruction_error(self):
        """ compute relative error between the true and reconstructed matrix """
        return np.sum(np.sqrt((self.T - self.reconstruct())**2)) / np.sum(np.sqrt(self.T**2))

    def fit(self):
        pi = self.find_ordering()
        self.construct_init_matrix(pi)

        print('Init Relative Error: %.3f' % self.compute_reconstruction_error())

        iter = 0
        while self.budget > 0 and len(pi) > 0:
            next_node = pi.pop(0)
            if next_node.dim == x_dim:
                while np.sum(self.sigma[next_node.idx, :] * self.y_computed) < self.nr - 0.1:
                    candidate_idx = np.nonzero((self.y_computed - self.sigma[next_node.idx, :])==1)[0]
                    max_degree = -1
                    max_degree_idx = -1
                    for _idx in candidate_idx:
                        candidate_degree = self.node_dict[(y_dim, _idx)].degree
                        if candidate_degree > max_degree:
                            max_degree = candidate_degree
                            max_degree_idx = _idx

                    self.query(next_node.idx, max_degree_idx)
                y_list = np.nonzero(self.sigma[next_node.idx, :] * self.y_computed)[0][:self.nr]
                self.x[next_node.idx, :] = linalg.lstsq(self.y[y_list, :], self.T_sigma[next_node.idx, y_list])[0]
                self.x_computed[next_node.idx] = 1
            else:
                while np.sum(self.sigma[:, next_node.idx] * self.x_computed) < self.nr - 0.1:
                    candidate_idx = np.nonzero((self.x_computed - self.sigma[:, next_node.idx])==1)[0]
                    max_degree = -1
                    max_degree_idx = -1
                    for _idx in candidate_idx:
                        candidate_degree = self.node_dict[(x_dim, _idx)].degree
                        if candidate_degree > max_degree:
                            max_degree = candidate_degree
                            max_degree_idx = _idx

                    self.query(max_degree_idx, next_node.idx)
                x_list = np.nonzero(self.sigma[:, next_node.idx] * self.x_computed)[0][:self.nr]
                self.y[next_node.idx, :] = linalg.lstsq(self.x[x_list, :], self.T_sigma[x_list, next_node.idx])[0]
                self.y_computed[next_node.idx] = 1

            iter += 1
            print('Iteration %d, Relative Error: %.3f' % (iter, self.compute_reconstruction_error()))

        print('Total budget used: %d' % self.budget_used)
        return self.reconstruct(), self.x, self.y


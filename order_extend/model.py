"""
Implementation of Order&Extend model
Reference: [Matrix Completion with Queries, Ruchansky et al, KDD 2015]
"""
__author__ = 'Dongwoo Kim'

from collections import defaultdict

import numpy as np
import numpy.linalg as linalg

from ..utils.formatted_logger import formatted_logger

log = formatted_logger('OrderExtend', 'info')

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
    def __init__(self, T, sigma, r, theta=1):
        self.T = T  # original (true) matrix
        self.sigma = sigma
        self.T_sigma = self.T * self.sigma # observed (+queried) matrix
        self.nx, self.ny = self.T.shape
        self.nr = r
        self.theta = theta

        self.queried = list()

        self.budget = self.nx * self.ny
        self.budget_used = 0

        self.x = np.zeros((self.nx, self.nr))
        self.y = np.zeros((self.ny, self.nr))

        self.x_computed = np.zeros(self.nx)
        self.y_computed = np.zeros(self.ny)

    def compute_lcn(self,A,t):
        """
        compute local condition number
        a linear system assumed to be unstable if a local condition number of the system, Ay=t, is greater than threahold theta
        """
        y = linalg.solve(A, t)
        lcn = linalg.norm(linalg.inv(A), ord=2)*linalg.norm(t, ord=2)/linalg.norm(y, ord=2)
        return lcn

    def compute_lcn_extend(self, A, t, C=None, alpha=None, ix=None, iy=None):
        """
        compute local condition number with extended vector
        """
        if self.sigma[ix,iy] == 1:
            tau = self.T_sigma[ix,iy]
        else:
            try:
                tau1 = np.random.choice(self.T_sigma[ix,self.sigma[ix,:]==1])
                tau2 = np.random.choice(self.T_sigma[self.sigma[:,iy]==1,iy])
                tau = np.random.choice([tau1,tau2])
            except:
                tau = np.random.choice(self.T_sigma[self.sigma==1])
        D = C - np.dot(np.dot(np.dot(C, alpha.T), alpha), C)/(1.+ np.dot(np.dot(alpha, C), alpha.T))
        A_tilda = np.concatenate((A, alpha[:,np.newaxis].T), axis=0)
        t_tilda = np.zeros(self.nr+1)
        t_tilda[:self.nr] = t
        t_tilda[self.nr] = tau
        y_tilda = np.dot(np.dot(D, A_tilda.T), t_tilda)
        lcn = linalg.norm(np.dot(D, A_tilda.T), ord=2)*linalg.norm(t_tilda, ord=2)/linalg.norm(y_tilda, ord=2)
        
        return lcn, tau

    def stabilize(self, A, t, node, selected):
        """
        check that the given node can be solved with a stable linear system
        """
        C = linalg.inv(np.dot(A.T, A))

        best_idx = 0
        c_min = self.theta+1
        a_idx = -1

        if node.dim == x_dim:
            for candidate in np.setdiff1d(np.nonzero(self.y_computed)[0], selected):
                (c, _tau) = self.compute_lcn_extend(A, t, C, self.y[candidate,:], node.idx, candidate)
                if c < c_min:
                    a_star = self.y[candidate,:]
                    a_idx = candidate
                    c_min = c 
                    tau = _tau
        else:
            for candidate in np.setdiff1d(np.nonzero(self.x_computed)[0], selected):
                (c, _tau) = self.compute_lcn_extend(A, t, C, self.x[candidate,:], candidate, node.idx)
                if c < c_min:
                    a_star = self.x[candidate,:]
                    a_idx = candidate
                    c_min = c
                    tau = _tau

        if c_min < self.theta:
            log.debug('c_min : %f' % c_min)
            return a_star, a_idx

        return None
        
    def find_ordering(self):
        """
        Find initial ordering with graph degeneracy algorithm
        Reference: https://en.wikipedia.org/wiki/Degeneracy_(graph_theory)
        Repositioning algorithm describe in section 4.2 of the original paper should be added
        (But the description seems not clear enough to implement)
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
        construct init matrix
        choose first r x-indices from ordered list pi and initialize A_x by I_r (r x r identity matrix)
        """
        x_list = list()
        x_cnt = 0
        idx = 0

        while x_cnt < self.nr:
            dim = pi[idx].dim

            if dim == x_dim and x_cnt < self.nr:
                x_list.append(pi.pop(idx).idx)
                x_cnt += 1
            else:
                idx += 1

        self.x[x_list,:] = np.identity(self.nr)
        self.x_computed[x_list] = 1

    def query(self, x_idx, y_idx):
        """
        query the value of T[x_idx,y_idx] to oracle
        """
        if self.budget > 0 and self.sigma[x_idx, y_idx] == 0:
            log.debug('\tNew Query on (%d,%d)' % (x_idx, y_idx))
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
        return linalg.norm(self.T - self.reconstruct()) / linalg.norm(self.T)

    def init(self):
        pi = self.find_ordering()
        self.construct_init_matrix(pi)
        return pi

    def fit(self, pi, max_iter=-1):
        log.debug('Init Relative Error: %.3f' % self.compute_reconstruction_error())

        if max_iter < 0:
            max_iter = self.budget*2

        iter = 1
        while self.budget > 0 and len(pi) > 0 and iter < max_iter:
            solve_system = True
            next_node = pi.pop(0)

            if next_node.dim == x_dim:
                # check that the model needs to query a new data point
                while np.sum(self.sigma[next_node.idx, :] * self.y_computed) < self.nr - 0.1:
                    candidate_idx = np.nonzero((self.y_computed - self.sigma[next_node.idx, :])==1)[0]
                    if len(candidate_idx) == 0:
                        solve_system = False
                        break
                    # max degree quirying strategy                        
                    # max_degree = -1
                    # query_idx = -1
                    # for _idx in candidate_idx:
                    #     candidate_degree = self.node_dict[(y_dim, _idx)].degree
                    #     if candidate_degree > max_degree:
                    #         max_degree = candidate_degree
                    #         query_idx = _idx

                    # random querying strategy
                    query_idx = np.random.choice(candidate_idx)
                    self.query(next_node.idx, query_idx)

                candidate_list = np.nonzero(self.sigma[next_node.idx, :] * self.y_computed)[0]

                # iterate every possible candidate set to find stable solution
                # this requires too much computation
                #for y_list in itertools.combinations(candidate_list, self.nr):
                
                if len(candidate_list) >= self.nr:
                    y_list = np.random.permutation(candidate_list)[:self.nr]
                    A = self.y[y_list, :]
                    t = self.T_sigma[next_node.idx, y_list]

                    if self.compute_lcn(A, t) > self.theta:
                        rval = self.stabilize(A, t, next_node, y_list)
                        if rval == None:
                            pi.append(next_node)
                            solve_system = False
                        else:
                            self.query(next_node.idx, rval[1])                            
                            A = np.concatenate((A, rval[0][:,np.newaxis].T), axis=0)
                            t = np.zeros(self.nr+1)
                            t[:self.nr] = self.T_sigma[next_node.idx, y_list]
                            t[self.nr] = self.T_sigma[next_node.idx, rval[1]]
                            solve_system = True
                            # break
                else:
                    solve_system = False

                if solve_system:
                    log.debug('x solved, idx: %d'%(next_node.idx))
                    self.x[next_node.idx, :] = linalg.lstsq(A, t)[0]
                    self.x_computed[next_node.idx] = 1
            else:
                while np.sum(self.sigma[:, next_node.idx] * self.x_computed) < self.nr - 0.1:
                    candidate_idx = np.nonzero((self.x_computed - self.sigma[:, next_node.idx])==1)[0]
                    if len(candidate_idx) == 0:
                        solve_system = False
                        break         
                    # max degree quirying strategy
                    # max_degree = -1
                    # candidate_idx = -1
                    # for _idx in candidate_idx:
                    #     candidate_degree = self.node_dict[(x_dim, _idx)].degree
                    #     if candidate_degree > max_degree:
                    #         max_degree = candidate_degree
                    #         candidate_idx = _idx

                    # random querying strategy
                    query_idx = np.random.choice(candidate_idx)
                    self.query(query_idx, next_node.idx)

                candidate_list = np.nonzero(self.sigma[:, next_node.idx] * self.x_computed)[0]

                # iterating over every possible combination requires too much computation
                # for x_list in itertools.combinations(candidate_list, self.nr):
                if len(candidate_list) >= self.nr:
                    x_list = np.random.permutation(candidate_list)[:self.nr]
                    A = self.x[x_list, :]
                    t = self.T_sigma[x_list, next_node.idx]

                    if self.compute_lcn(A, t) > self.theta:
                        rval = self.stabilize(A, t, next_node, x_list)
                        if rval == None:
                            pi.append(next_node)
                            solve_system = False
                        else:
                            self.query(rval[1], next_node.idx)
                            A = np.concatenate((A, rval[0][:,np.newaxis].T), axis=0)
                            t = np.zeros(self.nr+1)
                            t[:self.nr] = self.T_sigma[x_list, next_node.idx]
                            t[self.nr] = self.T_sigma[rval[1], next_node.idx]
                            solve_system = True                                  
                            # break
                else:
                    solve_system = False                            

                if solve_system:
                    log.debug('y solved, idx: %d'%(next_node.idx))
                    self.y[next_node.idx, :] = linalg.lstsq(A, t)[0]
                    self.y_computed[next_node.idx] = 1

            log.debug('Iteration %d, Relative Error: %.3f' % (iter, self.compute_reconstruction_error()))
            iter += 1

        log.info('Total budget used: %d / %d' % (self.budget_used, np.prod(self.sigma.shape)))
        log.info('Final relative error: %.3f' % self.compute_reconstruction_error())
        log.info('Solved x-dim: %d y-dim: %d' % (np.sum(self.x_computed), np.sum(self.y_computed)))
        return self.reconstruct(), self.x, self.y


def test():
    nx = 10  # size of x-dim
    ny = 10   # size of y-dim
    r = 2     # original rank
    p = 0.3   # mean proportion of observed items in matrix
    r_predicted = 2  # rank used for approx.
    theta = 2
    max_iter = 10000

    x = np.random.random((nx, r))
    y = np.random.random((ny, r))
    t = np.dot(x, y.T)  # true matrix

    sigma = np.random.binomial(1, p, size=(nx,ny))   # mask of observed items in matrix t

    model = OrderExtend(t, sigma, r_predicted, theta)
    order = model.init()
    model.fit(order, max_iter)

if __name__ == '__main__':
    test()

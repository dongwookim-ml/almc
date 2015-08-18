import numpy as np
from model import OrderExtend

def test():
    nx = 10  # size of x-dim
    ny = 5   # size of y-dim
    r = 2     # original rank
    p = 0.3   # mean proportion of observed items in matrix
    r_predicted = 4 # rank used for approx.
    max_iter = 100

    x = np.random.random((nx, r))
    y = np.random.random((ny, r))
    t = np.dot(x, y.T)  # true matrix

    sigma = np.random.binomial(1, p, size=(nx,ny))   # mask of observed items in matrix

    model = OrderExtend(t, sigma, r_predicted)
    order = model.init()
    model.fit(order, max_iter=max_iter)

if __name__ == '__main__':
    test()

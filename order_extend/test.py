import numpy as np
from model import OrderExtend

def test():
    nx = 100  # size of x-dim
    ny = 50   # size of y-dim
    r = 5     # original rank
    p = 0.2   # mean proportion of observed items in matrix
    r_predicted = 6  # rank used for approx.

    x = np.random.random((nx, r))
    y = np.random.random((ny, r))
    t = np.dot(x, y.T)  # true matrix

    sigma = np.random.binomial(1, p, size=(nx,ny))   # mask of observed items in matrix

    model = OrderExtend(t, sigma, r_predicted)
    model.fit()


if __name__ == '__main__':
    test()

import numpy as np
from scipy.io.matlab import loadmat
from brescal import BayesianRescal

if __name__ == '__main__':
    """
    Test with Kinship dataset
    Use all positive triples and negative triples as a training set
    See how the reconstruction error is reduced during training
    """
    mat = loadmat('../data/alyawarradata.mat')
    T = np.array(mat['Rs'], np.float32)

    D = 10
    T = np.swapaxes(T,1,2)
    T = np.swapaxes(T,0,1)
    print(T.shape)
    model = BayesianRescal(D)
    model.fit(T)


import numpy as np
from brescal import BayesianRescal
from scipy.io.matlab import loadmat

if __name__ == '__main__':
    """
    Test with Kinship dataset
    """
    mat = loadmat('../data/alyawarradata.mat')
    T = np.array(mat['Rs'], np.float32)

    D = 10
    T = np.swapaxes(T,1,2)
    T = np.swapaxes(T, 0, 1)  # [relation, entity, entity]
    print(T.shape)
    model = BayesianRescal(D)
    model.fit(T)


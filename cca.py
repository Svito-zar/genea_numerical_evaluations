from sklearn.cross_decomposition import CCA
from scipy import stats
import numpy as np


def calculate_CCA(input_array_one, input_array_two):
    """
    Calculate CCA (Canonical Correlation Analysis) coefficient
    Args:
        input_array_one: the first input array [T,D]
        input_array_two: the second input array [T,D]

    Returns:
        r:  Pearson Correlation Coefficient after CCA transformation (scalar)

    """

    # Define CCA model which considers the first CCA coefficient only
    cca = CCA(n_components=1)
    # Fit CCA model to the given data
    cca.fit(input_array_one, input_array_two)
    # Encode the given arrays into 1D space using the CCA linear transform
    encoding_one, encoding_two = cca.transform(input_array_one, input_array_two)

    # Standartize arrays shape: make it np.array of floats and remove any dummy dimensions
    encoding_one = np.array(encoding_one, dtype='float64').squeeze()
    encoding_two = np.array(encoding_two, dtype='float64').squeeze()

    # Calculate Pearson Correlation Coefficient (and p-value, which we don't use)
    r, p = stats.pearsonr(encoding_one, encoding_two)

    return r


if __name__ == '__main__':

    #X = [[0., 0., 1.], [1.,0.,0.], [2.,2.,2.], [3.,5.,4.]]
    #Y = [[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]]

    #X = [[1, 2], [2, 3], [3, 4], [4, 4], [0., 1.], [1.,0.], [2.,2.], [3.,4.], [1, 2], [2, 3], [3, 4], [4, 4], [0., 1.], [1.,0.], [2.,2.], [3.,4.]]
    #Y = [[2.5, 21], [2, 2.5], [2.01, 2], [2.8, 2.01], [0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3], [1, 2], [2, 3], [3, 4], [4, 4], [0., 1.], [1.,0.], [2.,2.], [3.,4.]]

    #X = [[1], [2], [3], [4], [1], [2], [3], [4], [1], [2], [3], [4], [1], [2], [3], [4]]
    #Y = [[2], [2], [2.01], [2.01], [2], [2], [2.01], [2.01], [2], [2], [2.01], [2.01], [2], [2], [2.01], [2.01]]

    n = 10000
    X = np.random.randint(1,11,(n,20))
    Y = X*2 + 7 + np.random.randint(0,15,(n,20))

    corr = calculate_CCA(X,Y)

    print("CCA is : ", corr)
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from scipy.signal import convolve

class Utils:
    def __init__(self):
        pass

    def linear_regression(self, x, y, nj, return_variance = False):
        """
        Performs a (weighted or not) linear regression.
        Finds 'a' that minimizes the error:
            sum_j { n[j]*(y[j] - (a*x[j] + b))**2 }

        Args:
            x, y : regression variables
            nj: list containg the weigths
        Returns:
            a, b: angular coefficient and intercept

        (!!!!!!!!!!!!!)
        IMPORTANT:

        return_variance NOT DEBUGGED
        (!!!!!!!!!!!!!)
        """

        bj = np.array(nj, dtype = np.float)
        assert len(bj) == len(x)


        V_0 = np.sum(bj)
        V_1 = np.sum(bj * x)
        V_2 = np.sum(bj * (x**2))

        weights_slope = bj * (V_0*x - V_1)/(V_0*V_2 - V_1*V_1)
        weights_intercept = bj * (V_2 - V_1*x)/(V_0*V_2 - V_1*V_1)

        a = np.sum(weights_slope*y)
        b = np.sum(weights_intercept*y)

        var_a = np.sum((1/bj)*weights_slope*weights_slope)

        if not return_variance:
            return a, b
        else:
            return a, b, var_a

#conv2d(S1, S2, mode) first convolves each column of A with the vector
#H1 and then convolves each row of the result with the vector H2. 
def conv2d(s1, s2, mode, method, axis = 1):

    conv = lambda approx: convolve(approx, s2, mode = 'full', method='direct')
    s1 = np.apply_along_axis(conv, axis, s1)    
    return np.array(s1)


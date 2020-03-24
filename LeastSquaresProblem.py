# Least Squares problem
import numpy as np

class LeastSquares:
    def __init__(self, A, b):
        '''
        Initializes Least Squares problem as normalized squared l2-norm of (Ax - b)

        :param A: The data matrix with dimensions m x n
        :param b: The response variable with dimensions m

        m > n in order to assume independence of columns of A
        '''

        self.A = A
        self.b = b

    def stochastic_gradient(self, index, x):
        '''

        :param x: n-dimensional vector
        :return:
        '''

        grad = self.A[index, :] * (np.dot(self.A[index, :], x) - self.b[index])
        return grad

    def get_param_dim(self):
        '''

        :return: dimension n of matrix A
        '''
        return self.A.shape

    def evaluate(self, x):
        nb_data_points = self.b.shape[0]
        l = np.dot(self.A, x) - self.b
        fn_val = np.dot(l.T, l) / nb_data_points
        return fn_val

    def get_param(self):
        return self.A, self.b

    def evaluate_summand(self, index, x):
        return (np.dot(self.A[index, :], x) - self.b[index]) / 2

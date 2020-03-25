# Stochastic Gradient Descent
import matplotlib.pyplot as plt
import numpy as np
from helper.Lipschitz import calculate_lipschitz_distribution
from objectives.LeastSquaresProblem import LeastSquares
np.random.seed(123)

class StochasticGradientDescent:
    '''
    Performs stochastic gradient descent to a finite sum problem specified by an objective function.

    '''

    def __init__(self):
        pass

    def optimize(self, objective, n_iter=100, sampling_strategy='uniform'):
        '''

        :param objective: the function to optimize
        :param step_size: determines the length of the gradient step
        :param n_iter: determines how many gradient steps the algorithmm performs
        :param record_weights: wheter to output the realizations of the stochastic process {w_k} for all iterations k
        '''

        # get dimension of data matrix
        n_rows, n_cols = objective.get_param_dim()

        AtA = objective.A.T.dot(objective.A)
        eig_vals, _ = np.linalg.eig(AtA)
        initial_step_size = 1. / max(eig_vals)
        step_size = initial_step_size

        if sampling_strategy == 'uniform':
            p = np.ones(n_rows)
            p /= p.sum()
        elif sampling_strategy == 'lipschitz':
            p = calculate_lipschitz_distribution(objective)
        else:
            raise ValueError('sampling_strategy should be uniform or lipschitz')

        # initialzie learnable parameter
        x = np.random.rand(n_cols)

        # initialize history data structures
        weights = np.zeros((n_iter * n_rows + 1, n_cols))
        fn = [None for _ in range(n_iter * n_rows + 1)]
        fn[0] = objective.evaluate(x)
        weights[0, :] = x
        idx = 1

        # initalize indicies list to keep track of which indicies are sampled
        indicies = list()

        # loop for n_iter iterations
        for k in range(n_iter):

            # perform n_rows inner loops per iteration
            for _ in range(n_rows):
                index = np.random.choice(n_rows, p=p)
                grad = objective.stochastic_gradient(index, x)
                indicies.append(index)


                x = x - step_size * grad / (p[index] * n_rows)

                weights[idx, :] = x
                fn[idx] = objective.evaluate(x)
                idx += 1

            # diminishing step size
            if k > 0:
                step_size = initial_step_size / k

        # plot the results
        title = 'Stochastic gradient descent using {} sampling strategy'.format(sampling_strategy)
        plt.plot(np.arange(len(fn)), fn)
        plt.title(title)
        plt.xlabel('# iterations')
        plt.ylabel('f(x)')
        plt.yscale('log')
        plt.show()

        return x, weights, fn, indicies

if __name__ == '__main__':
    m, n = 100, 2

    A = np.random.rand(m, n)
    b = np.random.rand(m)
    ls = LeastSquares(A, b)

    sgd = StochasticGradientDescent()

    sgd.optimize(ls, sampling_strategy = 'lipschitz')
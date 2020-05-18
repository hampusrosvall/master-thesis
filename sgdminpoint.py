# Stochastic Gradient Descent
import matplotlib.pyplot as plt
import numpy as np
from helper.Lipschitz import calculate_lipschitz_distribution
from objectives.LeastSquaresProblem import LeastSquares
np.random.seed(123)

class SGDMinPoint:
    '''
    Performs stochastic gradient descent to a finite sum problem specified by an objective function.

    '''

    def __init__(self, starting_point):
        self.w = starting_point

    def optimize(self, objective, n_iter=100, sampling_strategy='uniform', analytical_sol = None):
        '''

        :param objective: the function to optimize
        :param step_size: determines the length of the gradient step
        :param n_iter: determines how many gradient steps the algorithmm performs
        :param record_weights: wheter to output the realizations of the stochastic process {w_k} for all iterations k
        '''
        n_rows, n_cols = objective.get_param_dim()
        pnts, weights = objective.get_param()

        initial_step_size = 1. / np.sum(weights)
        step_size = initial_step_size

        if sampling_strategy == 'uniform':
            p = np.ones(n_rows)
            p /= p.sum()
        elif sampling_strategy == 'lipschitz':
            p = objective.get_lipschitz_proba()
        else:
            raise ValueError('sampling_strategy should be uniform or lipschitz')

        # initialize history data structures
        weights = np.zeros((n_iter * n_rows + 1, n_cols))
        fn = [None for _ in range(n_iter * n_rows + 1)]
        fn[0] = objective.evaluate(self.w)
        weights[0, :] = self.w
        idx = 1

        # initalize indicies list to keep track of which indicies are sampled
        indicies = list()
        steps = []
        if analytical_sol is not None:
            distance_to_w = []

        # loop for n_iter iterations
        for k in range(n_iter):

            # perform n_rows inner loops per iteration
            for _ in range(n_rows):
                index = np.random.choice(n_rows, p=p)
                grad = objective.stochastic_gradient(index, self.w)
                indicies.append(index)

                step = step_size / (p[index] * n_rows)
                steps.append(step)
                self.w = self.w - step * grad

                if analytical_sol is not None:
                    distance_to_w.append(np.linalg.norm(self.w - analytical_sol))

                weights[idx, :] = self.w
                fn[idx] = objective.evaluate(self.w)
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

        if analytical_sol is not None:
            return self.w, weights, fn, indicies, distance_to_w, steps
        else:
            return self.w, weights, fn, indicies

if __name__ == '__main__':
    m, n = 100, 2

    A = np.random.rand(m, n)
    b = np.random.rand(m)
    ls = LeastSquares(A, b)

    sgd = SGDMinPoint()

    sgd.optimize(ls, sampling_strategy = 'lipschitz')
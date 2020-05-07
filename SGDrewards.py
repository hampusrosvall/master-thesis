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

    def __init__(self, starting_point):
        self.w = starting_point

    def optimize(self, objective, n_iter=1000, sampling_strategy='uniform', analytical_sol = None):
        '''

        :param objective: the function to optimize
        :param step_size: determines the length of the gradient step
        :param n_iter: determines how many gradient steps the algorithmm performs
        :param record_weights: wheter to output the realizations of the stochastic process {w_k} for all iterations k
        '''
        # get the initial function value 
        initial_fn_val = objective.evaluate(self.w)

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

        rewards = 0 

        # loop for n_iter iterations
        for k in range(n_iter):

            # perform n_rows inner loops per iteration
            for _ in range(n_rows):
                index = np.random.choice(n_rows, p=p)
                grad = objective.stochastic_gradient(index, self.w)

                step = step_size / (p[index] * n_rows)
                self.w = self.w - step * grad

                rewards += (objective.evaluate(self.w) / initial_fn_val) * -1

            # diminishing step size
            if k > 0:
                step_size = initial_step_size / k

        fn_val = objective.evaluate(self.w) 
        distance_from_wopt = np.linalg.norm(self.w - analytical_sol)

        return rewards, fn_val, distance_from_wopt

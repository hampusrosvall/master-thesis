import numpy as np
np.random.seed(123)
ALPHA = 0.99

class StochasticGradientEnvironment:
    def __init__(self, objective):
        self.objective = objective
        self.X, self.Y = objective.get_param()

        self.initial_step_size = self._initialize_step_size()
        self.step_size = self.initial_step_size

        # initialize starting point
        self.starting_point = np.random.rand(self.X.shape[1])

        # initialize weights parameter
        self.w = self.starting_point

        # parameter to keep track of diminishing step size
        self.k = 1

        self.n_summands, _ = objective.get_param_dim()

    def reset(self):
        """

        :return:
        """
        self.step_size = self.initial_step_size
        self.w = self.starting_point
        self.k = 1
        return self.objective.evaluate(self.w), self.w

    def step(self, probabilities, iteration):
        """

        :param action: the index to take a stochastic gradient step w.r.t
        :return: (observation, reward, done) tuple
        """
        if not iteration:
            self.function_value = self.objective.evaluate(self.w)

        action = np.random.choice(self.n_summands, p = probabilities)

        grad = self.objective.stochastic_gradient(action, self.w)

        step = self.step_size / (probabilities[action] * self.n_summands)
        self.w = self.w - step * grad
        self.decrease_step_size(iteration)

        approx_func_val = self.objective.evaluate_summand(action, self.w)

        observation = ALPHA * approx_func_val + (1 - ALPHA) * self.function_value

        # todo: sätta reward till noll förutom under sista iterationen (evalueras över alla N summander)
        # todo: evaluerar alla summander för varje iteration
        #reward = self.function_value - approx_func_val
        # reward = self.function_value -> där function_value är det sanna funtionsvärdet
        reward = self.objective.evaluate(self.w)
        self.function_value = observation

        return (observation, reward, action, self.w, step)

    def _initialize_step_size(self):
        XtX = self.X.T.dot(self.X)
        eig_vals, _ = np.linalg.eig(XtX)
        initial_step_size = 1. / max(eig_vals)
        return initial_step_size

    def decrease_step_size(self, iteration):
        if iteration > 0 and iteration % self.n_summands == 0:
            self.step_size = self.initial_step_size / self.k
            self.k += 1




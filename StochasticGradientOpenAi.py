import numpy as np

ALPHA = 0.9

class StochasticGradientEnvironment:
    def __init__(self, objective):
        self.objective = objective
        self.X, self.Y = objective.get_param()
        self.initial_step_size = self._initialize_step_size()
        self.reset()
        # initialize function value
        self.function_value = self.objective.evaluate(self.w)

        # parameter to keep track of diminishing step size
        self.k = 1

    def reset(self):
        """

        :return:
        """
        w_dim = self.X.shape[1]
        self.w = np.random.rand(w_dim)
        return self.objective.evaluate(self.w)

    def step(self, probabilities, iteration):
        """

        :param action: the index to take a stochastic gradient step w.r.t
        :return: (observation, reward, done) tuple
        """


        action = np.random.choice(len(probabilities), p = probabilities)


        grad = self.objective.stochastic_gradient(action, self.w)

        step_size = self.initial_step_size / iteration if iteration > 0 else self.initial_step_size
        self.w = self.w - step_size * grad / (probabilities[action] * self.X.shape[0])

        approx_func_val = self.objective.evaluate_summand(action, self.w)

        observation = ALPHA * approx_func_val + (1 - ALPHA) * self.function_value

        # todo: sätta reward till noll förutom under sista iterationen (evalueras över alla N summander)
        # todo: evaluerar alla summander för varje iteration
        reward = self.function_value - approx_func_val
        # reward = self.function_value -> där function_value är det sanna funtionsvärdet
        self.function_value = observation

        return (observation, reward, action, self.w)

    def _initialize_step_size(self):
        XtX = self.X.T.dot(self.X)
        eig_vals, _ = np.linalg.eig(XtX)
        initial_step_size = 1. / max(eig_vals)
        return initial_step_size

    def decrease_step_size(self):
        self.step_size = self.initial_step_size / self.k
        self.k += 1




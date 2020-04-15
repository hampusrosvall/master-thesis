import numpy as np
np.random.seed(123)
ALPHA = 0.9

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

    def step(self, probabilities, iteration, reward_type, max_iter):
        """

        :param action: the index to take a stochastic gradient step w.r.t
        :return: (observation, reward, done) tuple
        """
        if not iteration:
            self.function_value = self.objective.evaluate(self.w)

        action = np.random.choice(self.n_summands, p = probabilities)

        grad = self.objective.stochastic_gradient(action, self.w)

        step = self.step_size / (probabilities[action] * self.n_summands)
        old_w = self.w
        self.w = self.w - step * grad

        self.decrease_step_size(iteration)

        approx_func_val = self.objective.evaluate_summand(action, self.w)

        observation = ALPHA * approx_func_val + (1 - ALPHA) * self.function_value

        reward = self.calulate_reward(reward_type, iteration, max_iter, old_w, self.w)
        self.function_value = observation

        return (observation, reward, action, self.w, step)

    def calulate_reward(self, reward_type, iteration, max_iteration, old_w, new_w):
        if reward_type == 'function_value':
            return -self.objective.evaluate(self.w)

        if reward_type == 'function_diff':
            return - (self.objective.evaluate(new_w) -  self.objective.evaluate(old_w))

        if reward_type == 'last_iteration':
            if iteration == max_iteration - 1:
                return -self.objective.evaluate(self.w)
            else:
                return 0




    def _initialize_step_size(self):
        XtX = self.X.T.dot(self.X)
        eig_vals, _ = np.linalg.eig(XtX)
        initial_step_size = 1. / max(eig_vals)
        return initial_step_size

    def decrease_step_size(self, iteration):
        if iteration > 0 and iteration % self.n_summands == 0:
            self.step_size = self.initial_step_size / self.k
            self.k += 1




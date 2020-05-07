import numpy as np
import sys 
import json
from collections import deque

np.random.seed(123)

class StochasticGradientEnvironment:
    def __init__(self, objective, reward_type):
        self.objective = objective
        self.X, self.Y = objective.get_param()
        self.reward_type = reward_type

        self.initial_step_size = self.initialize_step_size()
        self.step_size = self.initial_step_size

        # initialize starting point
        self.starting_point = np.random.rand(self.X.shape[1])

        self.starting_fn_value = self.objective.evaluate(self.starting_point)

        # initialize weights parameter
        self.w = self.starting_point

        # parameter to keep track of diminishing step size
        self.k = 1

        # set value of epsilon for epsilon-greedy policy 
        self.set_params()

        self.n_summands, _ = objective.get_param_dim()

        if self.reward_type == 'smoothing':
            self.reward_window = deque(maxlen = 10)

    def set_params(self):
        if len(sys.argv) == 1:
            file_name = './standard_paramters.json'
        else:
            file_name = sys.argv[1]

        with open(file_name, 'r') as f:
            data_input = json.load(f)
            param = data_input['parameters']['epsilon_policy']
            self.epsilon = param['start_value']
            self.epsilon_min = param['min_value']
            self.epsilon_decay = param['decay']

    def reset(self):
        """

        :return:
        """
        self.step_size = self.initial_step_size
        self.w = self.starting_point
        self.k = 1
        return self.w 

    def step(self, probabilities, iteration, reward_type, max_iter):
        """

        :param action: the index to take a stochastic gradient step w.r.t
        :return: (observation, reward, done) tuple
        """

        # epsilon-greedy policy 
        p = np.random.rand()

        if p < self.epsilon:
            action = np.random.randint(self.n_summands)
            prob = 1 / self.n_summands
        else:
            action = np.random.choice(self.n_summands, p = probabilities)
            prob = probabilities[action]

        action = np.random.choice(self.n_summands, p=probabilities)

        # decay epsilon 
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        old_w = self.w

        grad = self.objective.stochastic_gradient(action, self.w)
        step = self.step_size / (prob * self.n_summands)

        self.w = self.w - step * grad
        
        self.decrease_step_size(iteration)

        observation = self.w 

        reward = self.calculate_reward(reward_type, iteration, max_iter, old_w, self.w)

        return (observation, reward, action, self.w)

    def calculate_reward(self, reward_type, iteration, max_iteration, old_w, new_w):
        if reward_type == 'function_value':
            return -1 * self.objective.evaluate(self.w) / self.starting_fn_value

        if reward_type == 'function_diff':
            return (self.objective.evaluate(new_w) -  self.objective.evaluate(old_w)) / self.starting_fn_value

        if reward_type == 'smoothing':
            reward = -1 * self.objective.evaluate(self.w) / self.starting_fn_value
            self.reward_window.append(reward)
            return np.mean(self.reward_window)

        if reward_type == 'normalizing_function':
            reward = -1 * self.objective.evaluate(self.w)
            return np.sign(reward) * (reward ** 2 / (reward ** 2 + 10))

    def clip_reward(self, reward):
        """
        1) reward = - evaluateAt(current_weights) / evaluateAt(starting_weights)
        2) reward = - evaluateAt(current_weights) - lowerBound / evaluateAt(starting_weights) - lowerBound
        3) fn(x) = sign(x) * \frac{x^2}{x^2 + C} el. fn(x) = sigmoid(x)
        4) smoothing m. ex deque
        return -1 if -1 * reward < -1 else -1 * reward
        """

        pass


    def initialize_step_size(self):
        XtX = self.X.T.dot(self.X)
        eig_vals, _ = np.linalg.eig(XtX)
        initial_step_size = 1. / max(eig_vals)
        return initial_step_size

    def decrease_step_size(self, iteration):
        if iteration > 0 and iteration % self.n_summands == 0:
            self.step_size = self.initial_step_size / self.k
            self.k += 1



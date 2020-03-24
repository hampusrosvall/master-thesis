from LeastSquaresProblem import LeastSquares
from StochasticGradientOpenAi import StochasticGradientEnvironment
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt
from tqdm import tqdm


# initialize constants
REPLAY_MEMORY_SIZE = 500
GAMMA = 0.9
BATCH_SIZE = 16
MIN_BUFFER_SIZE = BATCH_SIZE
EPISODES = 100
UPDATE_TARGET_EVERY = 50
N_ITERATIONS = 2000
INPUT_SHAPE = (1,)
MAX_ITER_PER_EPISODE = 200

BATCH_LOOK_UP = dict(zip(['state', 'action', 'reward', 'successor_state'], range(0, 4)))



class SGDDQNAgent:
    def __init__(self):
        # initialize objective function and stochastic gradient descent API
        self.objective = self._init_objective()
        self.env = StochasticGradientEnvironment(self.objective)

        # initialize parameters
        self.X, self.Y = self.objective.get_param()
        self.action_space_dim = self.X.shape[0]

        # initialize Q-value approximators
        self.model = self.get_model()
        self.target_model = self.get_model()
        self.target_model.set_weights(self.model.get_weights())

        # initialize memory replay
        self.memory_buffer = deque(maxlen = REPLAY_MEMORY_SIZE)

        self.target_update_counter = 0

        self.optimal_w = self.analytical_solution()

    def analytical_solution(self):
        A, b = self.objective.get_param()
        pseudoinv = np.linalg.inv(np.matmul(A.T, A))
        pseudoinv = np.matmul(pseudoinv, A.T)
        w_star = np.dot(pseudoinv, b)
        return w_star

    def _init_objective(self):
        m, n = 100, 2

        A = np.random.rand(m, n)
        b = np.random.rand(m)
        return LeastSquares(A, b)

    def get_model(self):
        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape = INPUT_SHAPE))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_space_dim, activation='linear'))
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])

        return model

    def epsilon_greedy_policy(self, state, epsilon=0):
        """

        :param state:
        :param epsilon:
        :return:
        """
        if np.random.rand() < epsilon:
            return np.random_choice(self.X.shape[0])
        else:
            return np.argmax(self.model.predict(np.array([state])))

    def train(self):
        # only train when we have enough examples
        if len(self.memory_buffer) < MIN_BUFFER_SIZE:
            return

        # sample mini-batch
        mini_batch = random.sample(self.memory_buffer, BATCH_SIZE)

        # extract the states
        states = np.array(
            [example[BATCH_LOOK_UP['state']]
                  for example in mini_batch]
        )

        # predict the Q-values for the states
        Q_values = self.model.predict(states)

        # extract successor states
        successor_states = np.array(
            [example[BATCH_LOOK_UP['successor_state']]
             for example in mini_batch]
        )

        # predict Q-values for the successor states
        successor_Q_values = self.target_model.predict(successor_states)

        X = []
        y = []

        # build input and response tensors for network training
        for index, (state, action, reward, successor_state) in enumerate(mini_batch):
            target_Q = reward + GAMMA * np.max(successor_Q_values[index])

            # update Q-value for current (state, action)-pair
            current_Q = Q_values[index]
            current_Q[action] = target_Q

            X.append(state)
            y.append(current_Q)

        self.model.fit(np.array(X), np.array(y), batch_size = BATCH_SIZE, verbose = 0, shuffle = False)
        self.target_update_counter += 1

        # update target model every: UPDATE_TARGET_EVERY iterations
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def softmax(self, Q):
        q_exp = np.exp(Q)
        return q_exp / np.sum(q_exp)

    def train_for_N_episodes(self, iterations = N_ITERATIONS):
        plt.figure()
        state = self.env.reset()
        distance_to_w = []
        weights = []
        for iteration in tqdm(range(iterations)):
            # extract Q_values
            Q_values = self.model.predict(np.array([state]))

            # apply softmax function to achieve probability distribution
            probabilities = np.squeeze(self.softmax(Q_values))

            # perform gradient step
            successor_state, reward, action, w = self.env.step(probabilities, iteration)

            weights.append(w)
            distance_to_w.append(np.linalg.norm(w - self.optimal_w))

            # append experience to memory buffer
            self.memory_buffer.append((state, action, reward, successor_state))

            # train network
            self.train()

            state = successor_state

            return weights

        plt.xlabel('Iteration: #')
        plt.ylabel('$w$ - $w_{opt}$')
        plt.plot(range(len(distance_to_w)), distance_to_w)
        plt.show()

if __name__ == '__main__':
    agent = SGDDQNAgent()

    agent.train_for_N_episodes()

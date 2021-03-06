from objectives.LeastSquaresProblem import LeastSquares
from objectives.StochasticGradientOpenAi import StochasticGradientEnvironment
from StochasticGradientDescent import StochasticGradientDescent
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import OrderedDict
import sys
import json
from datetime import datetime
import os

BATCH_LOOK_UP = dict(zip(['state', 'action', 'reward', 'successor_state'], range(0, 4)))

np.random.seed(123)
np.seterr(all = 'warn')
INPUT_SHAPE = (1,)

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
        
        # initialize target network update counter 
        self.target_update_counter = 0

        # calculate analytical solution 
        self.optimal_w = self.analytical_solution()

        # initialize hyper-parameters
        self.initialize_hyper_parameters()

        # initialize memory replay
        self.memory_buffer = deque(maxlen=self.replay_memory_size)

    def initialize_hyper_parameters(self):
        if len(sys.argv) == 1:
            file_name = './standard_paramters.json'
        else:
            file_name = sys.argv[1]

        with open(file_name, 'r') as f:
            data_input = json.load(f)
            param = data_input['parameters']
            self.replay_memory_size = param['replay_memory_size']
            self.gamma = param['gamma']
            self.batch_size = param['batch_size']
            self.min_buffer_size = param['minimum_buffer_size']
            self.episodes = param['n_episodes']
            self.update_target = param['update_target']
            self.iterations = param['n_iterations']


    def analytical_solution(self):
        A, b = self.objective.get_param()
        pseudoinv = np.linalg.inv(np.matmul(A.T, A))
        pseudoinv = np.matmul(pseudoinv, A.T)
        w_star = np.dot(pseudoinv, b)
        return w_star

    def _init_objective(self):
        m, n = 100, 2

        A = np.random.rand(m, n)
        A[1:50, :] = A[1:50, :] * 10
        b = np.random.rand(m)
        return LeastSquares(A, b)

    def get_model(self):
        if len(sys.argv) == 1:
            file_name = './standard_paramters.json'
        else:
            file_name = sys.argv[1]

        with open(file_name, 'r') as f:
            data_input = json.load(f)
            network_param = data_input['network_architecture']
            layers = network_param['nbr_nodes']

        model = Sequential()
        model.add(Dense(32, activation='relu', input_shape = INPUT_SHAPE))

        for lr in layers: 
            model.add(Dense(lr, activation = 'relu'))

        model.add(Dense(self.action_space_dim, activation='linear'))
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])

        return model

    def train(self):
        # sample mini-batch
        mini_batch = random.sample(self.memory_buffer, self.batch_size)

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
            target_Q = reward + self.gamma * np.max(successor_Q_values[index])

            # update Q-value for current (state, action)-pair
            current_Q = Q_values[index]
            current_Q[action] = target_Q

            X.append(state)
            y.append(current_Q)

        self.model.fit(np.array(X), np.array(y), batch_size = self.batch_size, verbose = 0, shuffle = False)
        self.target_update_counter += 1

        # update target model every: UPDATE_TARGET_EVERY iterations
        if self.target_update_counter > self.update_target:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def softmax(self, Q):
        if not np.isfinite(Q).all():
            print('failed')
        q_exp = np.exp(Q)
        prob = q_exp / np.sum(q_exp)

        if not np.isfinite(q_exp).all() or not np.isfinite(prob).all() or not np.isfinite(Q).all():
            print('failed')

        return prob

    def train_for_N_episodes(self):
        # initialize hash tables to store information about training
        actions = OrderedDict()
        paths = OrderedDict()

        for episode in range(self.episodes):
            distance_to_w = []
            a = []
            state, starting_point = self.env.reset()
            for iteration in tqdm(range(self.iterations)):
                # extract Q_values
                Q_values = self.model.predict(np.array([state]))

                # apply softmax function to achieve probability distribution
                probabilities = np.squeeze(self.softmax(Q_values))

                # perform gradient step
                successor_state, reward, action, w, step = self.env.step(probabilities, iteration)

                # store information for visualization purposes
                a.append(action)
                distance_to_w.append(np.linalg.norm(w - self.optimal_w))

                # append experience to memory buffer
                self.memory_buffer.append((state, action, reward, successor_state))

                # only train when we have enough examples
                if len(self.memory_buffer) >= self.min_buffer_size:
                    self.train()

                state = successor_state

            actions[episode] = a
            paths[episode] = distance_to_w

        return actions, paths

    def train_session(self):
        # initialize directory to save data for the runs
        dt = str(datetime.now())
        date = dt.split()[0]
        time = dt.split()[1].split('.')[0].replace(':', '-')
        folder_name = date + '-' + time

        fpath = os.path.join('./experiments', folder_name)
        os.mkdir(fpath)

        if len(sys.argv) > 1:
            f_name = sys.argv[1]
            with open(f_name) as f:
                new_file = os.path.join(fpath, 'parameters.json')
                content = json.load(f)
                content = json.dumps(content, indent=4, sort_keys=True)
                file = open(new_file, 'w+')
                file.write(content)
            file.close()

        actions, paths = self.train_for_N_episodes()

        plt.figure()
        for episode, path in paths.items():
            plt.xlabel('Iteration: #')
            plt.ylabel('$w$ - $w_{opt}$')
            plt.plot(range(len(path)), path, label=f'episode: {episode}')
            plt.title('Convergence to optimal solution during training')

        plt.legend(loc = 'best')
        plt.savefig(os.path.join(fpath, 'episode-convergence.png'))

        for episode, action in actions.items():
            plt.figure()
            plt.hist(action, bins=100)
            plt.title(f'Actions during episode: {episode}')
            plt.savefig(os.path.join(fpath, 'actions-episode-{}.png'.format(episode)))

        #starting_point = agent.env.starting_point
        #objective = agent.env.objective
        #optimal_w = agent.optimal_w
        #sgd = StochasticGradientDescent(starting_point)

        #param = sgd.optimize(objective, n_iter=int(2000 / 100), analytical_sol=optimal_w)
        #distance_to_w_sgd = param[-2]
        #plt.figure()
        #plt.xlabel('Iteration: #')
        #plt.ylabel('$w$ - $w_{opt}$')
        #plt.plot(range(len(distance_to_w_sgd)), distance_to_w_sgd)
        #plt.title('Convergence to optimal solution SGD uniform sampling')
        #plt.savefig(os.path.join(fpath, 'SGD-convergence.png'))

if __name__ == '__main__':
    agent = SGDDQNAgent()
    agent.train_session()
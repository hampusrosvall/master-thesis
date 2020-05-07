from objectives.LeastSquaresProblem import LeastSquares
from objectives.sgdopenai import StochasticGradientEnvironment
from StochasticGradientDescent import StochasticGradientDescent
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import softmax
from tensorflow import convert_to_tensor
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

class SGDDQNAgent:
    def __init__(self):
        # initialize objective function
        self.objective = self.init_objective()

        # cache objective parameters 
        self.X, self.Y = self.objective.get_param()

        # cache data matrix dimensions 
        self.action_space_dim, self.input_shape = self.X.shape 

        # initialize Q-value function approximator
        self.model = self.get_model()

        # initialize Q-target network 
        self.target_model = self.get_model()

        # set the weights of the target network in according to the function approximator network 
        self.target_model.set_weights(self.model.get_weights())
        
        # initialize target network update counter 
        self.target_update_counter = 0

        # calculate analytical solution 
        self.optimal_w = self.analytical_solution()

        # initialize hyper-parameters
        self.initialize_parameters()

        # initialize reinforcement learning environment for the optimization problem 
        self.env = StochasticGradientEnvironment(self.objective, self.reward_type)

        # initialize memory replay
        self.memory_buffer = deque(maxlen=self.replay_memory_size)

    def initialize_parameters(self):
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
            self.reward_type = data_input['reward']

    def analytical_solution(self):
        A, b = self.objective.get_param()
        pseudoinv = np.linalg.inv(np.matmul(A.T, A))
        pseudoinv = np.matmul(pseudoinv, A.T)
        w_star = np.dot(pseudoinv, b)
        return w_star

    def init_objective(self):
        if len(sys.argv) == 1:
            file_name = './standard_paramters.json'
        else:
            file_name = sys.argv[1]

        with open(file_name, 'r') as f:
            data_input = json.load(f)
            problem_param = data_input['problem_info']
            m, n = problem_param["n_rows"], problem_param["n_cols"]
            scale_lipschitz = problem_param["scale_lipschitz"]

        A = np.random.rand(m, n)
        if scale_lipschitz["should_scale"]:
            rows_to_scale = m // 2
            A[:rows_to_scale, :] = A[:rows_to_scale, :] * scale_lipschitz["factor"]
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
        model.add(Dense(32, activation='relu', input_shape = (self.input_shape, )))

        for lr in layers: 
            model.add(Dense(lr, activation = 'relu'))

        model.add(Dense(self.action_space_dim, activation='linear'))
        model.compile(loss="mse", optimizer=Adam(lr=0.01), metrics=['accuracy'])

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

    def train_for_N_episodes(self):
        # initialize hash tables to store information about training
        actions = OrderedDict()
        f_val = []
        distance_to_w = []
        rewards = []
        for episode in range(self.episodes):
            a = []
            r = 0
            state = self.env.reset()
            for iteration in tqdm(range(self.iterations)):
                # extract Q_values
                Q_values = self.model.predict(np.array([state]))

                # apply softmax function to achieve probability distribution
                probabilities = np.array(softmax(convert_to_tensor(Q_values))[0])
                
                # perform gradient step
                successor_state, reward, action, w = self.env.step(probabilities, iteration,
                                                                         self.reward_type, self.iterations)

                r += reward

                # store information for visualization purposes
                a.append(action)

                if iteration == self.iterations - 1:
                    distance_to_w.append(np.linalg.norm(w - self.optimal_w))
                    f_val.append(self.env.objective.evaluate(w))
                    rewards.append(r)

                # append experience to memory buffer
                self.memory_buffer.append((state, action, reward, successor_state))

                # only train when we have enough examples
                if len(self.memory_buffer) >= self.min_buffer_size:
                    self.train()

                state = successor_state

            actions[episode] = a

        data = {
            "distance_to_w" : distance_to_w,
            "f_val" : f_val,
            "rewards": rewards,
            "actions": actions
        }

        return data

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

        data = self.train_for_N_episodes()
        data_path = os.path.join(fpath, 'data.json')
        with open(data_path, 'w') as f:
            json.dump(data, f)

if __name__ == '__main__':
    agent = SGDDQNAgent()
    agent.train_session()
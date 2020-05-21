# Introduction 
__Apparently, LaTeX is not supported by git markdown, hence, the readme looks messy at some points..__

This my implementation of a deep Q-network that solves optimization problems using stochastic gradient descent to solve finite sum problems. The network architecture is based on the work done by DeepMind [1]. Using vanillva stochastic gradient descent, the sampling strategy is explicitlly defined by a probability distribution over the summand indicies $I = \{0, ... , N - 1\}$. Popular sampling strategies are uniform and Lipschitz sampling [2]. 

# Implemenatation 
The agent-environment interaction is based on the work done by OpenAi [3]. The environment is defined under `objectives/sgdopenai.py` and takes in a objective function as input. The objective function is e.g., a least squares minimization problem that supports function evaluation and stochastic gradient evaluation (see `objectives/LeastSquaresProblem.py`). 

The training occurs in `sgddqn.py` where the DQN is fed a snapshot of the trainable paramters $w$ given a batch sample from the experience replay and updates the parameters with respect to to the loss function: 


$L_k = \Big(Q(s, a; \theta_k) - y_{target} \Big)^2$ where $y_{target} = R_{t + 1} + \gamma \underset{a'}{\text{max}} Q(S_{t + 1}, a'; \theta_k)$. 

The network generating $y_{target}$ is updated every $C \in \mathbb{N}$ iterations. 

## Defining actions, states and rewards. 
### Actions 
An action refers to which index $i$ with respect to which the stochastic gradient approximation is calculated. 

### States 
The states are a snapshot of the learnable parameters. 

### Rewards 
Rewards are based on two different functions: 

* `functionDiffReward` = $R(s^i_j) = \frac{f(s^i_j) - f(s^i_{j - 1})}{f(s_0)}$ where $s^i_j$ is the state at iteration $i$ during episode $j$ and $s_0$ is the initial state. 

* `functionValueReward` = $R(s^i_j) = - \frac{f(s^i_j)}{f(s_0)}$


## Setting the hyperparameters 
The hyperparameters are set in `custom_parameters.json`. 

## Running the program locally 
The easiest way to run the program locally is to first set up a new conda environment by `conda create -n myEnv python=3.7` and install the dependencies by `pip install -r requirements.txt`. 

Make sure you are in the top level folder and run `conda activate myEnv` followed by `python sgddqn.py custom_parameters.json`. If the parameters argument is not specified the parameters in `standard_paramters.json` will be used. 

Lastly, the result of one training instance will be stored locally under `./expermiments/<timestamp>`. The hyperparameters along with a JSON data object will be stored. Currently, the data stored is: 

* $w - w^*$ where $w^*$ is the global minumum (given a convex objective) for all episodes ($w$ in this case refers to a snapshot of the learnable paramters in the last iteration given an episode). 
* $f(w)$ $\forall \text{episodes}$ at last iteration 

* The actions performed for each episode 

* The accumulated rewards 

# References 
[1] - https://deepmind.com/research/publications/human-level-control-through-deep-reinforcement-learning

[2] - http://papers.nips.cc/paper/5355-stochastic-gradient-descent-weighted-sampling-and-the-randomized-kaczmarz-algorithm.pdf

[3] - https://gym.openai.com/
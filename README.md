# Introduction 
This my implementation of a deep Q-network that solves optimization problems using stochastic gradient descent to solve finite sum problems. The network architecture is based on the work done by DeepMind [1]. Using vanillva stochastic gradient descent, the sampling strategy is explicitlly defined by a probability distribution over the summand indicies $I = \{0, ... , N - 1\}$. Popular sampling strategies are uniform and Lipschitz sampling [2]. 

# Implemenatation 
The agent-environment interaction is based on the work done by OpenAi [3]. The environment is defined under objectives/sgdopenai.py and takes in a objective function as input. The objective function is e.g., a least squares minimization problem that supports function evaluation and stochastic gradient evaluation (see objectives/LeastSquaresProblem.py). 


# References 
[1] - https://deepmind.com/research/publications/human-level-control-through-deep-reinforcement-learning

[2] - http://papers.nips.cc/paper/5355-stochastic-gradient-descent-weighted-sampling-and-the-randomized-kaczmarz-algorithm.pdf

[3] - https://gym.openai.com/
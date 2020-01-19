# Solving the Multi-Armed Bandit Problem on a Photonic Quantum Computer
Research project focused on developing a quantum neural network to solve the non-stationary multi-armed bandit problem.

## Abstract
Machine learning is the study of how to teach computers (learning agents) to learn and gain
insight from either a dataset or a given environment. There are three common types of
machine learning: supervised, unsupervised, and reinforcement learning. In this paper we
focus primarily on reinforcement learning. Reinforcement learning is where a computer is
tasked with maximizing reward by interacting with an environment. The computer,
otherwise known as the agent, learns to map situations to actions it should take. In this
paper, we implement an agent through the use of a quantum neural network (QNN) to solve
the multi-armed bandit problem. The quantum neural network is based on the continuous
variable (CV) model of quantum computing. In the stochastic multi-armed bandit problem,
the environment consists of a 𝑘-armed bandit, in which each arm of the bandit gives a
reward for pulling it, described by a reward distribution. The agent’s goal is to learn which
arm provides the most reward, through interacting with all the various arms on the bandit.
We made the problem more difficult, by shuffling the rewards for each of the arms partway
through the learning process, so that we might observe the effectiveness of the QNN.
Our results show that, despite adding noise to the reward distribution and shuffling the
rewards for each arm, our QNN successfully distinguishes the most rewarding arm on the
bandit from the rest of the arms.

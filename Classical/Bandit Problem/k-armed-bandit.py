import numpy as np
import matplotlib.pyplot as plt
import random

def generate_problem(k):
    return np.random.normal(loc=0.0, scale=1, size=10)

def generate_reward(problem, action):
    return np.random.normal(loc=problem[action], scale=1)

def k_bandit(problem, k, steps, exploration_rate):
    Q = {i: 0 for i in range(k)}                            # 1. Value function
    N = {i: 0 for i in range(k)}                            # 2. Number of actions, for update rule
    rewards = []
    rewards_mean = []
    for i in range(steps):                                  # 3. Main loop
        explore = random.uniform(0, 1) < exploration_rate 
        if explore:
            action = random.randint(0, k - 1)               # 5. Exploration: Choosing random action
        else:
            action = max(Q, key=Q.get)                      # 6. Choose action with maximum mean reward
            
        reward = generate_reward(problem, action)           # 7. Get reward for current action
        N[action] += 1 # 8. Update action number
        Q[action] += (1 / N[action]) * (reward - Q[action]) # 9. Update value dict
        rewards.append(reward)
        if (i % 100 == 0):
            r_mean = np.mean(rewards[-100:])
            rewards_mean.append(r_mean)
    return rewards_mean

k = 4
steps = 10000
problem = generate_problem(k)
r_means = k_bandit(problem, k, steps, 0.01)
plt.plot(r_means)
plt.show()
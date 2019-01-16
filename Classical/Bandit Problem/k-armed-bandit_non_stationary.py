import numpy as np
import matplotlib.pyplot as plt
import random
import sys

class KBanditProblem:
    
    def __init__(self, k, stationary=True):
        self.k = k
        self.stationary = stationary
        self.values = np.random.normal(loc=0.0, scale=1, size=k)
        self.optimal = self.values.argmax() # this is called optimal becuase the things are random, and becuase it changes
                                            # over time, and every time a reqward is given, the distribution of rewards chnages 
                                            # with the random reward. The optimal solution changes over time, thus it has to be 
                                            # recalculated every time.
        
    def generate_reward(self, action):
        if not self.stationary:
            self.values += np.random.normal(loc=0.0, scale=0.01, size=self.k)
            self.optimal = self.values.argmax()
        return np.random.normal(loc=self.values[action], scale=1)

class KBanditSolution:
    
    def __init__(self, problem, steps):
        self.problem = problem
        self.steps = steps
        
        self.average_reward = 0
        self.average_rewards = np.array([])
        self.optimal_percentage = 0
        self.optimal_precentages = np.array([])
        
    def count_statistics(self, action, reward, step):
        self.average_reward += (1 / (step + 1)) * (reward - self.average_reward)
        self.optimal_percentage += (1 / (step + 1)) * ((1 if action == self.problem.optimal else 0) - self.optimal_percentage)
        self.average_rewards = np.append(self.average_rewards, self.average_reward)
        self.optimal_precentages = np.append(self.optimal_precentages, self.optimal_percentage)

class EGreedy(KBanditSolution):
    
    def solve(self, exploration_rate, initial_value):
        Q = {i: initial_value for i in range(k)} # 1. Value function
        N = {i: 0 for i in range(k)} # 2. Number of actions, for update rule
        rewards = []
        rewards_mean = []
        for i in range(self.steps): # 3. Main loop
            explore = random.uniform(0, 1) < exploration_rate  # 4. Exploration
            if explore:
                action = random.randint(0, k - 1) # 5. Exploration: Choosing random action
            else:
                action = max(Q, key=Q.get) # 6. Choose action with maximum mean reward

            reward = self.problem.generate_reward(action) # 7. Get reward for current action
            rewards.append(reward)
            N[action] += 1 # 8. Update action number
            Q[action] += (1 / N[action]) * (reward - Q[action]) # 9. Update value dict 
            if (i % 100 == 0):
                r_mean = np.mean(rewards[-100:])
                rewards_mean.append(r_mean)
            self.count_statistics(action, reward, i)
        return rewards_mean

    
    def plot_graph(self, values):
        plt.plot(values)
        plt.show()

class WeightedAverage(KBanditSolution):
    
    def solve(self, exploration_rate, step_size, initial_value):
        Q = {i: initial_value for i in range(k)} # 1. Value function
        N = {i: 0 for i in range(k)} # 2. Number of actions, for update rule

        for i in range(self.steps): # 3. Main loop
            explore = random.uniform(0, 1) < exploration_rate  # 4. Exploration
            if explore:
                action = random.randint(0, k - 1) # 5. Exploration: Choosing random action
            else:
                action = max(Q, key=Q.get) # 6. Choose action with maximum mean reward

            reward = self.problem.generate_reward(action) # 7. Get reward for current action
            N[action] += 1 # 8. Update action number
            Q[action] += step_size * (reward - Q[action]) # 9. Update value dict 
            self.count_statistics(action, reward, i)

class UCB(KBanditSolution):
    
    def count_ucb(self, q, c, step, n):
        if n == 0:
            return sys.maxsize
        return (q + (c * sqrt((log(step) / n))))
    
    def solve(self, c):
        Q = {i: 0 for i in range(k)} # 1. Value function        
        N = {i: 0 for i in range(k)} # 2. Number of actions, for update rule

        for i in range(self.steps): # 3. Main loop
            Q_ucb = {i: self.count_ucb(Q[i], c, i + 1, N[i]) for i in range(k)} # 4. Count UCB
            action = max(Q_ucb, key=Q_ucb.get) # 5. Choose action with maximum UCB

            reward = self.problem.generate_reward(action) # 6. Get reward for current action
            N[action] += 1 # 7. Update action number
            Q[action] += (1 / N[action]) * (reward - Q[action]) # 8. Update value dict 
            self.count_statistics(action, reward, i)
    
k = 4
steps = 50000
kb_problem = KBanditProblem(k, stationary=False)
#kb_solution = KBanditSolution(kb_problem, steps)
egreedy_boi = EGreedy(kb_problem, steps)
solved = egreedy_boi.solve(0.01, 0)
egreedy_boi.plot_graph(solved)
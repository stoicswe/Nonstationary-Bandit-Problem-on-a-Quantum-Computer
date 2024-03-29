import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import strawberryfields as sf
import operator
import warnings
from strawberryfields.ops import *

warnings.filterwarnings("ignore")

'''
    Author: Nathan Bunch
    Date: November 30, 2018
    Decription: Quantum Neural Network for solving
                the non-stationary bandit problem.
    
    Description (Long):
    This project is research that has been conducted by 
    Nathan Bunch in the field of Quantum Artificial
    Intelligence, with the primary goal in mind to solve
    the non stationary bandit problem using a quantum
    neural network, based on the Strawberry Fields 
    quantum computing API.

    This project has been conducted for reasearch under
    the Houghton College Computer Science Honors program.

    Classes and/or Methods:
    This project contains one method which generates the
    stochastic reward. This reward is based on a baseline
    (known as "initial_values") and then fluctuates that 
    baseline distribution by adding a random value to the 
    reward chosen, given the action. The higher the fluctuation
    on the reward distribution, the harder it is for the
    neural network to learn which bandit generates the most
    reward.

    Running the project:
    There are no parameters to be placed into the program in
    order for it to run, so just run it with the following.

    "python ./non_stationary_bandits_final.py"
'''
print("Initializing Variables for the Quantum Neural Network")
# Number of QModes/Neurons
eng, q = sf.Engine(4)
# Parameters for adjusting
# the entanglement of the
# QModes.
E0 = tf.Variable(0.1)
E1 = tf.Variable(0.1)
E2 = tf.Variable(0.1)
E3 = tf.Variable(0.1)
E4 = tf.Variable(0.1)
E5 = tf.Variable(0.1)
# Parameters for finetuning
# the neural network output
S0 = tf.Variable(0.1)
S1 = tf.Variable(0.1)
S2 = tf.Variable(0.1)
S3 = tf.Variable(0.1)
D0 = tf.Variable(0.1)
D1 = tf.Variable(0.1)
D2 = tf.Variable(0.1)
D3 = tf.Variable(0.1)
P0 = tf.Variable(0.1)
P1 = tf.Variable(0.1)
P2 = tf.Variable(0.1)
P3 = tf.Variable(0.1)
# Initialize the parameter for input
X = tf.placeholder(tf.float32, [1])
print("Building the Quantum Circuit")
with eng:
    # initialize the variables to learn
    Dgate(X[0], 0.) | q[0]
    Dgate(X[0], 0.) | q[1]
    Dgate(X[0], 0.) | q[2]
    Dgate(X[0], 0.) | q[3]

    # setup the entaglement
    BSgate(phi=E0) | (q[0], q[1])
    BSgate() | (q[0], q[1])

    BSgate(phi=E1) | (q[0], q[2])
    BSgate() | (q[0], q[2])

    BSgate(phi=E2) | (q[0], q[3])
    BSgate() | (q[0], q[3])

    BSgate(phi=E3) | (q[1], q[2])
    BSgate() | (q[1], q[2])

    BSgate(phi=E4) | (q[1], q[3])
    BSgate() | (q[1], q[3])

    BSgate(phi=E5) | (q[2], q[3])
    BSgate() | (q[2], q[3])

    # fine tune the results
    # Squeeze Gates
    Sgate(S0) | q[0]
    Sgate(S1) | q[1]
    Sgate(S2) | q[2]
    Sgate(S3) | q[3]
    # Displacement Gates
    Dgate(D0) | q[0]
    Dgate(D1) | q[1]
    Dgate(D2) | q[2]
    Dgate(D3) | q[3]
    # Phase Gates
    Pgate(P0) | q[0]
    Pgate(P1) | q[1]
    Pgate(P2) | q[2]
    Pgate(P3) | q[3]
print("Initializing Quantum to Classical Conversions")
state = eng.run('tf', cutoff_dim=10, eval=False)
# pulling results from the QNN
prob0 = state.fock_prob([2, 0, 0, 0])
prob1 = state.fock_prob([0, 2, 0, 0])
prob2 = state.fock_prob([0, 0, 2, 0])
prob3 = state.fock_prob([0, 0, 0, 2])
# find the normalization factor
normalization = prob0 + prob1 + prob2 + prob3 + 1e-10
# output the weights calulated by the qnn
circuit_output = [prob0 / normalization, prob1 / normalization, prob2 / normalization, prob3 / normalization]

def generate_reward(reward_distribution, action, k):
    # Visualize: 
    # a list of distributions of awards
    # and the probability of getting the awards
    # initializing first, then adding will give a basis for how to reward should operate
    # changing the scale values will make it harder or easier depending on the larger the scale
    # by changing the scale gradually, it makes the learning harder by
    # fluctuating the reward more (causing it to be more random, and to have a larger range)
    new_reward_distribution = reward_distribution + np.random.normal(loc=0.0, scale=0.01, size=k)
    optimal = np.argmax(new_reward_distribution)
    return np.random.normal(loc=new_reward_distribution[action], scale=1), optimal, new_reward_distribution

# Parameters for the project configuration
reward_distribution = [0.6, 0, 0.1, 0.3]    # This is the reward distribution for the bandits
num_bandits = len(reward_distribution)      # The number of bandits must be same as reward distribution
weights = circuit_output                    # Weights are calculated by the quantum neural network
chosen_action = tf.argmax(weights)          # Choose an action, based on the weights
total_episodes = 10000                      # Number of iterations
total_reward = np.zeros(num_bandits)        # Total rewards for each of the bandits
random_action_factor = 0.1                  # The % of the time we randomly choose an action, not based on weights

print("Setting up the Network...")
print("Reward holder")
reward_holder = tf.placeholder(shape=[1],dtype=tf.float32)
print("Action holder")
action_holder = tf.placeholder(shape=[1],dtype=tf.int32)
print("Responsible Weight")
responsible_weight = tf.slice(weights,action_holder,[1])
print("Loss Function")
loss = -(tf.log(responsible_weight)*reward_holder)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
update = optimizer.minimize(loss)

init = tf.initialize_all_variables()
# These lists are stored for graphing the rewards over time for each bandit
rewards0 = []
rewards1 = []
rewards2 = []
rewards3 = []
accuracy = []
accuracy_scores = []
sess = tf.Session()
sess.run(init)
temp_action = 0 # In order to generate the weights from the QNN for the first iteration, this has to be set
print("Training the network")
i = 0
while i < total_episodes:
    if np.random.rand(1) < random_action_factor:
        action = np.random.randint(num_bandits)
    else:
        action = sess.run(chosen_action, feed_dict={X : [temp_action]})
        
    #use optimal to use to measure the accuracy for each iteration, also to show that it is learning
    reward, optimal, _ = generate_reward(reward_distribution, action, num_bandits)
    _, resp, ww = sess.run([update, responsible_weight, weights], feed_dict={X: [action], reward_holder:[reward],action_holder:[action]})
    total_reward[action] += reward
    # measure the accuracy of the network
    if np.argmax(ww) == optimal:
        accuracy.append(1)
    else:
        accuracy.append(0)
    
    # store the accuracy scores for later
    if i % 50 == 0:
        accuracy_scores.append(np.mean(accuracy))
    
    # every 100 iterations, print to the user
    if i % 100 == 0:
        print("Running reward for the " + str(num_bandits) + " bandits: " + str(total_reward))
        print("Iteration: {0} Accuracy: {1}".format(i, round(np.mean(accuracy), 4)))
    rewards0.append(total_reward[0])
    rewards1.append(total_reward[1])
    rewards2.append(total_reward[2])
    rewards3.append(total_reward[3])
    i+=1
    temp_action = action
print("Accuracy for this network: {0}".format(np.mean(accuracy[int(total_episodes/2)])))
print( "The agent thinks bandit " + str(np.argmax(ww)+1) + " is the most promising....")
if np.argmax(ww) == np.argmax(np.array(reward_distribution)):
    print( "...and it was right!")
else:
    print( "...and it was wrong!")
# graph the rewards over time, so we can see the stochastic learning
plt.plot(rewards0)
plt.show()
plt.plot(rewards1)
plt.show()
plt.plot(rewards2)
plt.show()
plt.plot(rewards3)
plt.show()
plt.plot(accuracy_scores)
plt.show()
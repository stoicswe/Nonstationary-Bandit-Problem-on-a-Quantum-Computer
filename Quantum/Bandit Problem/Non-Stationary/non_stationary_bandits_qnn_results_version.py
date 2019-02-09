import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import strawberryfields as sf
import operator
import warnings
import random as rand
import copy
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

# Fair warning: this program takes a LONG time to run on a classical
# machine. You may want to either sleep or take a long coffee break
# while this program is running.

# Parameters for the project configuration
reward_distribution_original = [0.0, 0.5, 0.3, 0.2]                         # These numbers are chosen to give the arms contrast, but also to assist kin debugging.
reward_distribution = copy.deepcopy(reward_distribution_original)           # This is the reward distribution for the bandits
reward_distribution_shuffled = [0.3, 0.2, 0.0, 0.5]                         # This is a shuffled one we agreed upon for the results section of the thesis
num_bandits = len(reward_distribution)                                      # The number of bandits must be same as reward distribution
total_episodes = 50000                                                      # Number of iterations
learning_rate = 0.01                                                        # learning rate of the GD algorithm
swap_dist_test = int(total_episodes/10)                                     # partway through learning, swap two distributions and examine the change
random_action_factor = 0.10                                                 # The % of the time we randomly choose an action, not based on weights
accuracy_update = 50                                                        # every 100 iterations, reccord the accuracy for past 100 iterations
print_update = 200                                                          # every 200 iterations, output to the user
    
save_local = './Results/'
save_local_graphs = './Results/Graphs/'

# this is for the parameter printout to a file for analysis
qnn_parameters = [
    "Reward Distribution: {0}".format(reward_distribution_original), 
    "Shuffled Reward Distribution: {0}".format(reward_distribution_shuffled),
    "Number of Bandits: {0}".format(num_bandits), 
    "Total Iterations per Test: {0}".format(total_episodes), 
    "Learning Rate: {0}".format(learning_rate), 
    "Swap Distribution Iteration: {0}".format(swap_dist_test), 
    "Random Choice Factor: {0}".format(random_action_factor), 
    "Update Accuracy Score: {0}".format(accuracy_update), 
    "Print Update to User: {0}".format(print_update)
]
# print out the parameters
with open(save_local + 'PARAMETERS.txt', 'w') as f:
    for item in qnn_parameters:
        f.write("%s\n" % item)

interator_array = [0,3,6,7,8,9]

for interAtor in interator_array:
    for minorIterate in range(10):
        testnum = interAtor * 10
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
        # cubic phase gate stuff
        V0 = tf.Variable(0.1)
        V1 = tf.Variable(0.1)
        V2 = tf.Variable(0.1)
        V3 = tf.Variable(0.1)
        # Initialize the parameter for input
        X = tf.placeholder(tf.float32, [1])
        print("Building the Quantum Circuit")
        with eng:
            # initialize the variables to learn
            # phi o D o U2 o S o U1
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

            # fine tune the results of the quantum neural network
            # Squeeze Gates
            Sgate(S0) | q[0]
            Sgate(S1) | q[1]
            Sgate(S2) | q[2]
            Sgate(S3) | q[3]
            # Cubic phase gates
            Vgate(V0) | q[0]
            Vgate(V1) | q[1]
            Vgate(V2) | q[2]
            Vgate(V3) | q[3]
        print("Initializing Quantum to Classical Conversions")
        state = eng.run('tf', cutoff_dim=10, eval=False)
        # pulling results from the QNN in the Fock state basis
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
            new_reward_distribution = reward_distribution + np.random.normal(loc=0.0, scale=0.1*testnum, size=k)
            optimal = np.argmax(new_reward_distribution)
            return np.random.normal(loc=new_reward_distribution[action], scale=1), optimal, new_reward_distribution
            #return new_reward_distribution[action], optimal, new_reward_distribution
        
        # additional parameters that need to be reset every time the algorithm is run
        total_reward = np.zeros(num_bandits)                                        # Total rewards for each of the bandits
        weights = circuit_output                                                    # Weights are calculated by the quantum neural network
        chosen_action = tf.argmax(weights)                                          # Choose an action, based on the weights

        print("Setting up the Network...")
        print("Reward holder")
        reward_holder = tf.placeholder(shape=[1],dtype=tf.float32)
        print("Action holder")
        action_holder = tf.placeholder(shape=[1],dtype=tf.int32)
        print("Responsible Weight")
        responsible_weight = tf.slice(weights,action_holder,[1])
        print("Loss Function")
        loss = -(tf.log(responsible_weight)*reward_holder)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        update = optimizer.minimize(loss)

        init = tf.initialize_all_variables()
        # These lists are stored for graphing the rewards over time for each bandit
        rewards0 = []
        rewards1 = []
        rewards2 = []
        rewards3 = []
        accuracy = []
        accuracy_scores = []
        bandit_counts = [0,0,0,0]
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
            # reccord individual guesses for each bandit
            if best_guess == 0:
                bandit_counts[0] += 1
            if best_guess == 1:
                bandit_counts[1] += 1
            if best_guess == 2:
                bandit_counts[2] += 1
            if best_guess == 3:
                bandit_counts[3] += 1
            
            # when the network is 1part way through training, shuffle the reward distribution
            if i == swap_dist_test:
                #rand.shuffle(reward_distribution) # randomly shuffle the distributon
                # swap the distribution so we can test the network
                reward_distribution = []
                reward_distribution = reward_distribution_shuffled
            
            # every 100 iterations, print to the user
            if i % 200 == 0:
                print("---------------------------------------------------------------------------------------------")
                print( "Running reward for the " + str(num_bandits) + " bandits: " + str(total_reward))
                print("Iteration: {0}".format(i))
                print("Bandit Counts: | {0} | {1} | {2} | {3} |".format(bandit_counts[0], bandit_counts[1], bandit_counts[2], bandit_counts[3]))
                print("Weights: {0}".format(ww))

            # reccord the results of the accuracy, for analysis later
            rewards0.append(total_reward[0])
            rewards1.append(total_reward[1])
            rewards2.append(total_reward[2])
            rewards3.append(total_reward[3])
            i+=1
            temp_action = action

        # calculate the percentage of guesses made by the neural network
        b0_percent = round(bandit_counts[0] / total_episodes, 4)
        b1_percent = round(bandit_counts[1] / total_episodes, 4)
        b2_percent = round(bandit_counts[2] / total_episodes, 4)
        b3_percent = round(bandit_counts[3] / total_episodes, 4)
        # print the overall accuracy and make a prediction
        print("Final Weights: {0}".format(ww))
        print("Percentage of guesses during learning:")
        print("Bandit 1:{0} Bandit 2:{1} Bandit 3:{2} Bandit 4:{3}".format(b0_percent, b1_percent, b2_percent, b3_percent))
        print("The agent thinks bandit " + str(np.argmax(ww)+1) + " is the most promising....")
        if np.argmax(ww) == np.argmax(np.array(reward_distribution)):
            print( "...and it was right!")
        else:
            print( "...and it was wrong!")

        # reset the distribution for the network for more accurate readings
        reward_distribution = []
        reward_distribution = copy.deepcopy(reward_distribution_original)
        bandit_counts[0] = 0
        bandit_counts[1] = 0
        bandit_counts[2] = 0
        bandit_counts[3] = 0
        
        with open(save_local + str(testnum)+'bandit0_reward_'+str(minorIterate) +'.txt', 'w') as f:
            for item in rewards0:
                f.write("%s\n" % item)
        with open(save_local + str(testnum)+'bandit1_reward_'+str(minorIterate)  +'.txt', 'w') as f:
            for item in rewards1:
                f.write("%s\n" % item)
        with open(save_local + str(testnum)+'bandit2_reward_'+str(minorIterate)  +'.txt', 'w') as f:
            for item in rewards2:
                f.write("%s\n" % item)
        with open(save_local + str(testnum)+'bandit3_reward_'+str(minorIterate)  +'.txt', 'w') as f:
            for item in rewards3:
                f.write("%s\n" % item)
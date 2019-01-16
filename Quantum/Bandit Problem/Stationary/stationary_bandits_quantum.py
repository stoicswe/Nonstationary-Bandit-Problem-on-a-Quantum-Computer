import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import strawberryfields as sf
from strawberryfields.ops import *

#########################################################
#
# QUANTUM CIRCUIT
#
#########################################################

eng, q = sf.Engine(4)
# =========================
# beam spliting variables
alpha0 = tf.Variable(0.1)
alpha1 = tf.Variable(0.1)
alpha2 = tf.Variable(0.1)
alpha3 = tf.Variable(0.1)
alpha4 = tf.Variable(0.1)
alpha5 = tf.Variable(0.1)
# =========================

# =========================
# fine tuning the circuit
alpha6 = tf.Variable(0.1)
alpha7 = tf.Variable(0.1)
alpha8 = tf.Variable(0.1)
alpha9 = tf.Variable(0.1)
alpha10 = tf.Variable(0.1)
alpha11 = tf.Variable(0.1)
alpha12 = tf.Variable(0.1)
alpha13 = tf.Variable(0.1)
alpha14 = tf.Variable(0.1)
alpha15 = tf.Variable(0.1)
alpha16 = tf.Variable(0.1)
alpha17 = tf.Variable(0.1)
# =========================
X = tf.placeholder(tf.float32, [1])
#y = tf.Variable(0.) #tf.placeholder(tf.float32, [1])
print("Engine")
with eng:
    # initialize the variables to learn
    Dgate(X[0], 0.) | q[0]
    Dgate(X[0], 0.) | q[1]
    Dgate(X[0], 0.) | q[2]
    Dgate(X[0], 0.) | q[3]

    # setup the entaglement
    BSgate(phi=alpha0) | (q[0], q[1])
    BSgate() | (q[0], q[1])

    BSgate(phi=alpha1) | (q[0], q[2])
    BSgate() | (q[0], q[2])

    BSgate(phi=alpha2) | (q[0], q[3])
    BSgate() | (q[0], q[3])

    BSgate(phi=alpha3) | (q[1], q[2])
    BSgate() | (q[1], q[2])

    BSgate(phi=alpha4) | (q[1], q[3])
    BSgate() | (q[1], q[3])

    BSgate(phi=alpha5) | (q[2], q[3])
    BSgate() | (q[2], q[3])

    # fine tune the results
    Sgate(alpha6) | q[0]
    Sgate(alpha7) | q[1]
    Sgate(alpha8) | q[2]
    Sgate(alpha9) | q[3]

    Dgate(alpha10) | q[0]
    Dgate(alpha11) | q[1]
    Dgate(alpha12) | q[2]
    Dgate(alpha13) | q[3]

    Pgate(alpha14) | q[0]
    Pgate(alpha15) | q[1]
    Pgate(alpha16) | q[2]
    Pgate(alpha17) | q[3]
print("Setup")
state = eng.run('tf', cutoff_dim=10, eval=False)
p0 = state.fock_prob([2, 0, 0, 0])
p1 = state.fock_prob([0, 2, 0, 0])
p2 = state.fock_prob([0, 0, 2, 0])
p3 = state.fock_prob([0, 0, 0, 2])
normalization = p0 + p1 + p2 + p3 + 1e-10
# circuit output is the model
#circuit_output = p1 / normalization
circuit_output = [p0 / normalization, p1 / normalization, p2 / normalization, p3 / normalization]
print("testing")
#########################################################
#
#
# BEGIN THE TRAINING
#
#
##########################################################

def generate_reward(values, action, k):
    values = [0.1,0.1,0.1,0.7] # += np.random.normal(loc=0.0, scale=0.01, size=k)
    #optimal = values.argmax()
    return values, np.random.normal(loc=values[action], scale=1)

#List out our bandits. Currently bandit 4 (index#3) is set to most often provide a positive reward.
bandits = [0.2,0,-0.2,-5]
num_bandits = len(bandits)
def pullBandit(bandit):
    #Get a random number.
    result = np.random.randn(1)
    if result > bandit:
        #return a positive reward.
        return 1
    else:
        #return a negative reward.
        return -1

#tf.reset_default_graph()

#These two lines established the feed-forward part of the network. This does the actual choosing.
#weights = tf.Variable(tf.ones([num_bandits]))
#chosen_action = tf.argmax(weights,0)
weights = circuit_output
chosen_action = tf.argmax(weights)

#The next six lines establish the training proceedure. We feed the reward and chosen action into the network
#to compute the loss, and use it to update the network.
print("Setting up the loss fuction")
print("Reward holder")
reward_holder = tf.placeholder(shape=[1],dtype=tf.float32)
print("Action holder")
action_holder = tf.placeholder(shape=[1],dtype=tf.int32)
print("Responsible Weight")
responsible_weight = tf.slice(weights,action_holder,[1])
print("Loss")
loss = -(tf.log(responsible_weight)*reward_holder)
#loss = -(tf.losses.log_loss(labels=circuit_output, predictions=y[0])*reward_holder)
#loss = tf.losses.mean_squared_error(labels=circuit_output, predictions=y[0])
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
update = optimizer.minimize(loss)

total_episodes = 10000 #Set total number of episodes to train agent on.
total_reward = np.zeros(num_bandits) #Set scoreboard for bandits to 0.
e = 0.1 #Set the chance of taking a random action.

init = tf.initialize_all_variables()
k = 4
values = np.random.normal(loc=0.0, scale=1, size=k)
# Launch the tensorflow graph
rewards0 = []
rewards1 = []
rewards2 = []
rewards3 = []
sess = tf.Session()
sess.run(init)
temp_action = 0
i = 0
while i < total_episodes:
        
    #Choose either a random action or one from our network.
    if np.random.rand(1) < e:
        action = np.random.randint(num_bandits)
    else:
        action = sess.run(chosen_action, feed_dict={X : [temp_action]})
        
    reward = pullBandit(bandits[action]) #Get our reward from picking one of the bandits.
    #values, reward = generate_reward(values, action, k)
        
    #Update the network.
    #_,resp,ww = 
    #sess.run([update,responsible_weight,weights], feed_dict={X: action_holder[action], y: reward_holder[reward]}) #reward_holder:[reward],action_holder:[action]})
    #print(action)
    #sess.run([update, responsible_weight, weights], feed_dict={X: [action], y: [reward], reward_holder:[reward],action_holder:[action]})
    _, resp, ww = sess.run([update, responsible_weight, weights], feed_dict={X: [action], reward_holder:[reward],action_holder:[action]})
    #Update our running tally of scores.
    total_reward[action] += reward
    if i % 50 == 0:
        print( "Running reward for the " + str(num_bandits) + " bandits: " + str(total_reward))
    rewards0.append(total_reward[0])
    rewards1.append(total_reward[1])
    rewards2.append(total_reward[2])
    rewards3.append(total_reward[3])
    i+=1
    temp_action = action

print( "The agent thinks bandit " + str(np.argmax(ww)+1) + " is the most promising....")
if np.argmax(ww) == np.argmax(-np.array(bandits)):
    print( "...and it was right!")
else:
    print( "...and it was wrong!")
plt.plot(rewards0)
plt.show()
plt.plot(rewards1)
plt.show()
plt.plot(rewards2)
plt.show()
plt.plot(rewards3)
plt.show()
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def generate_reward(values, action, k):
    values += np.random.normal(loc=0.0, scale=0.01, size=k)
    optimal = values.argmax()
    return values, np.random.normal(loc=values[action], scale=1)

class contextual_bandit():
    def __init__(self):
        self.state = 0
        #List out our bandits. Currently arms 4, 2, and 1 (respectively) are the most optimal.
        self.bandits = np.array([[0.2,0,-0.0,-5],[0.1,-5,1,0.25],[-5,5,5,5],[1.5,-1,-5,3.5]])
        self.num_bandits = self.bandits.shape[0]
        self.num_actions = self.bandits.shape[1]
        
    def getBandit(self):
        self.state = np.random.randint(0,len(self.bandits)) #Returns a random state for each episode.
        return self.state
        
    def pullArm(self,action):
        #Get a random number.
        bandit = self.bandits[self.state,action]
        result = np.random.randn(1)
        if result > bandit:
            #return a positive reward.
            return 1
        else:
            #return a negative reward.
            return -1


class agent():
    def __init__(self, lr, s_size,a_size):
        #These lines established the feed-forward part of the network. The agent takes a state and produces an action.
        self.state_in= tf.placeholder(shape=[1],dtype=tf.int32)
        state_in_OH = slim.one_hot_encoding(self.state_in,s_size)
        output = slim.fully_connected(state_in_OH,a_size,\
            biases_initializer=None,activation_fn=tf.nn.sigmoid,weights_initializer=tf.ones_initializer())
        self.output = tf.reshape(output,[-1])
        self.chosen_action = tf.argmax(self.output,0)

        #The next six lines establish the training proceedure. We feed the reward and chosen action into the network
        #to compute the loss, and use it to from collections import defaultdictupdate the network.from collections import defaultdictfrom collections import defaultdict
        self.reward_holder = tf.placeholder(shape=[1],dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[1],dtype=tf.int32)
        self.responsible_weight = tf.slice(self.output,self.action_holder,[1])
        self.loss = -(tf.log(self.responsible_weight)*self.reward_holder)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        self.update = optimizer.minimize(self.loss)


tf.reset_default_graph() #Clear the Tensorflow graph.

cBandit = contextual_bandit() #Load the bandits.
myAgent = agent(lr=0.001,s_size=cBandit.num_bandits,a_size=cBandit.num_actions) #Load the agent.
weights = tf.trainable_variables()[0] #The weights we will evaluate to look into the network.

total_episodes = 10000 #Set total number of episodes to train agent on.
total_reward = np.zeros([cBandit.num_bandits,cBandit.num_actions]) #Set scoreboard for bandits to 0.
e = 0.1 #Set the chance of taking a random action.

init = tf.initialize_all_variables()

# k = 4
# 
# 
k = 4
rewards00 = []
rewards01 = []
rewards02 = []
rewards03 = []

rewards10 = []
rewards11 = []
rewards12 = []
rewards13 = []

rewards20 = []
rewards21 = []
rewards22 = []
rewards23 = []

rewards30 = []
rewards31 = []
rewards32 = []
rewards33 = []

values = np.random.normal(loc=0.0, scale=1, size=k)
bandits = defaultdict(list)
# Launch the tensorflow graph
with tf.Session() as sess:
    sess.run(init)
    i = 0
    while i < total_episodes:
        s = cBandit.getBandit() #Get a state from the environment.
        
        #Choose either a random action or one from our network.
        if np.random.rand(1) < e:
            action = np.random.randint(cBandit.num_actions)
        else:
            action = sess.run(myAgent.chosen_action,feed_dict={myAgent.state_in:[s]})
        
        #reward = cBandit.pullArm(action) #Get our reward for taking an action given a bandit.
        values, reward = generate_reward(values, action, k)
        
        #Update the network.
        feed_dict={myAgent.reward_holder:[reward],myAgent.action_holder:[action],myAgent.state_in:[s]}
        _,ww = sess.run([myAgent.update,weights], feed_dict=feed_dict)
        
        #Update our running tally of scores.
        total_reward[s,action] += reward
        #print(total_reward)
        if i % 500 == 0:
            print( "Mean reward for each of the " + str(cBandit.num_bandits) + " bandits: " + str(np.mean(total_reward,axis=1)))
        rewards00.append(total_reward[0][0])
        rewards01.append(total_reward[0][1])
        rewards02.append(total_reward[0][2])
        rewards03.append(total_reward[0][3])

        rewards10.append(total_reward[1][0])
        rewards11.append(total_reward[1][1])
        rewards12.append(total_reward[1][2])
        rewards13.append(total_reward[1][3])

        rewards20.append(total_reward[2][0])
        rewards21.append(total_reward[2][1])
        rewards22.append(total_reward[2][2])
        rewards23.append(total_reward[2][3])

        rewards30.append(total_reward[3][0])
        rewards31.append(total_reward[3][1])
        rewards32.append(total_reward[3][2])
        rewards33.append(total_reward[3][3])
        i+=1
for a in range(cBandit.num_bandits):
    print( "The agent thinks action " + str(np.argmax(ww[a])+1) + " for bandit " + str(a+1) + " is the most promising....")
    if np.argmax(ww[a]) == np.argmin(cBandit.bandits[a]):
        print( "...and it was right!")
    else:
        print( "...and it was wrong!")

plt.plot(rewards00)
plt.show()
plt.plot(rewards01)
plt.show()
plt.plot(rewards02)
plt.show()
plt.plot(rewards03)
plt.show()

plt.plot(rewards10)
plt.show()
plt.plot(rewards11)
plt.show()
plt.plot(rewards12)
plt.show()
plt.plot(rewards13)
plt.show()

plt.plot(rewards20)
plt.show()
plt.plot(rewards21)
plt.show()
plt.plot(rewards22)
plt.show()
plt.plot(rewards23)
plt.show()

plt.plot(rewards30)
plt.show()
plt.plot(rewards31)
plt.show()
plt.plot(rewards32)
plt.show()
plt.plot(rewards33)
plt.show()
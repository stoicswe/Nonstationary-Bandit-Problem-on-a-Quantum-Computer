import strawberryfields as sf
from strawberryfields.ops import *
import tensorflow as tf

eng, q = sf.Engine(2)

alpha0 = tf.Variable(0.1)
alpha1 = tf.Variable(0.1)
alpha2 = tf.Variable(0.1)
alpha3 = tf.Variable(0.1)
alpha4 = tf.Variable(0.1)
alpha5 = tf.Variable(0.1)
alpha6 = tf.Variable(0.1)
alpha7 = tf.Variable(0.1)
alpha8 = tf.Variable(0.1)
X = tf.placeholder(tf.float32, [2])
y = tf.placeholder(tf.float32)

with eng:
    Dgate(X[0], 0.) | q[0]
    Dgate(X[1], 0.) | q[1]
    BSgate(phi=alpha0) | (q[0], q[1])
    BSgate() | (q[0], q[1])
    Sgate(alpha3) | q[0]
    Sgate(alpha4) | q[1]
    Dgate(alpha5) | q[0]
    Dgate(alpha6) | q[1]
    Pgate(alpha7) | q[0]
    Pgate(alpha8) | q[1]

state = eng.run('tf', cutoff_dim=10, eval=False)
p0 = state.fock_prob([0, 2])
p1 = state.fock_prob([2, 0])
normalization = p0 + p1 + 1e-10
# circuit output is the model
circuit_output = p1 / normalization
loss = tf.losses.mean_squared_error(labels=circuit_output, predictions=y)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
minimize_op = optimizer.minimize(loss)

X_train = [[0,0], [0,1], [1,0], [1,1]]
y_train = [0, 1, 1, 1]

sess = tf.Session()
sess.run(tf.global_variables_initializer())

steps = 100
for s in range(steps):
    for i in range(4):
        sess.run(minimize_op, feed_dict={X: X_train[i], y: y_train[i]})
    if (s % 10 == 0):
        print("{0}%".format((s/steps)*100))

print("X       Prediction")
for x in X_train:
    print("{0} || {1}".format(x, sess.run(circuit_output, feed_dict={X: x})))
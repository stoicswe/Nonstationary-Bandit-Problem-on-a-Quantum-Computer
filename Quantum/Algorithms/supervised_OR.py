import strawberryfields as sf
from strawberryfields.ops import Dgate, BSgate, Vgate, Sgate
import tensorflow as tf
from qmlt.tf.helpers import make_param
from qmlt.tf import CircuitLearner

steps = 1000

def circuit(X):
    params = [make_param(name='phi', constant=2.), make_param(name="theta", constant=1.), make_param(name="theta2", constant=.1), make_param(name="theta3", constant=.1), 
              make_param(name="theta4", constant=.1), make_param(name="theta5", constant=.1),
              make_param(name="theta6", constant=.1), make_param(name="theta7", constant=.1),
              make_param(name="theta8", constant=.1), make_param(name="theta9", constant=.1),]

    eng, q = sf.Engine(2)

    with eng:
        Dgate(X[:, 0], 0.) | q[0]
        Dgate(X[:, 1], 0.) | q[1]
        BSgate(phi=params[0]) | (q[0], q[1])
        BSgate() | (q[0], q[1])
        Vgate(params[1]) | q[0]
        Vgate(params[2]) | q[1]
        
    num_inputs = X.get_shape().as_list()[0]
    state = eng.run('tf', cutoff_dim=10, eval=False, batch_size=num_inputs)

    p0 = state.fock_prob([0, 2])
    p1 = state.fock_prob([2, 0])
    normalization = p0 + p1 + 1e-10
    output1 = p1 / normalization
    
    
    with eng:
        X1 = output1
        Dgate(X1, 0.) | q[0]
        Dgate(0., X1) | q[1]
        BSgate(phi=params[1]) | (q[0], q[1])
        BSgate() | (q[0], q[1])
        Vgate(params[5]) | q[0]
        Vgate(params[6]) | q[1]

    num_inputs1 = X1.get_shape().as_list()[0]
    state2 = eng.run('tf', cutoff_dim=10, eval=False, batch_size=num_inputs1)

    p00 = state2.fock_prob([0, 2])
    p01 = state2.fock_prob([2, 0])
    normalization1 = p00 + p01 + 1e-10
    circuit_output = p01 / normalization1
    return circuit_output

def myloss(circuit_output, targets):
    return tf.losses.mean_squared_error(labels=circuit_output, predictions=targets)

def outputs_to_predictions(circuit_output):
    return tf.round(circuit_output)

X_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y_train = [0, 1, 1, 1]
X_test = X_train
Y_test = Y_train
X_pred = X_train

hyperparams = {'circuit': circuit,
               'task': 'supervised',
               'loss': myloss,
               'optimizer': 'SGD',
               'init_learning_rate': 0.01,
               'print_log': False}
learner = CircuitLearner(hyperparams=hyperparams)
print("Training on: X{0} Y{1}".format(X_train, Y_train))
learner.train_circuit(X=X_train, Y=Y_train, steps=steps)
test_score = learner.score_circuit(X=X_test, Y=Y_test,
                                   outputs_to_predictions=outputs_to_predictions)
print("\nPossible scores to print: {}".format(list(test_score.keys())))
print("Accuracy on test set: ", test_score['accuracy'])
print("Loss on test set: ", test_score['loss'])

outcomes = learner.run_circuit(X=X_pred, outputs_to_predictions=outputs_to_predictions)

print("Predicting based on X_Predict{0}".format(X_pred))
print("\nPossible outcomes to print: {}".format(list(outcomes.keys())))
print("Predictions for new inputs: {}".format(outcomes['predictions']))





# Optimizer !!!!!!! on website
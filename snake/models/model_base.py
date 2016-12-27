#load useful libraries
import tensorflow as tf

def model_forward(input_frames, n_input, n_hidden, n_classes):
    # initialize weights
    w1 = tf.Variable(tf.truncated_normal([n_input, n_hidden], stddev = .1))
    b1 = tf.Variable(tf.zeros([1, n_hidden]))
    w2 = tf.Variable(tf.truncated_normal([n_hidden, n_classes], stddev = .1))
    b2 = tf.Variable(tf.zeros([1, n_classes]))
    weights = {'w1':w1, 'b1':b1, 'w2':w2, 'b2':b2}
    
    # define network structure
    hidden_layer = tf.add(tf.matmul(input_frames, w1), b1)
    hidden_layer = tf.nn.relu(hidden_layer)
    out_layer = tf.add(tf.matmul(hidden_layer, w2), b2)
    out_probs = tf.nn.softmax(out_layer)
    
    return out_probs, weights
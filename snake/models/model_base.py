#load useful libraries
import tensorflow as tf

class FullyConnected:
    def __init__(self, n_input, n_hidden, n_classes):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_classes = n_classes

    def model_forward(self, input_frames):
        # initialize weights
        w1 = tf.Variable(tf.truncated_normal([self.n_input, self.n_hidden], stddev = .1))
        b1 = tf.Variable(tf.zeros([1, self.n_hidden]))
        w2 = tf.Variable(tf.truncated_normal([self.n_hidden, self.n_classes], stddev = .1))
        b2 = tf.Variable(tf.zeros([1, self.n_classes]))
        self.weights = {'w1':w1, 'b1':b1, 'w2':w2, 'b2':b2}

        # define network structure
        hidden_layer = tf.add(tf.matmul(input_frames, w1), b1)
        hidden_layer = tf.nn.relu(hidden_layer)
        out_layer = tf.add(tf.matmul(hidden_layer, w2), b2)
        out_probs = tf.nn.softmax(out_layer)

        return out_probs

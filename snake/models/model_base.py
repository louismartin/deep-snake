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

        # Define network structure
        # Reshape/flatten input to linear vector

        input_frames = tf.reshape(input_frames, shape=[-1, self.n_input])
        hidden_layer = tf.add(tf.matmul(input_frames, w1), b1)
        hidden_layer = tf.nn.relu(hidden_layer)
        out_layer = tf.add(tf.matmul(hidden_layer, w2), b2)
        out_probs = tf.nn.softmax(out_layer)

        return out_probs


# Create some wrappers for simplicity
#(based on https://github.com/aymericdamien/TensorFlow-Examples)
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

class ConvNet:
    def __init__(self, grid_size, n_frames, n_classes):
        self.n_frames = n_frames
        self.grid_size = grid_size
        self.n_classes = n_classes

    def model_forward(self, input_frames):
        n_frames = self.n_frames
        grid_size = self.grid_size
        n_classes = self.n_classes

        # Store layers weight & bias
        weights = {
            # 5x5 conv, 1 input, 32 outputs
            'wc1': tf.Variable(tf.random_normal([3, 3, n_frames, 16])),
            'bc1': tf.Variable(tf.random_normal([16])),
            # 5x5 conv, 32 inputs, 64 outputs
            'wc2': tf.Variable(tf.random_normal([3, 3, 16, 32])),
            'bc2': tf.Variable(tf.random_normal([32])),
            # fully connected, 7*7*64 inputs, 1024 outputs
            'wd1': tf.Variable(tf.random_normal([3*3*32, 128])),
            'bd1': tf.Variable(tf.random_normal([128])),
            # 1024 inputs, 10 outputs (class prediction)
            'wout': tf.Variable(tf.random_normal([128, n_classes])),
            'bout': tf.Variable(tf.random_normal([n_classes]))
        }
        self.weights = weights

        # TODO: add fire modules (1x1 convolutions for shielding 3x3 conv)
        # TODO: add max pooling
        # Convolution Layer
        conv1 = conv2d(input_frames, weights['wc1'], weights['bc1'])
        # Max Pooling (down-sampling)
        #conv1 = maxpool2d(conv1, k=2)

        # Convolution Layer
        conv2 = conv2d(conv1, weights['wc2'], weights['bc2'])
        # Max Pooling (down-sampling)
        conv2 = maxpool2d(conv2, k=2)

        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), weights['bd1'])
        fc1 = tf.nn.relu(fc1)
        # Apply Dropout
        #fc1 = tf.nn.dropout(fc1, dropout)

        # Output, class prediction
        out = tf.add(tf.matmul(fc1, weights['wout']), weights['bout'])
        out = tf.nn.softmax(out)
        return out

#load useful libraries
import tensorflow as tf

class FullyConnected:
    def __init__(self, n_input, n_hidden, n_classes):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_classes = n_classes

    def forward(self, input_frames):
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


class ConvNet:
    '''
    Takes a grid as input and outputs an action (int)
    '''
    def __init__(self, n_frames, n_classes, nb_blocks=2):
        '''
        Args:
            nb_blocks (int): Number of residual blocks
        '''
        self.weights = {}
        self.n_frames = n_frames
        self.nb_blocks = nb_blocks
        self.n_classes = n_classes

    def conv2d(self, x, name, filter_size, nb_filter, stride=1, relu=True):
        ''' Conv2d wrapper, with bias and relu activation '''
        nb_input = x.get_shape().as_list()[3]
        W = tf.Variable(tf.random_normal([filter_size, filter_size, nb_input, nb_filter]))
        b = tf.Variable(tf.random_normal([nb_filter]))
        self.weights['{}_w'.format(name)] = W
        self.weights['{}_b'.format(name)] = b
        x = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        if relu:
            x = tf.nn.relu(x)
        return x

    def dense(self, x, name, nb_input, nb_filter, relu=True):
        ''' Dense / fully connected layer '''
        W = tf.Variable(tf.random_normal([nb_input, nb_filter]))
        b = tf.Variable(tf.random_normal([nb_filter]))
        self.weights['{}_w'.format(name)] = W
        self.weights['{}_b'.format(name)] = b
        x = tf.add(tf.matmul(x, W), b)
        if relu:
            x = tf.nn.relu(x)
        return x

    def maxpool2d(self, x, k=2):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                              padding='SAME')

    def residual_block(self, x, name):
        # Squeeze input with 1x1 convolutions
        conv1 = self.conv2d(x, '{}_conv1'.format(name), 1, 8, 1, relu=True)
        # TODO: Batch norm
        conv2 = self.conv2d(conv1, '{}_conv2'.format(name), 3, 32, 1, relu=False)
        # TODO: Batch norm
        out = conv2 + x # Residual connection
        return out

    def forward(self, input_frames):
        n_frames = self.n_frames
        n_classes = self.n_classes

        # TODO: add fire modules (1x1 convolutions for shielding 3x3 conv)
        # TODO: add max pooling
        # First convolutional Layer
        conv1 = self.conv2d(input_frames, 'conv1', 3, 32, 1, relu=True)
        # Max Pooling (down-sampling)
        #conv1 = maxpool2d(conv1, k=2)

        # Residual blocks
        x = conv1
        for i in range(self.nb_blocks):
            x =  self.residual_block(x, 'res{}'.format(i+1))

        # Convolution Layer
        conv2 = self.conv2d(input_frames, 'conv1', 3, 32, 1, relu=True)
        x = x + conv1 # Residual connection
        # Max Pooling (down-sampling)
        conv2 = self.maxpool2d(conv2, k=2)

        # Fully connected layer
        nb_input = 3*3*32
        # Reshape conv2 output to fit fully connected layer input
        fc1 = tf.reshape(conv2, [-1, nb_input])
        fc1 = self.dense(fc1, 'fc1', nb_input, 128)
        # Apply Dropout
        #fc1 = tf.nn.dropout(fc1, dropout)

        # Output, class prediction
        out = self.dense(fc1, 'fc1', 128, n_classes, relu=False)
        out = tf.nn.softmax(out)
        return out

#load useful libraries
import numpy as np
import tensorflow as tf
import pickle as pkl
import matplotlib.pyplot as plt

from time import time
from numpy import random
from tools import sample_from_policy, discount_rewards, play_game
from tqdm import tqdm

# ------- Train ------- #
def train(model, snake, batch_size=100, n_iterations=100, gamma=1, learning_rate=0.001, n_frames=2, plot=True):
    print('Start training')
    # define placeholders for inputs and outputs
    input_frames = tf.placeholder(tf.float32, [None, snake.grid_size, snake.grid_size, n_frames])
    y_played = tf.placeholder(tf.float32, [None, model.n_classes])
    advantages = tf.placeholder(tf.float32, [1, None])

    # load model
    print('Loading %s model' % model.__class__.__name__)
    out_probs = model.forward(input_frames)

    # define loss and optimizer
    epsilon = 1e-15
    log_probs = tf.log(tf.add(out_probs, epsilon))
    loss = tf.reduce_sum(tf.matmul(advantages,(tf.mul(y_played, log_probs))))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    # initialize the variables
    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        # initialize variables
        sess.run(init)

        # used later to save variables for the batch
        frames_stacked = []
        targets_stacked = []
        rewards_stacked = []
        lifetime = []
        avg_lifetime = []
        avg_reward = []
        description = ''
        pbar = tqdm(range(n_iterations), desc=description)
        for iterations_count in pbar:
            fruits_count = 0
            # One iteration is a batch of batch_size games
            for game_count in range(batch_size):
                # Play one game
                frames, actions, rewards, fruits = play_game(snake, model, sess, n_frames=2)

                # stack rewards
                fruits_count += fruits
                lifetime.append(len(rewards))
                rewards_stacked.append(discount_rewards(rewards, gamma))
                for f in frames:
                    frames_stacked.append(f)
                for a in actions:
                    targets_stacked.append(a)

            # Update progress bar description
            description = 'Lifetime: {}, Fruits: {}'.format(np.mean(lifetime), fruits_count)
            pbar.set_description(description)

            # stack frames, targets and rewards
            frames_stacked = np.vstack(frames_stacked)
            targets_stacked = np.vstack(targets_stacked)
            rewards_stacked = np.hstack(rewards_stacked)
            rewards_stacked = np.reshape(rewards_stacked, (1, len(rewards_stacked)))*1.
            avg_lifetime.append(np.mean(lifetime))
            avg_reward.append(np.mean(rewards_stacked))

            # normalize rewards
            rewards_stacked -= np.mean(rewards_stacked)
            std = np.std(rewards_stacked)
            if std != 0:
                rewards_stacked /= std

            # backpropagate
            sess.run(optimizer, feed_dict={input_frames: frames_stacked, y_played: targets_stacked, advantages: rewards_stacked})

            # reset variables
            frames_stacked = []
            targets_stacked = []
            rewards_stacked = []
            lifetime = []


        # save model
        model_path = 'weights/weights_fc_' + model.__class__.__name__ + '.p'
        print('Saving model to ' + model_path)
        pkl.dump({k: v.eval() for k, v in model.weights.items()}, open(model_path,'w'))

        if plot:
            # Plot useful statistics
            plt.plot(avg_lifetime)
            plt.title('Average lifetime')
            plt.xlabel('Iteration')
            plt.savefig('graphs/average_lifetime_' + model.__class__.__name__ + '.png')
            plt.show()

            plt.plot(avg_reward)
            plt.title('Average reward')
            plt.xlabel('Iteration')
            plt.savefig('graphs/average_reward_' + model.__class__.__name__ + '.png')
            plt.show()

# ---- Test ---- #
def test(model, snake, n_frames=2):

    # initialize parameters
    n_classes = 4

    # load model
    input_frames = tf.placeholder(tf.float32, [None, snake.grid_size, snake.grid_size, n_frames])
    out_probs = model.forward(input_frames)

    # asssign weights
    model_path = 'weights/weights_fc_' + model.__class__.__name__ + '.p'
    print('Loading model from ' + model_path)

    weights_trained = pkl.load(open(model_path, 'rb'))
    assigns = []
    for weight_key in weights_trained:
        assert weight_key in model.weights
        assign = tf.assign(model.weights[weight_key], weights_trained[weight_key])
        assigns.append(assign)

    # initialize the variables
    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        # initialize variables
        sess.run(init)
        # Load weights
        for assign in assigns:
            sess.run(assign)

        # loop for n games
        n = 10
        for i in range(n):
            frames, actions, rewards, fruits = play_game(snake, model, sess, n_frames=2, display=True)

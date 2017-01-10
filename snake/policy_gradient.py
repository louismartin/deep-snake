import time
import os.path

import numpy as np
import tensorflow as tf
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from numpy import random

from tools import sample_from_policy, discount_rewards, play_game


def train(model, snake, warm_restart=False, batch_size=100, n_iterations=100,
          gamma=1, learning_rate=0.001, n_frames=2, plot=True):
    print('Start training')
    start_time = time.time()
    # define placeholders for inputs and outputs
    input_frames = tf.placeholder(tf.float32, [None, snake.grid_size,
                                               snake.grid_size, n_frames])
    y_played = tf.placeholder(tf.float32, [None, model.n_classes])
    advantages = tf.placeholder(tf.float32, [1, None])

    # load model
    print('Loading %s model' % model.__class__.__name__)
    out_probs = model.forward(input_frames)

    # define loss and optimizer
    epsilon = 1e-15
    log_probs = tf.log(tf.add(out_probs, epsilon))
    loss = tf.reduce_sum(tf.matmul(advantages, (tf.mul(y_played, log_probs))))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    optimizer = optimizer.minimize(loss)
    # initialize the variables
    init = tf.global_variables_initializer()

    ops = []
    if warm_restart:
        # Load previous weights
        model_path = 'weights/weights_' + model.__class__.__name__ + '.p'
        print('Loading model from ' + model_path)

        weights_trained = pkl.load(open(model_path, 'rb'))
        for weight_key in weights_trained:
            assert weight_key in model.weights
            assign = tf.assign(model.weights[weight_key],
                               weights_trained[weight_key])
            ops.append(assign)

    with tf.Session() as sess:

        # initialize variables
        sess.run(init)
        for op in ops:
            sess.run(ops)

        avg_lifetime = np.zeros(n_iterations)
        avg_fruits = np.zeros(n_iterations)
        avg_reward = np.zeros(n_iterations)
        pbar = tqdm(range(n_iterations), desc='')
        for i in pbar:
            fruits_count = []
            frames_stacked = []
            targets_stacked = []
            rewards_stacked = []
            lifetime = []
            # One iteration is a batch of batch_size games
            for game_count in range(batch_size):
                # Play one game
                frames, actions, rewards, fruits = play_game(
                    snake, model, sess, n_frames=n_frames
                )

                # stack rewards
                fruits_count.append(fruits)
                lifetime.append(len(rewards))
                rewards = discount_rewards(rewards, gamma)
                rewards_stacked.append(rewards)
                for f in frames:
                    frames_stacked.append(f)
                for a in actions:
                    targets_stacked.append(a)
            # Update progress bar description
            description = 'Lifetime: {}, Fruits: {}'.format(
                np.mean(lifetime), np.mean(fruits_count)
            )
            pbar.set_description(description)

            # stack frames, targets and rewards
            frames_stacked = np.vstack(frames_stacked)
            targets_stacked = np.vstack(targets_stacked)
            rewards_stacked = np.hstack(rewards_stacked).astype(np.float32)
            rewards_stacked = rewards_stacked.reshape(1, len(rewards_stacked))

            avg_lifetime[i] = np.mean(lifetime)
            avg_reward[i] = np.mean(rewards_stacked)
            avg_fruits[i] = np.mean(fruits_count)

            # normalize rewards
            rewards_stacked -= np.mean(rewards_stacked)
            std = np.std(rewards_stacked)
            if std != 0:
                rewards_stacked /= std

            # backpropagate
            sess.run(optimizer, feed_dict={input_frames: frames_stacked,
                                           y_played: targets_stacked,
                                           advantages: rewards_stacked})
        total_time = time.time() - start_time
        # save model
        model_path = 'weights/weights_' + model.__class__.__name__ + '.p'
        print('Saving model to ' + model_path)
        pkl.dump({k: v.eval() for k, v in model.weights.items()},
                 open(model_path, 'w'))

        if plot:
            sns.set_style('darkgrid')
            # Plot useful statistics
            plt.plot(avg_lifetime)
            plt.title('Average lifetime')
            plt.xlabel('Iteration')
            plt.savefig('graphs/average_lifetime_{}.png'.format(
                model.__class__.__name__)
            )
            plt.show()

            plt.plot(avg_fruits)
            plt.title('Average fruits eaten')
            plt.xlabel('Iteration')
            plt.savefig('graphs/average_fruits_{}.png'.format(
                model.__class__.__name__)
            )
            plt.show()
            sns.reset_orig()
    return avg_lifetime, avg_fruits, total_time


def test(model, snake, n_frames=2, display=True, save=False):
    # load model
    input_frames = tf.placeholder(tf.float32, [None, snake.grid_size,
                                               snake.grid_size, n_frames])
    out_probs = model.forward(input_frames)

    # asssign weights
    model_path = 'weights/weights_' + model.__class__.__name__ + '.p'
    print('Loading model from ' + model_path)

    weights_trained = pkl.load(open(model_path, 'rb'))
    assigns = []
    for weight_key in weights_trained:
        assert weight_key in model.weights
        assign = tf.assign(model.weights[weight_key],
                           weights_trained[weight_key])
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
        lifetime = []
        n = 100
        for i in range(n):
            frames, actions, rewards, fruits = play_game(
                snake, model, sess, n_frames=n_frames, display=display, save=save
            )
            lifetime.append(len(rewards))
    return np.mean(lifetime)

#load useful libraries
import numpy as np
import tensorflow as tf
import pickle as pkl
import matplotlib.pyplot as plt

from time import time
from numpy import random
from tools import sample_from_policy, discount_rewards

# ------- Train ------- #
def train(model, snake, n_batch = 100, n_iterations = 100, n_hidden = 200, gamma = 1, learning_rate = 0.001):
    # define placeholders for inputs and outputs
    input_frames = tf.placeholder(tf.float32, [None, model.n_input])
    y_played = tf.placeholder(tf.float32, [None, model.n_classes])
    advantages = tf.placeholder(tf.float32, [1, None])

    # load model
    print('Loading %s model' % model.__class__.__name__)
    out_probs = model.model_forward(input_frames)

    # define loss and optimizer
    epsilon = 1e-15
    log_probs = tf.log(tf.add(out_probs, epsilon))
    loss = tf.reduce_sum(tf.matmul(advantages,(tf.mul(y_played, log_probs))))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    # initialize the variables
    init = tf.global_variables_initializer()

    # initialize counts
    games_count = 0
    iterations_count = 0
    start_time = time()

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
        fruits_count = 0

        while iterations_count < n_iterations:

            # initialize snake environment and some variables
            snake.reset()
            frame_curr = np.zeros((snake.grid_size, snake.grid_size))
            rewards_running = []

            # loop for one game
            while not snake.game_over:

                # get current frame
                frame_prev = np.copy(frame_curr)
                frame_curr = snake.grid

                # forward previous and current frames
                last_two_frames = np.reshape(np.hstack((frame_prev, frame_curr)), (1, model.n_input))
                frames_stacked.append(last_two_frames)
                policy = np.ravel(sess.run(out_probs, feed_dict = {input_frames : last_two_frames}))

                # sample action from returned policy
                action = sample_from_policy(policy)

                # build labels according to the sampled action
                target = np.zeros(4)
                target[action] = 1

                # play THE SNAKE and get the reward associated to the action
                reward = snake.play(action)
                if snake.is_food_eaten:
                    fruits_count += 1

                # save targets and rewards
                targets_stacked.append(target)
                rewards_running.append(reward)

                # to avoid infinite loops which can make one game very long
                if len(rewards_running) > 50:
                    break

            # stack rewards
            games_count += 1
            lifetime.append(len(rewards_running)*1.)
            rewards_stacked.append(discount_rewards(rewards_running, gamma))

            if games_count % n_batch == 0:
                iterations_count += 1

                # display
                if iterations_count % (n_iterations//10) == 0:
                    print("Batch #%d, average lifetime: %.2f, fruits eaten: %d, games played: %d, time: %d sec" %
                            (iterations_count, np.mean(lifetime), fruits_count, games_count, time() - start_time))

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
                fruits_count = 0

        # save model
        model_path = 'weights/weights_fc_' + model.__class__.__name__ + '.p'
        print('Saving model to ' + model_path)
        pkl.dump({k: v.eval() for k, v in model.weights.items()}, open(model_path,'w'))

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
def test(model, snake, n_iterations = 100, n_hidden = 200):

    # initialize parameters
    n_input = 2 * snake.grid_size * snake.grid_size
    n_classes = 4

    # load model
    input_frames = tf.placeholder(tf.float32, [None, n_input])
    out_probs = model.model_forward(input_frames)

    # asssign weights
    model_path = 'weights/weights_fc_' + model.__class__.__name__ + '.p'
    print('Loading model from ' + model_path)

    weights_trained = pkl.load(open(model_path, 'rb'))
    assign_w1 = tf.assign(model.weights['w1'], weights_trained['w1'])
    assign_b1 = tf.assign(model.weights['b1'], weights_trained['b1'])
    assign_w2 = tf.assign(model.weights['w2'], weights_trained['w2'])
    assign_b2 = tf.assign(model.weights['b2'], weights_trained['b2'])

    # initialize the variables
    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        # initialize variables
        sess.run(init)
        sess.run(assign_w1)
        sess.run(assign_b1)
        sess.run(assign_w2)
        sess.run(assign_b2)

        # loop for n games
        n = 10
        for i in range(n):
            # initialize snake environment and some variables
            snake.reset()
            frame_curr = snake.grid
            rewards_running = []

            while not snake.game_over:
                snake.display()
                # get current frame
                frame_prev = np.copy(frame_curr)
                frame_curr = snake.grid

                # forward previous and current frames
                last_two_frames = np.reshape(np.hstack((frame_prev, frame_curr)), (1, n_input))
                policy = np.ravel(sess.run(out_probs, feed_dict = {input_frames : last_two_frames}))

                # sample action from returned policy
                action = np.argmax(policy)

                # play THE SNAKE and get the reward associated to the action
                reward = snake.play(action)
                rewards_running += [reward]
                # to avoid infinite loops which can make one game very long
                if len(rewards_running) > 50:
                    break
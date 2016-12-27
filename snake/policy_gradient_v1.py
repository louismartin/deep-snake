# ------- Setup ------ #
#load useful libraries
import numpy as np
import tensorflow as tf
from numpy import random
from tools import sample_from_policy

# load THE SNAKE
from snake.snake import Snake
snake = Snake()

# define parameters
n_input = 2 * snake.grid_size * snake.grid_size
n_hidden = 200
n_classes = 4

# --- Policy Network --- #

# define placeholders for inputs and outputs
input_frames = tf.placeholder(tf.float32, [None, n_input])
y_true = tf.placeholder(tf.float32, [None, n_classes])
advantages = tf.placeholder(tf.float32, [1, None])

# initialize weights
w1 = tf.Variable(tf.truncated_normal([n_input, n_hidden]))
b1 = tf.Variable(tf.zeros([1, n_hidden]))
w2 = tf.Variable(tf.truncated_normal([n_hidden, n_classes]))
b2 = tf.Variable(tf.zeros([1, n_classes]))

# define network structure
hidden_layer = tf.add(tf.matmul(input_frames, w1), b1)
hidden_layer = tf.nn.relu(hidden_layer)
out_layer = tf.add(tf.matmul(hidden_layer, w2), b2)
out_probs = tf.nn.softmax(out_layer)

# define loss and optimizer
epsilon = 1e-15
log_probs = tf.log(tf.add(out_probs, epsilon))
loss = tf.reduce_sum(tf.matmul(advantages,(tf.mul(y_true, log_probs))))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# ------ Train ------ #
def train(n_batch, n_iterations):

    # initialize the variables
    init = tf.global_variables_initializer()
    
    # initialize counts
    games_count = 0
    iterations_count = 0

    with tf.Session() as sess:

        # initialize variables
        sess.run(init)

        # used later to save variables for the batch
        frames_stacked = []
        targets_stacked = []
        rewards_stacked = []
        life_time = []
        fruits_count = 0

        while iterations_count < n_iterations:

            # initialize snake environment and some variables
            snake = Snake()
            frame_curr = snake.display()
            rewards_running = []
            reset = False

            # loop for one game
            while not reset: 

                # get current frame
                frame_prev = np.copy(frame_curr)
                frame_curr = snake.display()

                # forward previous and current frames
                last_two_frames = np.reshape(np.hstack((frame_prev, frame_curr)), (1, n_input))
                frames_stacked.append(last_two_frames)
                policy = np.ravel(sess.run(out_probs, feed_dict = {input_frames : last_two_frames}))

                # sample action from returned policy 
                action = sample_from_policy(policy)

                # build labels according to the sampled action
                target = np.zeros(4)
                target[action] = 1

                # play THE SNAKE and get the reward associated to the action
                reward, reset = snake.play(action)
                if reward == 100:
                    fruits_count += 1
                rewards_running += [reward] 

                # save targets
                targets_stacked.append(target)

            # stack rewards
            games_count += 1
            life_time.append(len(rewards_running)*1.)
            rewards_stacked.append([np.sum(rewards_running)] * len(rewards_running)) # TODO: add gamma factor


            # every batch
            if games_count % n_batch == 0: 
                
                # display every 10 batches
                if iterations_count % 10 == 0:
                    print("Batch %d, average lifetime %.2f, fruits eaten %d" %(iterations_count + 1, np.mean(life_time), fruits_count))

                # stack frames, targets and rewards
                frames_stacked = np.vstack(frames_stacked)
                targets_stacked = np.vstack(targets_stacked)
                rewards_stacked = np.hstack(rewards_stacked)
                rewards_stacked = np.reshape(rewards_stacked, (1, len(rewards_stacked)))*1.

                # normalize rewards
                rewards_stacked -= np.mean(rewards_stacked)
                rewards_stacked /= np.std(rewards_stacked)

                # backpropagate 
                sess.run([optimizer, loss], feed_dict={input_frames: frames_stacked, y_true: targets_stacked, advantages: rewards_stacked})

                # update variables
                iterations_count += 1
                frames_stacked = []
                targets_stacked = []
                rewards_stacked = []
                life_time = []
                fruits_count = 0

        return w1.eval(), b1.eval(), w2.eval(), b2.eval()   
    
# ---- Test ---- #
def test(weights):
     
    # asssign weights
    assign_w1 = tf.assign(w1, weights[0])
    assign_b1 = tf.assign(b1, weights[1])
    assign_w2 = tf.assign(w2, weights[2])
    assign_b2 = tf.assign(b2, weights[3])

    # initialize the variables
    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        # initialize variables
        sess.run(init)
        sess.run(assign_w1)
        sess.run(assign_b1)
        sess.run(assign_w2)
        sess.run(assign_b2)

        # initialize snake environment and some variables
        snake = Snake()
        frame_curr = snake.display()
        rewards_running = []
        reset = False

        # loop for one game
        while not reset: 

            # get current frame
            frame_prev = np.copy(frame_curr)
            frame_curr = snake.display()
            print(frame_curr)
            
            # forward previous and current frames
            last_two_frames = np.reshape(np.hstack((frame_prev, frame_curr)), (1, n_input))
            policy = np.ravel(sess.run(out_probs, feed_dict = {input_frames : last_two_frames}))
            print(policy)
            
            # sample action from returned policy 
            action = sample_from_policy(policy)
            print(action)
            # play THE SNAKE and get the reward associated to the action
            reward, reset = snake.play(action)
            rewards_running += [reward]      
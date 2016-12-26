# load useful libraries
import numpy as np
from numpy import random
from tools import sample_from_policy

# load THE SNAKE
from snake import Snake
snake = Snake()

# define parameters
n_grid = snake.grid_size
n_hidden = 200
n_iterations = 1
batch_size = 100 #1 batch = 100 games

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop

# build 1-hidden layer policy network with ReLU activations
policy_network = Sequential()
policy_network.add(Dense(output_dim=n_hidden, input_shape=(1, 2 * n_grid * n_grid)))
policy_network.add(Activation("relu"))
policy_network.add(Dense(output_dim=4))
policy_network.add(Activation("softmax"))

# define loss function and optimization method
opt = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
policy_network.compile(loss='categorical_crossentropy', optimizer=opt)

games_count = 0

while iterations_count < n_iterations:

    reward_cum = 0
    frames_stacked =
    targets_stacked =

    # loop for one game
    while not reset:
        # forward two consecutive frames
        frames = random.random((2, n_grid, n_grid)).reshape(1, 1, 2 * n_grid * n_grid)
        policy = policy_network.predict(frames)

        # sample action from returned policy
        action = sample_from_policy(policy)

        # build labels according to the sampled action
        target = np.zeros(4)
        target[action] = 1

        # play THE SNAKE and get the reward associated to the action
        reward, reset = snake.play(action)
        reward_cum += reward # TODO: add gamma factor

        # save frames and targets
        frames_stacked = np.vstack(())
        targets_stacked = np.vstack(())

    if game_count % batch_size == 0:
        #perform backpropagation and update parameters
        #the loss is ponderated by the advantage of the games
        # TODO: the forward pass is done twice, which is bad
        # TODO: ponderate the loss by the advantage of the games
        policy_network.fit(frames_stacked, targets_stacked)

        # update counts
        iterations_count += 1

    # update counts
    games_count += 1

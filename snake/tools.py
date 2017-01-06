import numpy as np
from collections import deque
from numpy import random

def sample_from_policy(t):
    p = random.random()
    cdf = np.cumsum(t)
    return np.where(cdf >= p)[0][0]

def discount_rewards(rewards, gamma):
    rewards_new = np.zeros(len(rewards))
    discount_sum = 0
    for i in reversed(xrange(len(rewards))):
        discount_sum *= gamma
        discount_sum += rewards[i]
        rewards_new[i] = discount_sum
    return rewards_new

def play_game(snake, model, sess, n_frames=2, display=False):
    '''
    Play one game and returns the rewards
    Must be called inside a tensorflow session with variable initialized
    '''
    snake.reset()
    fruits_count = 0
    time_since_last_fruit = 0
    rewards = []
    actions = []
    frames = []
    running_frames = deque()
    # Add frames filled of zeros to be able to consider n_frames at the
    # beginning.
    for i in range(n_frames):
        running_frames.append(np.zeros((snake.grid_size, snake.grid_size)))

    while not snake.game_over:
        if display: snake.display()
        # get current frame and remove last frame
        running_frames.popleft()
        running_frames.append(snake.grid)
        current_frames = np.array(running_frames)
        # Transform to shape (1,grid_size,grid_size,n_frames)
        current_frames = np.expand_dims(current_frames.transpose((1,2,0)), 0)
        frames.append(current_frames)

        policy = np.ravel(sess.run(model.out_probs, feed_dict={model.input_frames: current_frames}))

        # sample action from returned policy and convert to one-hot vector
        action = sample_from_policy(policy)
        target = np.zeros(4)
        target[action] = 1

        # Play
        reward = snake.play(action)
        if snake.is_food_eaten:
            fruits_count += 1 # TODO: convert to internal attribute of snake
            time_since_last_fruit = 0
        actions.append(target)
        rewards.append(reward)

        # Prevent infinite loop
        if time_since_last_fruit > 2*snake.grid_size:
            break
        time_since_last_fruit += 1

    return np.array(frames), np.array(actions), np.array(rewards), fruits_count

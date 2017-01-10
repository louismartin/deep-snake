from collections import deque
from random import randint

import numpy as np
import matplotlib.pyplot as plt


class Snake:
    def __init__(self, grid_size=5, snake_length=3,
                 rewards={'nothing': -10, 'bitten': -1, 'out': -1, 'food': 10},
                 verbose=0):
        self.grid_size = grid_size
        self.snake_length = snake_length
        self.rewards = rewards
        self.verbose = verbose
        # TODO: add idle action ? (Check if performance increases)
        self.actions = {
            0: (0, -1),  # Move left
            1: (-1, 0),  # Move up
            2: (0, +1),  # Move right
            3: (+1, 0)   # Move down
        }
        self.im = None
        self.reset()

    def reset(self):
        # Position of the snake; leftmost element is the head,
        # the rest is the tail
        # Example: snake = [(0,0), (0,1), (0,2)]
        self.grid = np.zeros((self.grid_size, self.grid_size))
        row_init = randint(0, self.grid_size - 1)
        snake = [(row_init, col) for col in range(self.snake_length)]
        self.snake = deque(snake) # deque for fast FIFO queue
        for (row,col) in self.snake:
            self.grid[row,col] = -1
        self.spawn_food()
        self.is_out = False
        self.is_bitten = False
        self.is_food_eaten = False
        self.game_over = False

    @property
    def head(self):
        return self.snake[-1]

    def move(self, row_inc, col_inc):
        (row, col) = self.head
        (row, col) = (row + row_inc, col + col_inc)
        if row<0 or row>=self.grid_size or col<0 or col>=self.grid_size:
            self.is_out = True
        elif self.grid[(row,col)] == -1:
            self.is_bitten = True
        else:
            self.snake.append((row, col))
            self.grid[(row, col)] = -1


        if self.head == self.food:
            self.is_food_eaten = True
        else:
            self.is_food_eaten = False

    def spawn_food(self):
        self.food = (randint(0,self.grid_size-1), randint(0,self.grid_size-1))
        # Spawn food again if it appeared in the snake
        if self.food in self.snake:
            self.spawn_food()
        self.grid[self.food] = 1

    def play(self, action):
        (row_inc, col_inc) = self.actions[action]
        self.move(row_inc, col_inc)
        if self.is_food_eaten:
            self.spawn_food()
            reward = self.rewards['food']
            if self.verbose: print('Food eaten :) - Reward:{}'.format(reward))
        elif self.is_bitten:
            reward = self.rewards['bitten']
            self.game_over = True
            if self.verbose: print('Tail bitten :( - Reward:{}'.format(reward))
        elif self.is_out:
            reward = self.rewards['out']
            self.game_over = True
            if self.verbose: print('Wall hit :( - Reward:{}'.format(reward))
        else:
            # Snake did not grow, pop end of tail
            self.grid[self.snake.popleft()] = 0
            reward =  self.rewards['nothing']
            if self.verbose: print('Nothing happened - Reward:{}'.format(reward))
        return reward

    def display(self, filename=None):
        if self.im:
            self.im.set_data(self.grid)
        else:
            plt.ion()
            plt.axis('off')
            self.im = plt.imshow(self.grid, interpolation='nearest')
        if filename:
            plt.savefig(filename, bbox_inches='tight')
        plt.pause(0.00001)

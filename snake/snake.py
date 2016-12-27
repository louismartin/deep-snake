import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from random import randint

class Snake:
    def __init__(self, grid_size=5, snake_length=4, verbose=0):
        self.grid_size = grid_size
        self.snake_length = snake_length
        self.verbose = verbose
        # TODO: add idle action ? (Check if performance increases)
        self.actions = {
            0: (0,-1), # Move left
            1: (-1,0), # Move up
            2: (0,+1), # Move right
            3: (+1,0)  # Move down
        }
        self.reset()

    def reset(self):
        # Position of the snake; leftmost element is the head,
        # the rest is the tail
        # Example: snake = [(0,0), (0,1), (0,2)]
        self.grid = np.zeros((self.grid_size, self.grid_size))
        snake = [(0,col) for col in range(self.snake_length)]
        self.snake = deque(snake) # deque for fast FIFO queue
        for (row,col) in self.snake:
            self.grid[row,col] = -1
        self.spawn_food()

    @property
    def head(self):
        return self.snake[-1]

    @property
    def tail(self): # TODO: might not be fast (for is_bitten)
        return list(self.snake)[:-1]

    def is_bitten(self):
        if self.head in self.tail:
            return True
        else:
            return False

    def move(self, row_inc, col_inc):
        (row, col) = self.head
        (row, col) = (row + row_inc, col + col_inc)
        if row<0 or row>=self.grid_size or col<0 or col>=self.grid_size:
            self.is_out = True
        else:
            self.snake.append((row, col))
            self.grid[(row, col)] = -1
            self.is_out = False

    def spawn_food(self):
        self.food = (randint(0,self.grid_size-1), randint(0,self.grid_size-1))
        # Spawn food again if it appeared in the snake
        if self.food in self.snake:
            self.spawn_food()
        self.grid[self.food] = 1

    def play(self, action):
        assert action in self.actions.keys()
        (row_inc, col_inc) = self.actions[action]
        self.move(row_inc, col_inc)
        if self.head == self.food:
            if self.verbose: print('Food eaten :)')
            self.spawn_food()
            reward = +20
            reset = False
        elif self.is_bitten():
            if self.verbose: print('Tail bitten :(')
            reward = -10
            reset = True
            self.reset()
        elif self.is_out:
            if self.verbose: print('Wall hit :(')
            reward = -10
            reset = True
            self.reset()
        else:
            # Snake did not grow, pop end of tail
            self.grid[self.snake.popleft()] = 0
            reward = -1
            reset = False
        return reward, reset

    def display(self):
        plt.ion()
        plt.imshow(self.grid, interpolation='nearest')
        plt.pause(0.1)

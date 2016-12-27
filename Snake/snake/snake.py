import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from random import randint

class Snake:
    def __init__(self):
        self.grid_size = 5
        self.snake_length = 3 # Snake length
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
        snake = [(0,col) for col in range(self.snake_length)]
        self.snake = deque(snake) # deque for fast FIFO queue
        self.spawn_food()

    @property
    def head(self):
        return self.snake[-1]

    @property
    def tail(self): # TODO: might not be fast (for is_bitten)
        return list(self.snake)[:-1]

    def is_out(self):
        (row,col) = self.head
        if row<0 or row>=self.grid_size or col<0 or col>=self.grid_size:
            return True
        else:
            return False

    def is_bitten(self):
        if self.head in self.tail:
            return True
        else:
            return False

    def move(self, row_inc, col_inc):
        (row, col) = self.head
        self.snake.append((row + row_inc, col + col_inc))

    def spawn_food(self):
        self.food = (randint(0,self.grid_size-1), randint(0,self.grid_size-1))
        # Spawn food again if it appeared in the snake
        if self.food in self.snake:
            self.spawn_food()

    def play(self, action):
        assert action in self.actions.keys()
        (row_inc, col_inc) = self.actions[action]
        self.move(row_inc, col_inc)
        if self.head == self.food:
            #print('Food eaten :)')
            self.spawn_food()
            reward = +10
            reset = False
        else:
            self.snake.popleft() # Snake did not grow, pop end of tail
            if self.is_out():
                #print('Wall hit :(')
                reward = -10
                reset = True
                self.reset()
            elif self.is_bitten():
                #print('Tail bitten :(')
                reward = -10
                reset = True
                self.reset()
            else:
                reward = -1
                reset = False
        return reward, reset

    def display(self, plot = 0):
        grid = np.zeros((self.grid_size, self.grid_size))
        for (row,col) in self.snake:
            grid[row,col] = -1
        grid[self.food] = 1 

        if plot:
            plt.ion()
            plt.imshow(grid, interpolation='nearest')
            plt.pause(0.1)
       
        return grid

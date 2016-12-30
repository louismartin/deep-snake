from snake import Snake
from timeit import timeit
import random
import cProfile
from tools import sample_from_policy

def test(snake):
    frames_stacked = []
    for i in range(100):
        snake.play(random.randint(0,3))
        frames_stacked.append(snake.grid)

#t = timeit(stmt='test(snake)', setup='from __main__ import test; from snake import Snake;snake=Snake();', number=1000))
#print(t)

def test_complete():
    for j in range(100):
        snake = Snake()

        frames_stacked = []
        for i in range(100):
            snake.play(random.randint(0,3))
            frames_stacked.append(snake.grid)

cProfile.run('for i in range(10000): sample_from_policy([0.1,0.2,0.3,0.4])')

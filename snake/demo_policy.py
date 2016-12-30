import time

from policy_gradient import train, test
from models.model_base import FullyConnected, ConvNet
from snake import Snake

# load THE SNAKE
snake = Snake(rewards={'nothing':-1, 'bitten':-10, 'out':-10, 'food':500})

# initialize the model
n_input = 2 * snake.grid_size * snake.grid_size
#model = FullyConnected(n_input=n_input, n_hidden=200, n_classes=4)
model = ConvNet(grid_size=snake.grid_size, n_frames=2, n_classes=4)

start_time = time.time()
train(model, snake, batch_size=200, n_iterations=2000, gamma=.95, learning_rate=0.001)
end_time = time.time()
print('Elapsed %.2f seconds' % (end_time - start_time))

test(model, snake)

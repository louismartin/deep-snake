import time

from policy_gradient import train, test
from models.model_base import FullyConnected, ConvNet
from snake import Snake

# load THE SNAKE
snake = Snake(rewards={'nothing':-10, 'bitten':-1, 'out':-1, 'food':10}, verbose=False)

# initialize the model
n_input = 2 * snake.grid_size * snake.grid_size
model = FullyConnected(n_input=n_input, n_hidden=200, n_classes=4)
#model = ConvNet(n_frames=2, n_classes=4, n_blocks=0)

start_time = time.time()
train(model, snake, batch_size=100, n_iterations=500, gamma=0.95, learning_rate=0.001, plot=True)
end_time = time.time()
print('Elapsed %.2f seconds' % (end_time - start_time))

test(model, snake)

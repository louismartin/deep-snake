from policy_gradient_fc import train, test
import time

start_time = time.time()
train(n_batch = 100, n_iterations = 1000, n_hidden = 200, gamma = .95, learning_rate = 0.001, rewards = {'nothing':-1, 'bitten':-10, 'out':-10, 'food':500}, settings = 'base')
end_time = time.time()
print('Elapsed %.2f seconds' % (end_time - start_time))
from policy_gradient_fc import train, test
import time

start_time = time.time()
train(n_batch = 100, n_iterations = 100, gamma = 1, learning_rate = 0.001, rewards = {'nothing':-1, 'bitten':-10, 'out':-10, 'food':11}, model_name = 'model')
end_time = time.time()
print('Elapsed %.2f seconds' % (end_time - start_time))

start_time = time.time()
train(n_batch = 100, n_iterations = 100, gamma = 1, learning_rate = 0.001, rewards = {'nothing':-1, 'bitten':-10, 'out':-10, 'food':11}, model_name = 'model')
end_time = time.time()
print('Elapsed %.2f seconds' % (end_time - start_time))

start_time = time.time()
train(n_batch = 100, n_iterations = 100, gamma = 1, learning_rate = 0.001, rewards = {'nothing':-1, 'bitten':-10, 'out':-10, 'food':11}, model_name = 'model')
end_time = time.time()
print('Elapsed %.2f seconds' % (end_time - start_time))

start_time = time.time()
train(n_batch = 100, n_iterations = 100, gamma = 1, learning_rate = 0.001, rewards = {'nothing':-1, 'bitten':-10, 'out':-10, 'food':11}, model_name = 'model')
end_time = time.time()
print('Elapsed %.2f seconds' % (end_time - start_time))

start_time = time.time()
train(n_batch = 100, n_iterations = 100, gamma = 1, learning_rate = 0.001, rewards = {'nothing':-1, 'bitten':-10, 'out':-10, 'food':11}, model_name = 'model')
end_time = time.time()
print('Elapsed %.2f seconds' % (end_time - start_time))




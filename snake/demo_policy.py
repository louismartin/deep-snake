from policy_gradient_fc import train, test
import time

start_time = time.time()
train(n_batch = 200, n_iterations = 100, n_hidden = 200, gamma = 1, learning_rate = 0.001, rewards = {'nothing':-1, 'bitten':-10, 'out':-10, 'food':11}, settings = 'base')
end_time = time.time()
print('Elapsed %.2f seconds' % (end_time - start_time))

#start_time = time.time()
#train(n_batch = 200, n_iterations = 5000, n_hidden = 200, gamma = .9, learning_rate = 0.001, rewards = {'nothing':-1, 'bitten':-10, 'out':-10, 'food':11}, settings = 'gamma')
#end_time = time.time()
#print('Elapsed %.2f seconds' % (end_time - start_time))
#
#start_time = time.time()
#train(n_batch = 200, n_iterations = 5000, n_hidden = 200, gamma = 1, learning_rate = 0.01, rewards = {'nothing':-1, 'bitten':-10, 'out':-10, 'food':11}, settings = 'lr')
#end_time = time.time()
#print('Elapsed %.2f seconds' % (end_time - start_time))
#
#start_time = time.time()
#train(n_batch = 400, n_iterations = 5000, n_hidden = 200, gamma = 1, learning_rate = 0.001, rewards = {'nothing':-1, 'bitten':-10, 'out':-10, 'food':11}, settings = 'batch')
#end_time = time.time()
#print('Elapsed %.2f seconds' % (end_time - start_time))
#
#start_time = time.time()
#train(n_batch = 200, n_iterations = 5000, n_hidden = 400, gamma = 1, learning_rate = 0.001, rewards = {'nothing':-1, 'bitten':-10, 'out':-10, 'food':100}, settings = 'hidden')
#end_time = time.time()
#print('Elapsed %.2f seconds' % (end_time - start_time))
#
#start_time = time.time()
#train(n_batch = 600, n_iterations = 5000, n_hidden = 200, gamma = 1, learning_rate = 0.001, rewards = {'nothing':-1, 'bitten':-10, 'out':-10, 'food':11}, settings = 'big_batch')
#end_time = time.time()
#print('Elapsed %.2f seconds' % (end_time - start_time))




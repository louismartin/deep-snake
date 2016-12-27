from policy_gradient_fc import train, test
import time

start_time = time.time()
weights = train(batch_size=100, n_iterations=100, learning_rate=0.001)
end_time = time.time()
print('Elapsed %.2f seconds' % (end_time - start_time))

test(weights)

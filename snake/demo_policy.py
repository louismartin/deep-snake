from policy_gradient_fc import train, test

weights = train(n_batch = 100, n_iterations = 100, learning_rate = 0.001)
test(weights)
import numpy as np
from numpy import random

def sample_from_policy(t):
    p = random.random()
    cdf = np.cumsum(t)
    return np.where(cdf >= p)[0][0]

def discount_rewards(rewards, gamma):
    discounted_sum = 0.
    for i in xrange(len(rewards)):
        discounted_sum *= gamma
        discounted_sum += rewards[i]
    return [discounted_sum] * len(rewards)

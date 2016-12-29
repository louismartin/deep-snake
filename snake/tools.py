import numpy as np
from numpy import random

def sample_from_policy(t):
    p = random.random()
    cdf = np.cumsum(t)
    return np.where(cdf >= p)[0][0]

def discount_rewards(rewards, gamma):
    rewards_new = np.zeros(len(rewards))
    discount_sum = 0
    for i in reversed(xrange(len(rewards))):
        discount_sum *= gamma
        discount_sum += rewards[i]
        rewards_new[i] = discount_sum
    return rewards_new

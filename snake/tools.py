import numpy as np
from numpy import random

def sample_from_policy(t):
    p = random.random()
    cdf = np.cumsum(t)
    return np.where(cdf >= p)[0][0]

import numpy as np

def get_reward(distance, threshold):
    return 2. / (1 + np.exp(- distance + threshold)) - 1.
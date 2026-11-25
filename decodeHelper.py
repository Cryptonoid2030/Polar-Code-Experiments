import numpy as np


def f(l1, l2):
    return np.sign(l1) * np.sign(l2) * np.minimum(np.abs(l1), np.abs(l2))

def g(l1, l2, s_left):
    # s_left is known bit (left) to predict right; enhanced bit
    return l2 + (1 - 2 * s_left) * l1
import numpy as np


# returns the indices of the frozen and send bits based on send amount and good bits
def Bhattacharyya(N, K, p_design):
    """
    Pick the N-K least reliable subchannels (Bhattacharyya bound) for a BSC
    with crossover probability p_design. Use ONE fixed p_design for a whole
    FER sweep so the code does not change from point to point.
    """
    z = np.array([2.0 * np.sqrt(p_design * (1.0 - p_design))])
    while len(z) < N:
        y = z - z**2
        new_z = np.empty(2 * len(z))
        new_z[0::2] = np.clip(z + y, 0, 1)  # degraded subchannel
        new_z[1::2] = z - y                 # upgraded subchannel
        z = new_z
    ranked = np.argsort(z)
    return np.sort(ranked[K:])
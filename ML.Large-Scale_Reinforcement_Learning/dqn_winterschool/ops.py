import numpy as np

def one_hot_encode(states, num_states):
    """

    :param states: list or 1D numpy array, ie: [1,2,5,7]
    :param num_states: integer; number of states
    :return: numpy 2D array using one-hot encoding.
    """

    num_encoding = len(states)
    encoding = np.zeros([num_encoding, num_states])
    encoding[np.arange(num_encoding), states] = 1

    return encoding
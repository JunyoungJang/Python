import numpy as np


class ENVIRONMENT:
    def __init__(self):
        
        " States and Actions "
        self._states = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self._actions = [0, 1, 2, 3]
        self._num_state = len(self._states)
        self._num_action = len(self._actions)

        # "state 3" : (terminal-state) +1 reward
        # "state 6" : (terminal-state) -1 reward
        self._terminal_state = [3, 6]

        " Transition self._Probabilities "
        self._P = np.zeros((self._num_state, self._num_action, self._num_state))

        self._P[0, 0, :] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self._P[0, 1, :] = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self._P[0, 2, :] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self._P[0, 3, :] = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

        self._P[1, 0, :] = [0.9, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0]
        self._P[1, 1, :] = [0, 0, 0.9, 0, 0, 0.1, 0, 0, 0, 0, 0]
        self._P[1, 2, :] = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self._P[1, 3, :] = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        self._P[2, 0, :] = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self._P[2, 1, :] = [0, 0, 0, 0.9, 0, 0, 0.1, 0, 0, 0, 0]
        self._P[2, 2, :] = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        self._P[2, 3, :] = [0, 0, 0, 0, 0, 0.9, 0.1, 0, 0, 0, 0]

        self._P[3, 0, :] = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        self._P[3, 1, :] = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        self._P[3, 2, :] = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        self._P[3, 3, :] = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

        self._P[4, 0, :] = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        self._P[4, 1, :] = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        self._P[4, 2, :] = [0.9, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self._P[4, 3, :] = [0, 0, 0, 0, 0, 0, 0, 0.9, 0.1, 0, 0]

        self._P[5, 0, :] = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        self._P[5, 1, :] = [0, 0, 0, 0.1, 0, 0, 0.8, 0, 0, 0, 0.1]
        self._P[5, 2, :] = [0, 0.1, 0.8, 0.1, 0, 0, 0, 0, 0, 0, 0]
        self._P[5, 3, :] = [0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0.8, 0.1]

        self._P[6, 0, :] = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        self._P[6, 1, :] = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        self._P[6, 2, :] = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        self._P[6, 3, :] = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

        self._P[7, 0, :] = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        self._P[7, 1, :] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        self._P[7, 2, :] = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        self._P[7, 3, :] = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]

        self._P[8, 0, :] = [0, 0, 0, 0, 0.1, 0, 0, 0.9, 0, 0, 0]
        self._P[8, 1, :] = [0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0.9, 0]
        self._P[8, 2, :] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        self._P[8, 3, :] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]

        self._P[9, 0, :] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        self._P[9, 1, :] = [0, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0.9]
        self._P[9, 2, :] = [0, 0, 0, 0, 0, 0.9, 0.1, 0, 0, 0, 0]
        self._P[9, 3, :] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]

        self._P[10, 0, :] = [0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0.9, 0]
        self._P[10, 1, :] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        self._P[10, 2, :] = [0, 0, 0, 0, 0, 0.1, 0.9, 0, 0, 0, 0]
        self._P[10, 3, :] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

        " Current State"
        self._current_state = None
        
    def reset(self):
        # Terminal-state cannot be an initial state.
        initial_state = np.random.choice([0, 1, 2, 4, 5, 7, 8, 9, 10])
        self._current_state = initial_state
        return initial_state

    def step(self, action):
        if self._current_state is None:
            raise ValueError("Current State must be initialized")
        if self._current_state in self._terminal_state:
            raise ValueError("Current State is at the terminal state")

        action = int(action)
        reward = -0.02
        done = False
        state = np.random.choice(self._num_state, 1, p=self._P[self._current_state, action])[0]
        if state == self._terminal_state[0]:
            reward += 1.0
            done = True
        if state == self._terminal_state[1]:
            reward += -1.0
            done = True

        self._current_state = state

        return state, reward, done

    def action_sample(self):
        return np.random.randint(0, self._num_action)

    @property
    def num_state(self):
        return self._num_state

    @property
    def num_action(self):
        return self._num_action

    @property
    def win_state(self):
        return self._terminal_state[0]

    @property
    def lose_state(self):
        return self._terminal_state[1]



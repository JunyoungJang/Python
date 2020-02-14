import numpy as np
import random
import matplotlib.pyplot as plt
from copy import deepcopy
from collections import deque, namedtuple
import logging

from env import ENVIRONMENT
import ops

logging.basicConfig(level=logging.DEBUG, format='%(message)s')

""" Hyper Parameters """
gamma = 0.99
max_episodes = 10000
epsilon = 0.10
memory_size = 1000
num_burning_episode = 100
batch_size = 32
learning_rate = 0.01
copy_period = 100
test_period = 100
epsilon_test = 0.00

""" Environment """
env = ENVIRONMENT()
exp = namedtuple('experience', ['state', 'action', 'reward', 'next_state', 'done'])

""" Q-net and target Q-net """
W = np.random.uniform(0.0, 1.0, [env.num_state, env.num_action])
W_target = deepcopy(W)

""" Replay Memory"""
replay_memory = deque(maxlen=memory_size)

""" Burning Period """
logging.info('Initial Burning Period')
for _ in range(num_burning_episode):
    state = env.reset()
    while(1):
        action = env.action_sample()
        next_state, reward, done = env.step(action)
        sample = exp(state=state, action=action, reward=reward, next_state=next_state, done=done)
        replay_memory.append(sample)

        if done:
            break
        state = next_state

""" Training """
logging.info('Training Start')
train_step = 0
cum_return_test = []
step_history = []
for episode_num in range(max_episodes):
    # Reset Environment and Reset Cum. Reward
    state = env.reset()
    done = False

    learning_rate = 1.0 / ((episode_num * 0.1) + 1.0)
    epsilon = 1. / ((episode_num * 0.1) + 1.0)

    while(1):
        # Action Selection
        action_seed = np.random.sample()
        if action_seed > epsilon:
            action = np.argmax(np.matmul(ops.one_hot_encode([state], env.num_state), W))
        else:
            action = env.action_sample()
        next_state, reward, done = env.step(action)

        sample = exp(state=state, action=action, reward=reward, next_state=next_state, done=done)
        replay_memory.append(sample)
        state = next_state

        # Sample scenarios from replay-memory.
        samples = random.sample(replay_memory, batch_size)

        states_ = [sample.state for sample in samples]
        actions_ = [sample.action for sample in samples]
        rewards_ = [sample.reward for sample in samples]
        next_states_ = [sample.next_state for sample in samples]
        dones_ = [float(sample.done) for sample in samples]

        states = ops.one_hot_encode(states_, env.num_state)
        actions = ops.one_hot_encode(actions_, env.num_action)
        rewards = np.reshape(np.array(rewards_, dtype=np.float32), [-1, 1])
        next_states = ops.one_hot_encode(next_states_, env.num_state)
        dones = np.reshape(np.array(dones_, dtype=np.float32), [-1, 1])

        # Make target for Q-net.
        q_target_max = np.amax(np.matmul(next_states, W_target), axis=1, keepdims=True)
        targets = (1.0 - dones) * (rewards + gamma * q_target_max) + dones * rewards

        # Weight Update
        q_val = np.sum(np.matmul(states, W) * actions, axis=1, keepdims=True)
        W[states_, actions_] += learning_rate * np.reshape((targets - q_val), [-1])

        train_step += 1

        # Network Synchronization
        if train_step % copy_period == 0:
            W_target = deepcopy(W)

        # Terminal of Scenario.
        if done:
            if state == env.win_state:
                step_history.append(1)
            if state == env.lose_state:
                step_history.append(0)
            break

    if (episode_num + 1) % 100 == 0:
        logging.info('Episode  %d' % (episode_num+1))

    """ Test """
    if episode_num % test_period == 0:
        state = env.reset()
        done = False
        cum_rwd = 0.0

        while(1):
            # Action Selection
            action_seed = np.random.sample()
            if action_seed > epsilon_test:
                action = np.argmax(np.matmul(ops.one_hot_encode([state], env.num_state), W))
            else:
                action = env.action_sample()

            next_state, reward, done = env.step(action)
            cum_rwd += reward
            if done:
                break
            if cum_rwd < -10.0:
                break

            state = next_state

        cum_return_test.append(cum_rwd)

fig = plt.figure(1)
history = np.cumsum(step_history) / (np.arange(max_episodes) + 1)
plt.plot(history)
plt.show()

fig = plt.figure(2)
plt.plot(np.arange(len(cum_return_test)) * test_period, cum_return_test)
plt.xlabel('episode.')
plt.ylabel('cum. reward')
plt.show()
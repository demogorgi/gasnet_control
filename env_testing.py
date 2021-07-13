import abc
import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

import network_environment

# define environment
environment = network_environment.GasNetworkEnv(10)
# test its functionality as a python environment
# utils.validate_py_environment(environment)

# wrap the python environment into a tf environment
tf_env = tf_py_environment.TFPyEnvironment(environment)

# test 5 episodes of the environment
time_step = tf_env.reset()
rewards = []
steps = []
num_episodes = 5

for _ in range(num_episodes):
  episode_reward = 0
  episode_steps = 0
  while not time_step.is_last():
    valve1 = np.random.randint(0, 2)
    valve2 = np.random.randint(0, 2)
    compressor = np.random.randint(0, 11)

    action = np.array([[valve1, valve2, 0, compressor]], dtype=np.int32)
    time_step = tf_env.step(action)
    episode_steps += 1
    episode_reward += time_step.reward.numpy()
  rewards.append(episode_reward)
  steps.append(episode_steps)
  time_step = tf_env.reset()

num_steps = np.sum(steps)
avg_length = np.mean(steps)
avg_reward = np.mean(rewards)

print('num_episodes:', num_episodes, 'num_steps:', num_steps)
print('avg_length', avg_length, 'avg_reward:', avg_reward)

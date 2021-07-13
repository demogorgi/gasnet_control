from __future__ import absolute_import, division, print_function

import base64
import os
import reverb
import tempfile

import numpy as np
import tensorflow as tf

from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.environments import suite_pybullet
from tf_agents.metrics import py_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.policies import greedy_policy
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_py_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.train import triggers
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import strategy_utils
from tf_agents.train.utils import train_utils

from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.metrics import py_metrics
from tf_agents.networks import sequential
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
import gc

import network_environment

num_iterations = 100000 # @param {type:"integer"}

initial_collect_steps = 10000 # @param {type:"integer"}
collect_steps_per_iteration = 1 # @param {type:"integer"}
replay_buffer_capacity = 10000 # @param {type:"integer"}

batch_size = 256 # @param {type:"integer"}

critic_learning_rate = 3e-4 # @param {type:"number"}
actor_learning_rate = 3e-4 # @param {type:"number"}
alpha_learning_rate = 3e-4 # @param {type:"number"}
target_update_tau = 0.005 # @param {type:"number"}
target_update_period = 1 # @param {type:"number"}
gamma = 0.99 # @param {type:"number"}
reward_scale_factor = 1.0 # @param {type:"number"}

actor_fc_layer_params = (256, 256)
critic_joint_fc_layer_params = (256, 256)

log_interval = 5000 # @param {type:"integer"}

num_eval_episodes = 20 # @param {type:"integer"}
eval_interval = 10000 # @param {type:"integer"}

policy_save_interval = 5000 # @param {type:"integer"}


################# FUNCTION TESTING PROCEDURE ##################################
environment = network_environment.GasNetworkEnv(10)
tf_env = tf_py_environment.TFPyEnvironment(environment)

# test the basic functions
tf_env.reset()
print("Observation Spec:")
print(tf_env.time_step_spec().observation)
print("Reward Spec:")
print(tf_env.time_step_spec().reward)
print("Action Spec:")
print(tf_env.action_spec())

time_step = tf_env.reset()
print("Time step:")
print(time_step)

action = np.array([[1, 1, 0, 3]], dtype=np.int32)

next_time_step = tf_env.step(action)
print("Next time step:")
print(next_time_step)

#####################AGENT TRAINING PROCEDURE##################################
# initialize a training and an evaluation environment
collect_py_environment = network_environment.GasNetworkEnv(10)
eval_py_environment = network_environment.GasNetworkEnv(10)
# wrap them into tf environments
collect_environment = tf_py_environment.TFPyEnvironment(collect_py_environment)
eval_environment = tf_py_environment.TFPyEnvironment(eval_py_environment)

# define the Q network
# fc_layer_params = (100, 50)
# action_tensor_spec = tensor_spec.from_spec(environment.action_spec())
# num_actions = len(action_tensor_spec.maximum) #action_tensor_spec.maximum - action_tensor_spec.minimum + 1
#
#
# # helper function to create Dense Layers configured with the right
# # activation and kernel initializer
# # TODO: check if suitable activation function
# def dense_layer(num_units):
#     return tf.keras.layers.Dense(
#         num_units,
#         activation=tf.keras.activations.relu,
#         kernel_initializer=tf.keras.initializers.VarianceScaling(
#             scale=0.2, mode='fan_in', distribution='truncated_normal'
#         )
#     )
#
#
# # QNetwork = Sequence of Layers + q value layer with amount outputs = amount
# # actions
# # TODO: check values for initializers
# dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
# q_values_layer = tf.keras.layers.Dense(
#     num_actions,
#     activation=None,
#     kernel_initializer=tf.keras.initializers.RandomUniform(
#         minval=-0.03, maxval=0.03
#     ),
#     bias_initializer=tf.keras.initializers.Constant(-0.2)
# )
# q_network = sequential.Sequential(dense_layers + [q_values_layer])

# TODO: clarify differences between the optimizers
# # define the DQN Agent NOT USABLE SINCE ONLY 1-D ACTIONS ALLOWED
#
# optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
#
# train_step_counter = tf.Variable(0)
#
# agent = dqn_agent.DqnAgent(
#     train_environment.time_step_spec(),
#     train_environment.action_spec(),
#     q_network=q_network,
#     optimizer=optimizer,
#     td_errors_loss_fn=common.element_wise_squared_loss,
#     train_step_counter=train_step_counter
# )
#
# agent.initialize()




## define REINFORCE agent POSSIBLY NOT USEABLE DUE TO NEED OF SAME DIMENSIONS
## FOR ALL ACTIONS TODO: maybe only dependent on actorDistNetwork
# fc_layer_params = (100,)
# # define actor network
# actor_net = actor_distribution_network.ActorDistributionNetwork(
#     train_environment.observation_spec(),
#     train_environment.action_spec(),
#     fc_layer_params=fc_layer_params
# )
#
# optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
#
# train_step_counter = tf.compat.v2.Variable(0)
#
# tf_agent = reinforce_agent.ReinforceAgent(
#     train_environment.time_step_spec(),
#     train_environment.action_spec(),
#     actor_network=actor_net,
#     optimizer=optimizer,
#     normalize_returns=True,
#     train_step_counter=train_step_counter
# )
#
# tf_agent.initialize()

# define the specificities
observation_spec, action_spec, time_step_spec = (
    spec_utils.get_tensor_specs(collect_environment)
)

# define the critic network, TODO: check the kernel initializers for func
critic_net = critic_network.CriticNetwork(
    (observation_spec, action_spec),
    observation_fc_layer_params=None,
    action_fc_layer_params=None,
    joint_fc_layer_params=critic_joint_fc_layer_params,
    kernel_initializer='glorot_uniform',
    last_kernel_initializer='glorot_uniform'
)

#define actor network
actor_net = actor_distribution_network.ActorDistributionNetwork(
    observation_spec,
    action_spec,
    fc_layer_params=actor_fc_layer_params,
    continuous_projection_net=(
        tanh_normal_projection_network.TanhNormalProjectionNetwork
    )
)

# training
train_step = train_utils.create_train_step()




import tensorflow as tf
import os
import network_environment

from tf_agents.environments import tf_py_environment

# hyperparameters
num_eval_agent_steps = 4

simulations_per_agent_step = 8

# evaluation procedure
# import the policy accomplished through training
temp_dir = os.getcwd() + '/instances'
policy_dir = os.path.join(temp_dir, 'policy')
trained_policy = tf.compat.v2.saved_model.load(policy_dir)

# define the environment with nominations from file
eval_py_env = network_environment.GasNetworkEnv(
    discretization_steps=10,
    convert_action=True,
    steps_per_agent_step=simulations_per_agent_step,
    max_agent_steps=num_eval_agent_steps,
    random_nominations=False,
    print_actions=True
)

eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

# iterate the amount of steps
for _ in range(num_eval_agent_steps):

    time_step = eval_env.reset()
    if not time_step.is_last():
        action_step = trained_policy.action(time_step)
        time_step = eval_env.step(action_step.action)







import tensorflow as tf
import os
import network_environment
import time
import evaluation_network_environment

from tf_agents.environments import tf_py_environment

# hyperparameters
num_eval_agent_steps = 10

simulations_per_agent_step = 8

# evaluation procedure
# import the policy accomplished through training
temp_dir = os.getcwd() + '/instances/da2/policies/400kiters_allrandom/'
policy_dir = os.path.join(temp_dir, "policy_" +\
                          f"cdqn_(22, 32, 42)realQ_"
                          f"iters{400}_" +\
                          f"rate1e-2to1e-05_" +\
                          f"clip{None}_" +\
                          f"update{500}_" +\
                          f"epsilondecay10to0001_sigmoid")
                          #f"boltzmann{0.1}")
trained_policy = tf.compat.v2.saved_model.load(policy_dir)

# define the environment with nominations from file
# eval_py_env = network_environment.GasNetworkEnv(
#     discretization_steps=10,
#     convert_action=True,
#     steps_per_agent_step=simulations_per_agent_step,
#     max_agent_steps=num_eval_agent_steps,
#     random_nominations=True,
#     print_actions=True
# )
eval_py_env = evaluation_network_environment.GasNetworkEnv(
    discretization_steps=10,
    convert_action=True,
    steps_per_agent_step=simulations_per_agent_step,
    max_agent_steps=num_eval_agent_steps,
    random_nominations=False,
    print_actions=True,
    decision_string="evaluation_scenario.yml"
)

eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

# iterate the amount of steps
time_step = eval_env.reset()
step = 0
for _ in range(num_eval_agent_steps):
    #time.sleep(5)
    print("#"*15 + f"Evaluation of step {step}" + "#"*15)
    if not time_step.is_last():
        action_step = trained_policy.action(time_step)
        time_step = eval_env.step(action_step.action)
    print("#"*15 + f"End of evaluation of step {step}" + "#"*9 + "\n\n")
    step += 1





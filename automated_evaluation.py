
import tensorflow as tf
import os
import evaluation_network_environment
import time
import numpy as np

from tf_agents.environments import tf_py_environment

# hyperparameters
num_eval_agent_steps = 36

simulations_per_agent_step = 8

reset_files = True

# evaluation procedure
# import the policy accomplished through training
temp_dir = os.getcwd() + '/instances/da2/policies_automated_testing'

if reset_files:
    with open("policy_results.csv", "w") as policy_results:
        policy_results.write("net;iterations;lr start;lr end;update steps;"
                             "eps start;eps end;reward sum;reward avg;"
                             "activation\n")
    with open("policy_table.txt", "w") as policy_table:
        table_header = ["net\t\t\t\t", "|iterations\t", "|lr start\t",
                        "|lr end\t", "|update steps\t", "|eps start\t",
                        "|eps end\t", "|reward sum\t", "|reward avg\t",
                        "|activation\t"]
        header_widths = []
        for header in table_header:
            policy_table.write(header)
            if len(header.replace("\t", "")) % 4 == 0:
                header_widths.append(len(header.replace("\t", "")) + header.count("\t") * 4)
            else:
                header_widths.append(np.ceil(len(header.replace("\t", ""))/4) * 4 + (header.count("\t") - 1) * 4)
        policy_table.write("\n")
        width = sum(header_widths)
        policy_table.write("-" * int(width))
        policy_table.write("\n")
    policy_table.close()
    policy_results.close()
# for each file in temp_dir do evaluation
for policy_name in os.listdir(temp_dir):
    policy_dir = temp_dir + "/" + policy_name
    trained_policy = tf.compat.v2.saved_model.load(policy_dir)
    eval_py_env = evaluation_network_environment.GasNetworkEnv(
        discretization_steps=10,
        convert_action=True,
        steps_per_agent_step=simulations_per_agent_step,
        max_agent_steps=num_eval_agent_steps,
        random_nominations=False,
        print_actions=False
    )

    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    time_step = eval_env.reset()
    step = 0
    for _ in range(num_eval_agent_steps):
        if not time_step.is_last():
            action_step = trained_policy.action(time_step)
            time_step = eval_env.step(action_step.action)

    with open("rewardfile.csv", "r") as rewardfile:
        reward_string = rewardfile.read()
    rewardfile.close()
    rewards = []
    for reward in reward_string.split(";"):
        if reward != '':
            rewards += [float(reward)]

    assert(len(rewards) == num_eval_agent_steps)

    policy_attributes = policy_name.split("_")
    network = policy_attributes[2][:-5]
    iterations = int(policy_attributes[3][5:])
    learning_rate_start = float(policy_attributes[4][4:8])
    learning_rate_end = float(policy_attributes[4][10:])
    update_steps = int(policy_attributes[6][6:])
    epsilon_start = policy_attributes[7][12:].split("to")[0]
    epsilon_start = float(epsilon_start[0] + "." + epsilon_start[1:])
    epsilon_end = policy_attributes[7].split("to")[1]
    epsilon_end = float(epsilon_end[0] + "." + epsilon_end[1:])
    reward_sum = np.round(sum(rewards), 2)
    reward_avg = np.round(reward_sum/num_eval_agent_steps, 3)
    activation = "tanh" if "tanh" in policy_name else "sigmoid"
    policy_attributes = [network, iterations, learning_rate_start,
                         learning_rate_end, update_steps, epsilon_start,
                         epsilon_end, reward_sum, reward_avg, activation]

    with open("policy_results.csv", "a+") as policy_results:
        policy_results.write(network + ";" + str(iterations) + ";" +
                             str(learning_rate_start) + ";" +
                             str(learning_rate_end) +
                             ";" + str(update_steps) + ";" +
                             str(epsilon_start) + ";" +
                             str(epsilon_end) + ";" + str(np.round(reward_sum, 2)) + ";" +
                             str(np.round(reward_avg, 3)) + ";" +
                             str(activation) + "\n")
    with open("policy_table.txt", "a+") as policy_table:
        for (index, attribute) in enumerate(policy_attributes):
            tab_len = header_widths[index] - len(str(attribute)) - 1
            n_tabs = int(np.ceil(tab_len/4))
            policy_table.write(str(attribute) + "\t"*n_tabs + "|")
        policy_table.write("\n")
    policy_results.close()
    policy_table.close()
    # save overall reward and average reward per policy as table and csv





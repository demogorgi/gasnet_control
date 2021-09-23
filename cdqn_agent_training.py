from __future__ import absolute_import, division, print_function

import base64
import os
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time

import tensorflow as tf

from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import categorical_q_network
from tf_agents.policies import random_tf_policy
from tf_agents.policies import policy_saver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common

import network_environment

# hyper-parameters
on_cluster = True

if len(sys.argv) > 4:
    in_target_update_steps_options = [int(steps) for steps in
                                      sys.argv[4].split("-")]
else:
    in_target_update_steps_options = [1]  # 100, 250, 400, 550, 700, 850,

if len(sys.argv) > 5:
    in_start_epsilon = float(sys.argv[5])
else:
    in_start_epsilon = 0.1

if len(sys.argv) > 6:
    in_gradient_clippings = [float(sys.argv[6]) if sys.argv[6] != 'None'
                             else None]  # dont forget None value
else:
    in_gradient_clippings = [None]  # dont forget None value

if len(sys.argv) > 7:
    if 'to' in sys.argv[7]:
        in_out_rates = sys.argv[7].split('to')
        in_learning_rates = [float(in_out_rates[0])]
        in_end_learning_rate = float(in_out_rates[1])
        learning_rate_decay = True
    else:
        in_learning_rates = [float(sys.argv[7])]
        in_end_learning_rate = 1e-5
        learning_rate_decay = False
else:
    in_learning_rates = [1e-2]
    in_end_learning_rate = 1e-5
    learning_rate_decay = False

if len(sys.argv) > 8:
    layers = sys.argv[8].split("-")
    if len(layers) == 1:
        fc_layer_param = (int(layers[0]),)
    elif len(sys.argv[8].split("-")) == 2:
        layer1 = int(layers[0])
        layer2 = int(layers[1])
        fc_layer_param = (layer1,layer2)
    else:
        layer1 = int(layers[0])
        layer2 = int(layers[1])
        layer3 = int(layers[2])
        fc_layer_param = (layer1,layer2,layer3)
else:
    fc_layer_param = (50,)

if len(sys.argv) > 9:
    if sys.argv[9] == 'T':
        decaying_epsilon = True
    else:
        decaying_epsilon = False
else:
    decaying_epsilon = False

if len(sys.argv) > 10:
    in_end_epsilons = [float(sys.argv[10])]
else:
    in_end_epsilons = [0.001]

if len(sys.argv) > 11:
    no_run = int(sys.argv[11])
else:
    no_run = -1


in_num_iterations_options = [200_000]#[200000]#[5000, 20000, 50000]
in_boltzmann_temperatures = []
# in_target_update_steps_options = [5000] #100, 250, 400, 550, 700, 850, 1000


def cdqn_agent_training(
        in_num_iterations=20000,
        in_learning_rate=1e-5,
        in_end_learning_rate=1e-5,
        in_start_epsilon=0.1,
        in_end_epsilon=1e-4,
        in_use_epsilon=False,
        in_boltzmann_temperatur=1.0,
        in_target_update_steps=200,
        in_gradient_clipping=1.0,
        in_show_plot=True
):
    num_iterations = in_num_iterations    # @param {type:"integer"}

    initial_collect_steps = 1000  # before:100  # @param {type:"integer"}
    collect_steps_per_iteration = 1  # @param {type:"integer"}
    replay_buffer_max_length = 100000  # @param {type:"integer"}

    global_step = tf.compat.v1.train.get_or_create_global_step()
    batch_size = 64  # @param {type:"integer"}
    # if required define a decaying learning rate
    if learning_rate_decay:
        learning_rate = tf.compat.v1.train.polynomial_decay(
            learning_rate=in_learning_rate,
            global_step=global_step,
            decay_steps=in_num_iterations,
            end_learning_rate=in_end_learning_rate
        )
    else:
        learning_rate = in_learning_rate  # @param {type:"number"}
    log_interval = 200  # @param {type:"integer"}

    num_eval_episodes = 10  # @param {type:"integer"}
    eval_interval = 1000  # @param {type:"integer"}

    # new parameters for cdqn
    num_atoms = 51
    min_q_value = -6
    max_q_value = 6
    n_step_update = 2

    # define a decaying epsilon over time, a boltzmann and which to use
    start_epsilon = in_start_epsilon
    end_epsilon = in_end_epsilon
    if decaying_epsilon:
        epsilon = tf.compat.v1.train.polynomial_decay(
            start_epsilon,
            global_step,
            num_iterations,
            end_learning_rate=end_epsilon
        )
    else:
        epsilon = start_epsilon

    boltzmann_temperatur = in_boltzmann_temperatur
    use_epsilon = in_use_epsilon

    # update steps for the new definition of the q target network
    target_update_steps = in_target_update_steps    # @param {type:"integer"}
    # norm to clip the gradient to for gradient descent
    gradient_clipping = in_gradient_clipping     # @param {type:"float"}

    tf.compat.v1.enable_v2_behavior()

    if not on_cluster:
        temp_dir = os.getcwd() + '/instances'

    # custom hyperoarameters

    max_agent_steps = 6 # @param {type:"integer"}
    steps_per_agent_step = 8    # @param {type:"integer"}
    discretization = 10 # @param {type:"integer"}

    convert_action = True   # @param {type:"boolean"}
    random_entry_nominations = False   # @param {type:"boolean"}

    show_plot = in_show_plot

    ########## TRAINING SECTION ##########
    train_py_env = network_environment. \
        GasNetworkEnv(discretization_steps=discretization,
                      convert_action=convert_action,
                      steps_per_agent_step=steps_per_agent_step,
                      max_agent_steps=max_agent_steps,
                      random_nominations=random_entry_nominations)
    eval_py_env = network_environment. \
        GasNetworkEnv(discretization_steps=discretization,
                      convert_action=convert_action,
                      steps_per_agent_step=steps_per_agent_step,
                      max_agent_steps=max_agent_steps,
                      random_nominations=random_entry_nominations)

    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    categorical_q_net = categorical_q_network.CategoricalQNetwork(
        input_tensor_spec=train_env.observation_spec(),
        action_spec=train_env.action_spec(),
        num_atoms=num_atoms,
        fc_layer_params=fc_layer_param,
        activation_fn=tf.nn.tanh
    )

    # instantiate dqn agent
    #optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate,
    #                                    momentum=1.0)
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

    agent = categorical_dqn_agent.CategoricalDqnAgent(
        time_step_spec=train_env.time_step_spec(),
        action_spec=train_env.action_spec(),
        categorical_q_network=categorical_q_net,
        optimizer=optimizer,
        min_q_value=min_q_value,
        max_q_value=max_q_value,
        n_step_update=n_step_update,
        td_errors_loss_fn=common.element_wise_squared_loss,
        gradient_clipping=gradient_clipping,
        target_update_period=target_update_steps,
        train_step_counter=global_step,
        epsilon_greedy=epsilon if use_epsilon else None,
        boltzmann_temperature=None if use_epsilon else boltzmann_temperatur
    )

    agent.initialize()

    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                    train_env.action_spec())

    def compute_avg_return(environment, policy, num_episodes=10):

        total_return_local = 0.0
        for _ in range(num_episodes):

            time_step = environment.reset()
            episode_return_local = 0.0

            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = environment.step(action_step.action)
                episode_return_local += time_step.reward
            total_return_local += episode_return_local

        avg_return_local = total_return_local / num_episodes

        if avg_return_local > max_agent_steps*0.9 and not on_cluster:
            time_step = environment.reset()
            while not time_step.is_last():
                action_step = policy.action(time_step)
                print(f"PROMISING action via {action_step.action}")
                time_step = environment.step(action_step.action)
        return avg_return_local.numpy()[0]

    compute_avg_return(eval_env, random_policy, num_eval_episodes)

    # initialize replay buffer
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_max_length
    )

    # define and test data collection
    def collect_step(environment, policy, buffer):
        time_step = environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step,
                                          next_time_step)

        # add trajectory to the replay buffer
        buffer.add_batch(traj)

    def collect_data(environment, policy, buffer, steps):
        for _ in range(steps):
            collect_step(environment, policy, buffer)

    collect_data(train_env, random_policy, replay_buffer,
                 initial_collect_steps)

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=batch_size,
        num_steps=n_step_update + 1
    ).prefetch(3)

    iterator = iter(dataset)

    ##### actual training procedure
    agent.train = common.function(agent.train)

    # reset the training step
    agent.train_step_counter.assign(0)

    # evaluate the agents policy once before training
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    returns = [avg_return]
    losses = []

    # define the file name specificities
    file_spec = f"cdqn_{fc_layer_param}realQ_" +\
                f"iters{int(num_iterations/1000)}_" +\
                f"rate{'{:.0e}'.format(in_learning_rate).replace('0', '')}"
    if learning_rate_decay:
        file_spec += f"to{in_end_learning_rate}_"
    else:
        file_spec += "_"
    if gradient_clipping is not None:
        file_spec += f"clip{int(gradient_clipping)}_"
    else:
        file_spec += "clipNone_"

    file_spec += f"update{target_update_steps}_"
    if use_epsilon:
        if decaying_epsilon:
            file_spec += f"epsilondecay{str(start_epsilon).replace('.', '')}"
            file_spec += f"to{str(end_epsilon).replace('.', '')}"
        else:
            file_spec += f"epsilon{str(start_epsilon).replace('.', '')}"
    else:
        file_spec += f"boltzmann{boltzmann_temperatur}"

    if no_run >= 0:
        file_spec += f"_run{no_run}"

    # initialize the necessary variables for saving the policy for later use
    policy_name = f"policy_" + file_spec
    if on_cluster:
        policy_dir = f"/home/hpc/mpwm/mpwm023h/masterthesis/policies/" \
                     + policy_name
    else:
        policy_dir = os.path.join(temp_dir, policy_name)
    tf_policy_saver = policy_saver.PolicySaver(agent.policy)
    start_time = time.time()

    # train
    for _ in range(num_iterations):
        # collect a few steps using collect_policy & save to the replay buffer
        collect_data(train_env, agent.collect_policy, replay_buffer,
                     collect_steps_per_iteration)

        # sample a batch of data from the buffer and update the agent's network
        experience, unused_info = next(iterator)
        train_loss = agent.train(experience).loss

        step = agent.train_step_counter.numpy()

        if step % log_interval == 0:
            if not on_cluster:
                print('step = {0}: loss = {1}'.format(step, train_loss))
            losses += [train_loss]

        if step % eval_interval == 0:
            avg_return = compute_avg_return(eval_env, agent.policy,
                                            num_eval_episodes)
            if not on_cluster:
                print('step = {0}: Average Return = {1}'.format(step,
                                                                avg_return))
            returns.append(avg_return)

    if on_cluster:
        try:
            out_path = f'/home/hpc/mpwm/mpwm023h/masterthesis/outfiles/'
            out_file_name = "loss_reward_" + file_spec + ".out"
            with open(out_path + out_file_name, 'a+') as out_file:
                out_file.write("returns:\n")
                for ret in returns:
                    out_file.write(str(ret) + ",")
                out_file.write("\nlosses:\n")
                for loss in losses:
                    out_file.write(str(loss) + ",")
        except:
            print("outfile didn't work")

    # save the policy
    tf_policy_saver.save(policy_dir)

    # print the result training
    iterations = range(0, num_iterations + 1, eval_interval)
    plt.figure()
    plt.plot(iterations, returns)
    plt.hlines(max_agent_steps*0.95, iterations[0], iterations[-1], 'r')
    plt.xlabel('Iterations')
    plt.ylabel('Average Return')
    if on_cluster:
        fig_path = f"/home/hpc/mpwm/mpwm023h/masterthesis/plots/"
    else:
        fig_path = f"/home/adi/Uni/SoSe21/Masterarbeit/" +\
                          f"reward_exppress_contflow/"
    plt.savefig(fig_path + "reward_" + file_spec + ".pdf")
    if show_plot:
        plt.show()

    iterations = range(0, num_iterations, log_interval)
    plt.figure()
    plt.plot(iterations, losses)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.savefig(fig_path + "loss_" + file_spec + ".pdf")
    if show_plot:
        plt.show()

    ##### evaluation #####
    total_return = 0.0
    max_return = max_agent_steps * num_eval_episodes
    for _ in range(num_eval_episodes):
        eval_time_step = eval_env.reset()
        episode_return = 0.0
        while not eval_time_step.is_last():
            eval_action_step = agent.policy.action(eval_time_step)
            eval_time_step = eval_env.step(eval_action_step)
            episode_return += eval_time_step.reward
        print(f"episode return: {episode_return}")
        total_return += episode_return

    print("Evaluation of the agent:")
    print(f"In {num_eval_episodes} episodes the trained agent got "
          f"{total_return}/{max_return} reward.")
    end_time = time.time()
    procedure_time = end_time - start_time
    print(f"Needed {procedure_time} seconds which is {procedure_time/60} minutes "
          f"or {procedure_time/3600} hours for {num_iterations} iterations")


for iterations in in_num_iterations_options:
    for rate in in_learning_rates:
        for target_update in in_target_update_steps_options:
            for gradient in in_gradient_clippings:
                # first perform epsilon decay
                for eps in in_end_epsilons:
                    cdqn_agent_training(
                        in_num_iterations=iterations,
                        in_learning_rate=rate,
                        in_end_learning_rate=in_end_learning_rate,
                        in_start_epsilon=in_start_epsilon,
                        in_end_epsilon=eps,
                        in_use_epsilon=True,
                        in_boltzmann_temperatur=0.0,
                        in_target_update_steps=target_update,
                        in_gradient_clipping=gradient,
                        in_show_plot=False
                    )
                for temp in in_boltzmann_temperatures:
                    cdqn_agent_training(
                        iterations,
                        rate,
                        in_start_epsilon,
                        1e-3,
                        False,
                        temp,
                        target_update,
                        gradient,
                        False
                    )

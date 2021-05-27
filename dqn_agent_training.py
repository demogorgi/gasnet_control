from __future__ import absolute_import, division, print_function

import base64
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time

import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.policies import random_tf_policy
from tf_agents.policies import policy_saver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common

import network_environment

# hyperparameters
in_num_iterations_options = [30000]#[5000, 20000, 50000]
in_learning_rates = [1e-5]
in_end_epsilons = []
in_boltzmann_temperatures = [0.1]
in_target_update_steps_options = [700] #100, 250, 400, 550, 700, 850, 1000
in_gradient_clippings = [1.0] #dont forget None value

def dqn_agent_training(
        in_num_iterations=20000,
        in_learning_rate=1e-5,
        in_end_epsilon=1e-4,
        in_use_epsilon=False,
        in_boltzmann_temperatur=1.0,
        in_target_update_steps=200,
        in_gradient_clipping=1.0,
        in_show_plot=True
):
    num_iterations = in_num_iterations    # @param {type:"integer"}

    initial_collect_steps = 100  # @param {type:"integer"}
    collect_steps_per_iteration = 1  # @param {type:"integer"}
    replay_buffer_max_length = 100000  # @param {type:"integer"}

    batch_size = 64  # @param {type:"integer"}
    learning_rate = in_learning_rate  # @param {type:"number"}
    log_interval = 200  # @param {type:"integer"}

    num_eval_episodes = 10  # @param {type:"integer"}
    eval_interval = 100#0  # @param {type:"integer"}

    # define a decaying epsilon over time, a boltzmann and which to use
    global_step = tf.compat.v1.train.get_or_create_global_step()
    start_epsilon = 0.1
    end_epsilon = in_end_epsilon
    epsilon = tf.compat.v1.train.polynomial_decay(
        start_epsilon,
        global_step,
        num_iterations,
        end_learning_rate=end_epsilon
    )
    boltzmann_temperatur = in_boltzmann_temperatur
    use_epsilon = in_use_epsilon

    # update steps for the new definition of the q target network
    target_update_steps = in_target_update_steps    # @param {type:"integer"}
    # norm to clip the gradient to for gradient descent
    gradient_clipping = in_gradient_clipping     # @param {type:"float"}

    #tf.compat.v1.enable_v2_behavior()

    temp_dir = os.getcwd() + '/instances'

    # custom hyperoarameters

    max_agent_steps = 10 # @param {type:"integer"}
    steps_per_agent_step = 8    # @param {type:"integer"}
    discretization = 10 # @param {type:"integer"}

    convert_action = True   # @param {type:"boolean"}
    random_entry_nominations = True   # @param {type:"boolean"}

    show_plot = in_show_plot

    ###### TESTING FUNCTIONALITY ############
    env = network_environment.\
        GasNetworkEnv(discretization_steps=discretization,
                      convert_action=convert_action,
                      steps_per_agent_step=steps_per_agent_step,
                      max_agent_steps=max_agent_steps,
                      random_nominations=random_entry_nominations)
    env.reset()

    print('Observation Spec:')
    print(env.time_step_spec().observation)

    print('Reward Spec:')
    print(env.time_step_spec().reward)

    print('Action Spec:')
    print(env.action_spec())

    time_step0 = env.reset()
    print('Time step:')
    print(time_step0)

    action = np.array(35, dtype=np.int32)

    next_time_step0 = env.step(action)
    print('Next time step:')
    print(next_time_step0)

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

    fc_layer_param = (100, 50)
    action_tensor_spec = tensor_spec.from_spec(env.action_spec())
    num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1


    # helper function for creation of dense layers
    def dense_layer(num_units):
        return tf.keras.layers.Dense(
            num_units,
            activation=tf.keras.activations.relu,
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=2.0, mode='fan_in', distribution='truncated_normal'
            )
        )


    # define q network with its layers
    dense_layers = [dense_layer(num_units) for num_units in fc_layer_param]
    q_values_layer = tf.keras.layers.Dense(
        num_actions,
        activation=None,
        kernel_initializer=tf.keras.initializers.RandomUniform(
            minval=-0.03, maxval=0.03
        ),
        bias_initializer=tf.keras.initializers.Constant(-0.2)
    )
    q_net = sequential.Sequential(dense_layers + [q_values_layer])

    # instantiate dqn agent
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    train_step_counter = tf.Variable(0)

    agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        #n_step_update=10,
        gradient_clipping=gradient_clipping,
        target_update_period=target_update_steps,
        train_step_counter=train_step_counter,
        epsilon_greedy=epsilon if use_epsilon else None,
        boltzmann_temperature=None if use_epsilon else boltzmann_temperatur
    )

    agent.initialize()

    # define policies
    eval_policy = agent.policy
    collect_policy = agent.collect_policy

    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                    train_env.action_spec())

    # example for testing the random policy
    # example_env = tf_py_environment.TFPyEnvironment(
    #     network_environment.GasNetworkEnv(10)
    # )
    # time_step1 = example_env.reset()
    # random_policy.action(time_step1)


    def compute_avg_return(environment, policy, num_episodes=10):

        total_return = 0.0
        for _ in range(num_episodes):

            time_step = environment.reset()
            episode_return = 0.0

            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = environment.step(action_step.action)
                episode_return += time_step.reward
            total_return += episode_return

        avg_return = total_return / num_episodes
        return avg_return.numpy()[0]


    # testing above function
    # compute_avg_return(eval_env, random_policy, num_eval_episodes)

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
        traj = trajectory.from_transition(time_step, action_step, next_time_step)

        # add trajectory to the replay buffer
        buffer.add_batch(traj)


    def collect_data(environment, policy, buffer, steps):
        for _ in range(steps):
            collect_step(environment, policy, buffer)


    collect_data(train_env, random_policy, replay_buffer, initial_collect_steps)

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=batch_size,
        num_steps=2
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

    # initialize the necessary variables for saving the policy for later use
    policy_name = f"policy_iters{int(num_iterations/1000)}_" +\
                  f"rate{str(learning_rate).replace('0', '')}_" +\
                  f"clip{int(gradient_clipping) if gradient_clipping is not None else 'None'}_" +\
                f"update{target_update_steps}_" +\
                f"{'epsilondecay' if use_epsilon else 'boltzmann'}" +\
                f"{end_epsilon if use_epsilon else boltzmann_temperatur}"
    policy_dir = os.path.join(temp_dir, policy_name)
    tf_policy_saver = policy_saver.PolicySaver(agent.policy)
    start_time = time.time()

    # train
    for _ in range(num_iterations):
        # collect a few steps using collect_policy and save to the replay buffer
        collect_data(train_env, agent.collect_policy, replay_buffer,
                     collect_steps_per_iteration)

        # sample a batch of data from the buffer and update the agent's network
        experience, unused_info = next(iterator)
        train_loss = agent.train(experience).loss

        step = agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss))
            losses += [train_loss]

        if step % eval_interval == 0:
            avg_return = compute_avg_return(eval_env, agent.policy,
                                            1)#num_eval_episodes)
            print('step = {0}: Average Return = {1}'.format(step, avg_return))
            returns.append(avg_return)

    print(returns)

    # save the policy
    tf_policy_saver.save(policy_dir)

    # print the result training
    iterations = range(0, num_iterations + 1, eval_interval)
    plt.figure()
    plt.plot(iterations, returns)
    plt.xlabel('Iterations')
    plt.ylabel('Average Return')
    plt.savefig(f"/home/adi/Uni/SoSe21/Masterarbeit/" +
                f"weight10_agent10_simulation8_discret10_systematic/" +
                f"reward_iters{int(num_iterations/1000)}_" +
                f"rate{str(learning_rate).replace('0', '')}_" +
                f"clip{int(gradient_clipping) if gradient_clipping is not None else 'None'}_" +
                f"update{target_update_steps}_" +
                f"{'epsilondecay' if use_epsilon else 'boltzmann'}" +
                f"{end_epsilon if use_epsilon else boltzmann_temperatur}.png")
    if show_plot:
        plt.show()

    iterations = range(0, num_iterations, log_interval)
    plt.figure()
    plt.plot(iterations, losses)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.savefig(f"/home/adi/Uni/SoSe21/Masterarbeit/" +
                f"weight10_agent10_simulation8_discret10_systematic/" +
                f"loss_iters{int(num_iterations/1000)}_" +
                f"rate{str(learning_rate).replace('0', '')}_" +
                f"clip{int(gradient_clipping) if gradient_clipping is not None else 'None'}_" +
                f"update{target_update_steps}_" +
                f"{'epsilondecay' if use_epsilon else 'boltzmann'}" +
                f"{end_epsilon if use_epsilon else boltzmann_temperatur}.png")
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
                    dqn_agent_training(
                        in_num_iterations=iterations,
                        in_learning_rate=rate,
                        in_end_epsilon=eps,
                        in_use_epsilon=True,
                        in_boltzmann_temperatur=0.0,
                        in_target_update_steps=target_update,
                        in_gradient_clipping=gradient,
                        in_show_plot=False
                    )
                for temp in in_boltzmann_temperatures:
                    dqn_agent_training(
                        iterations,
                        rate,
                        1e-3,
                        False,
                        temp,
                        target_update,
                        gradient,
                        False
                    )

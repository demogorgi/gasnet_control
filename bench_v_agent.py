import numpy as np
from benchmark import *
import matplotlib.pyplot as plt
import tensorflow as tf
import evaluation_network_environment
import random

from tf_agents.environments import tf_py_environment

obs_co = importlib.import_module(wd + ".observable_connections")

eval_const = False
eval_randomstart = True
eval_randomstartstep = False
eval_allrandom = False


def eval_constant_scenario():
    # basic parameter definition
    simulations_per = 8
    n_agent_steps = 6

    # initial plot preparation
    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches(10, 5)
    y_limits = [0.7, 1.0]
    x = range(1, n_agent_steps + 1)

    # run the benchmark
    decisions, init_decisions = get_benchmark(simulation_steps=simulations_per,
                                              n_episodes=n_agent_steps,
                                              flow_variant=False,
                                              rounded_decimal=0,
                                              enable_idle_compressor=True,
                                              custom_init_decisions=None)

    rewards = perform_benchmark(decisions=decisions,
                                simulation_steps=simulations_per,
                                n_episodes=n_agent_steps,
                                custom_init_decisions=None)
    axes[0].set_ylim(y_limits)
    axes[0].plot(x, rewards)

    policy_dir = f"/home/adi/anaconda3/envs/pyGasnetwork/gasnet_control/" +\
                 f"instances/da2/policies/{'200kiters_constant'}/" + \
                 "policy_" + \
                 f"cdqn_{(32,)}realQ_"  +\
                 f"iters{200}_" + \
                 f"rate1e-2to1e-05_" + \
                 f"clipNone_" + \
                 f"update{500}_" + \
                 f"epsilondecay{'01'}to0001_{'sigmoid'}"
    trained_policy = tf.compat.v2.saved_model.load(policy_dir)
    eval_py_env = evaluation_network_environment.GasNetworkEnv(
        discretization_steps=10,
        convert_action=True,
        steps_per_agent_step=simulations_per,
        max_agent_steps=n_agent_steps,
        random_nominations=False,
        print_actions=True
    )

    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
    time_step = eval_env.reset()

    for _ in range(n_agent_steps):
        # time.sleep(5)
        if not time_step.is_last():
            action_step = trained_policy.action(time_step)
            time_step = eval_env.step(action_step.action)

    with open("rewardfile.csv", "r") as rewardfile:
        reward_string = rewardfile.read()
    rewardfile.close()
    agent_rewards = []
    for reward in reward_string.split(";"):
        if reward != '':
            agent_rewards += [float(reward)]

    axes[1].set_ylim(y_limits)
    axes[1].plot(x, agent_rewards)
    plt.show()

    print(f"Overall Benchmark reward: \t{sum(rewards)}")
    print(f"Overall Agent reward: \t\t{sum(agent_rewards)}")


def eval_randomstart_scenario():
    # basic parameter definition
    simulations_per = 8
    entry_offset = -2 * simulations_per
    n_agent_steps = 6
    n_scenarios = 10
    sample_randomly = False
    C_to_evaluate = [1000]
    eps_to_evaluate = ['01']

    # initial plot preparation
    fig, axes = plt.subplots(1, 1)
    fig.set_size_inches(8, 5)
    y_limits = [0.8, 1.0]
    x = range(1, n_agent_steps + 1)

    # evaluate benchmark and agent on random scenarios
    scenarios = []
    if sample_randomly:
        scenario_nominations = [-1]
    else:
        scenario_nominations = [0, 150, 400, 300, 450, 550, 750, 600, 350, 800]
    agent_rewards = {}
    for update_steps in C_to_evaluate:
        agent_rewards[update_steps] = {}
        for eps in eps_to_evaluate:
            agent_rewards[update_steps][eps] = {}
    benchmark_rewards = {}
    entries = obs_co.special
    for scenario in range(n_scenarios):
        # sample the random nomination
        if sample_randomly:
            while True:
                n0 = random.choices(range(0, 1101, 50), k=1)[0]
                if n0 not in scenario_nominations:
                    scenario_nominations += [n0]
                    break
        else:
            n0 = scenario_nominations[scenario]

        scenarios += [[1100 - n0]*n_agent_steps]
        # extract the decision file
        with open(path.join(data_path, "init_decisions.yml")) as init_file:
            init_decs = yaml.load(init_file, Loader=yaml.FullLoader)

        # insert the decisions
        for step in range(n_agent_steps):
            ind = simulations_per * step + entry_offset
            init_decs["entry_nom"]["S"][joiner(entries[0])][ind] = n0
            init_decs["entry_nom"]["S"][joiner(entries[1])][ind] = 1100 - n0

        # save the decisions for the agent evaluation
        temp_decisions_string = "init_decisions_temp.yml"
        with open(path.join(data_path, temp_decisions_string), "w") \
                as temp_decisions:
            yaml.dump(init_decs, temp_decisions)

        # perform the benchmark evaluation
        bench_decisions, _ = get_benchmark(simulation_steps=simulations_per,
                                           n_episodes=n_agent_steps,
                                           flow_variant=False,
                                           rounded_decimal=0,
                                           enable_idle_compressor=True,
                                           custom_init_decisions=init_decs.copy())
        benchmark_rewards[scenario] = perform_benchmark(
            decisions=bench_decisions,
            simulation_steps=simulations_per,
            n_episodes=n_agent_steps,
            custom_init_decisions=init_decs)

        # perform the agent evaluation
        for update_steps in C_to_evaluate:
            for eps in eps_to_evaluate:
                policy_dir = f"/home/adi/anaconda3/envs/pyGasnetwork/gasnet_control/" +\
                             f"instances/da2/policies/{'200kiters_randomstart'}/" + \
                             "policy_" + \
                             f"cdqn_{(22, 32, 42)}realQ_"  +\
                             f"iters{200}_" + \
                             f"rate1e-2to1e-05_" + \
                             f"clipNone_" + \
                             f"update{update_steps}_" + \
                             f"epsilondecay{eps}to0001_{'tanh'}"
                trained_policy = tf.compat.v2.saved_model.load(policy_dir)
                eval_py_env = evaluation_network_environment.GasNetworkEnv(
                    discretization_steps=10,
                    convert_action=True,
                    steps_per_agent_step=simulations_per,
                    max_agent_steps=n_agent_steps,
                    random_nominations=False,
                    print_actions=False,
                    decision_string=temp_decisions_string
                )

                eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
                time_step = eval_env.reset()

                for _ in range(n_agent_steps):
                    if not time_step.is_last():
                        action_step = trained_policy.action(time_step)
                        time_step = eval_env.step(action_step.action)

                with open("rewardfile.csv", "r") as rewardfile:
                    reward_string = rewardfile.read()
                rewardfile.close()
                agent_rewards[update_steps][eps][scenario] = []
                for reward in reward_string.split(";"):
                    if reward != '':
                        agent_rewards[update_steps][eps][scenario] += [float(reward)]

    print(f"Scenario Nominations: {scenario_nominations}")
    # print(f"Overall Benchmark reward: \t{benchmark_rewards}")
    # print(f"Overall Agent reward: \t\t{agent_rewards}")
    if len(C_to_evaluate) == 1 and len(eps_to_evaluate) == 1:
        benchmark_y = []
        agent_y = []
        for eval_step in range(n_agent_steps):
            benchmark_y += [0]
            agent_y += [0]
            for scenario in range(n_scenarios):
                benchmark_y[-1] += benchmark_rewards[scenario][eval_step]
                agent_y[-1] += agent_rewards[C_to_evaluate[0]][eps_to_evaluate[0]][scenario][eval_step]
            benchmark_y[-1] /= n_scenarios
            agent_y[-1] /= n_scenarios

        axes.set_ylim(y_limits)
        axes.set_xlabel("Benchmark/Agent Decisions")
        axes.set_ylabel("Average Reward per Decision")
        axes.plot(x, benchmark_y, 'bx-', label="Benchmark")
        #axes[1].set_ylim(y_limits)
        #axes[1].set_xlabel("Agent Steps")
        axes.plot(x, agent_y, 'gx-', label="Agent")
        axes.legend(loc='lower right')
        plt.show()

        print(f"Average overall benchmark reward {np.sum(benchmark_y)}")
        print(f"Average overall agent reward {np.sum(agent_y)}")

        table_content = ""
        for scenario in range(n_scenarios):
            table_content += f"{scenario + 1}"
            for nomination in scenarios[scenario]:
                table_content += f"&{nomination}"
            table_content += f"\\\\ \\hline\n"
        with open("scenario_table_b.txt", "w") as scenario_table:
            scenario_table.write(table_content)
    else:
        benchmark_reward_sum = [
            np.sum(benchmark_rewards[scenario]) for scenario in range(n_scenarios)
        ]
        print(f"Overall Benchmark Reward per Scenario:\t{benchmark_reward_sum}")
        print(f"Overall Benchmark Reward: \t{np.sum(benchmark_reward_sum)}")
        for update_steps in C_to_evaluate:
            for eps in eps_to_evaluate:
                agent_reward_sum = [
                    np.sum(agent_rewards[update_steps][eps][scenario])
                    for scenario in range(n_scenarios)
                ]
                #print(f"Overall Agent Reward per Scenario for {update_steps} {eps}:\t{agent_reward_sum}")
                print(f"Overall Agent Reward per Scenario for {update_steps} {eps}:\t{np.sum(agent_reward_sum)}")


def eval_randomstartstep_scenario():
    # basic parameter definition
    simulations_per = 8
    entry_offset = -2 * simulations_per
    n_agent_steps = 6
    n_scenarios = 10
    sample_randomly = False
    C_to_evaluate = [1000]#[1, 5, 20, 100, 500, 1000, 2000, 4000]
    eps_to_evaluate = ['10'] #['01', '025', '05', '10']

    # initial plot preparation
    fig, axes = plt.subplots(1, 1)
    fig.set_size_inches(8, 5)
    y_limits = [0.7, 1.0]
    x = range(1, n_agent_steps + 1)

    # evaluate benchmark and agent on random scenarios
    scenarios = []
    if sample_randomly:
        scenario_nom_starts = []
        scenario_nom_steps = []
    else:
        #[450, 850, 200, 1000, 0, 800, 300, 950, 100, 150]#
        #[200, 150, 1100, 200, 50, 700, 200, 650, 1000, 950]#
        scenario_nom_starts = [650, 250, 900, 100, 1100, 300, 800, 150, 1000, 950]#[400]# [650, 250, 900, 100, 1100, 300, 800, 150, 1000, 950]
        scenario_nom_steps = [900, 950, 0, 900, 1050, 400, 900, 450, 100, 150]#[800]#[900, 950, 0, 900, 1050, 400, 900, 450, 100, 150]
    agent_rewards = {}
    for update_steps in C_to_evaluate:
        agent_rewards[update_steps] = {}
        for eps in eps_to_evaluate:
            agent_rewards[update_steps][eps] = {}
    benchmark_rewards = {}
    entries = obs_co.special
    for scenario in range(n_scenarios):
        # sample the random nomination
        if sample_randomly:
            while True:
                n0 = random.choices(range(0, 1101, 50), k=1)[0]
                if n0 not in scenario_nom_starts:
                    scenario_nom_starts += [n0]
                    break
            while True:
                n1 = random.choices(range(0, 1101, 50), k=1)[0]
                if n1 != n0:
                    scenario_nom_steps += [n1]
                    break
        else:
            n0 = scenario_nom_starts[scenario]
            n1 = scenario_nom_steps[scenario]

        scenario_first = [1100 - n0] * int(n_agent_steps/2)
        scenario_second = [1100 - n1] * int(n_agent_steps/2)
        scenarios += [scenario_first + scenario_second]
        # extract the decision file
        with open(path.join(data_path, "init_decisions.yml")) as init_file:
            init_decs = yaml.load(init_file, Loader=yaml.FullLoader)

        # insert the decisions
        for step in range(n_agent_steps + 1):
            ind = simulations_per * step + entry_offset
            if step < 3:
                nom = n0
            else:
                nom = n1
            init_decs["entry_nom"]["S"][joiner(entries[0])][ind] = nom
            init_decs["entry_nom"]["S"][joiner(entries[1])][ind] = 1100 - nom

        # save the decisions for the agent evaluation
        temp_decisions_string = "init_decisions_temp.yml"
        with open(path.join(data_path, temp_decisions_string), "w") \
                as temp_decisions:
            yaml.dump(init_decs, temp_decisions)

        # perform the benchmark evaluation
        bench_decisions, _ = get_benchmark(simulation_steps=simulations_per,
                                           n_episodes=n_agent_steps,
                                           flow_variant=False,
                                           rounded_decimal=0,
                                           enable_idle_compressor=True,
                                           custom_init_decisions=init_decs.copy())
        benchmark_rewards[scenario] = perform_benchmark(
            decisions=bench_decisions,
            simulation_steps=simulations_per,
            n_episodes=n_agent_steps,
            custom_init_decisions=init_decs)

        # perform the agent evaluation
        for update_steps in C_to_evaluate:
            for eps in eps_to_evaluate:
                policy_dir = f"/home/adi/anaconda3/envs/pyGasnetwork/gasnet_control/" +\
                             f"instances/da2/policies/{'200kiters_randomstepstart'}/" + \
                             "policy_" + \
                             f"cdqn_{(25, 38)}realQ_" +\
                             f"iters{200}_" + \
                             f"rate1e-2to1e-05_" + \
                             f"clipNone_" + \
                             f"update{update_steps}_" + \
                             f"epsilondecay{eps}to0001_{'tanh'}"
                try:
                    trained_policy = tf.compat.v2.saved_model.load(policy_dir)
                except OSError:
                    continue
                eval_py_env = evaluation_network_environment.GasNetworkEnv(
                    discretization_steps=10,
                    convert_action=True,
                    steps_per_agent_step=simulations_per,
                    max_agent_steps=n_agent_steps,
                    random_nominations=False,
                    print_actions=False,
                    decision_string=temp_decisions_string
                )

                eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
                time_step = eval_env.reset()

                for _ in range(n_agent_steps):
                    if not time_step.is_last():
                        action_step = trained_policy.action(time_step)
                        time_step = eval_env.step(action_step.action)

                with open("rewardfile.csv", "r") as rewardfile:
                    reward_string = rewardfile.read()
                rewardfile.close()
                agent_rewards[update_steps][eps][scenario] = []
                for reward in reward_string.split(";"):
                    if reward != '':
                        agent_rewards[update_steps][eps][scenario] += [float(reward)]

    print(f"Scenario Start Nominations: {scenario_nom_starts}")
    print(f"Scenario Step Nominations: {scenario_nom_steps}")
    # print(f"Overall Benchmark reward: \t{benchmark_rewards}")
    # print(f"Overall Agent reward: \t\t{agent_rewards}")
    if len(C_to_evaluate) == 1 and len(eps_to_evaluate) == 1:
        benchmark_y = []
        agent_y = []
        for eval_step in range(n_agent_steps):
            benchmark_y += [0]
            agent_y += [0]
            for scenario in range(n_scenarios):
                benchmark_y[-1] += benchmark_rewards[scenario][eval_step]
                agent_y[-1] += agent_rewards[C_to_evaluate[0]][eps_to_evaluate[0]][scenario][eval_step]
            benchmark_y[-1] /= n_scenarios
            agent_y[-1] /= n_scenarios

        axes.set_ylim(y_limits)
        axes.set_xlabel("Benchmark/Agent Decisions")
        axes.set_ylabel("Average Reward per Decision")
        axes.plot(x, benchmark_y, 'bx-', label="Benchmark")
        # axes[1].set_ylim(y_limits)
        # axes[1].set_xlabel("Agent Steps")
        # axes[1].plot(x, agent_y)
        axes.plot(x, agent_y, 'gx-', label="Agent")
        axes.legend(loc='lower right')
        plt.show()

        print(f"Average overall benchmark reward {np.sum(benchmark_y)}")
        print(f"Average overall agent reward {np.sum(agent_y)}")

        table_content = ""
        for scenario in range(n_scenarios):
            table_content += f"{scenario + 1}"
            for nomination in scenarios[scenario]:
                table_content += f"&{nomination}"
            table_content += f"\\\\ \\hline\n"
        with open("scenario_table_c.txt", "w") as scenario_table:
            scenario_table.write(table_content)
    else:
        benchmark_reward_sum = [
            np.sum(benchmark_rewards[scenario]) for scenario in range(n_scenarios)
        ]
        print(f"Overall Benchmark Reward per Scenario:\t{benchmark_reward_sum}")
        print(f"Overall Benchmark Reward: \t{np.sum(benchmark_reward_sum)}")
        for update_steps in C_to_evaluate:
            for eps in eps_to_evaluate:
                try:
                    agent_reward_sum = [
                        np.sum(agent_rewards[update_steps][eps][scenario])
                        for scenario in range(n_scenarios)
                    ]
                except KeyError:
                    continue
                #print(f"Overall Agent Reward per Scenario for {update_steps} {eps}:\t{agent_reward_sum}")
                print(f"Overall Agent Reward per Scenario for {update_steps} {eps}:\t{np.sum(agent_reward_sum)}")


def eval_allrandom_scenario():
    # basic parameter definition
    simulations_per = 8
    entry_offset = -2 * simulations_per
    n_agent_steps = 6
    n_scenarios = 1
    sample_randomly = False
    use_oge_scenario = True
    C_to_evaluate = [500]
    eps_to_evaluate = ['10']

    # initial plot preparation
    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches(10, 5)
    y_limits = [-0.1, 1.05]
    x = range(1, n_agent_steps + 1)

    # evaluate benchmark and agent on random scenarios
    entries = obs_co.special
    scenarios = []
    if sample_randomly:
        scenario_nom_starts = []
        scenario_nom_steps = []
    else:
        if use_oge_scenario:
            with open(path.join(data_path, "evaluation_scenario.yml")) as eval_file:
                eval_decs = yaml.load(eval_file, Loader=yaml.FullLoader)
            eval_scenario = []
            for step in range(n_agent_steps + 1):
                ind = simulations_per * step + entry_offset
                eval_scenario += [eval_decs["entry_nom"]["S"][joiner(entries[1])][ind]]

        else:
            eval_scenario = [100, 200, 300, 400, 500, 600]
    agent_rewards = {}
    for update_steps in C_to_evaluate:
        agent_rewards[update_steps] = {}
        for eps in eps_to_evaluate:
            agent_rewards[update_steps][eps] = {}
    benchmark_rewards = {}

    for scenario in range(n_scenarios):
        # sample the random nomination
        if sample_randomly:
            while True:
                n0 = random.choices(range(0, 1101, 50), k=1)[0]
                if n0 not in scenario_nom_starts:
                    scenario_nom_starts += [n0]
                    break
            while True:
                n1 = random.choices(range(0, 1101, 50), k=1)[0]
                if n1 != n0:
                    scenario_nom_steps += [n1]
                    break
        else:
            scenarios = [eval_scenario]
        # extract the decision file
        with open(path.join(data_path, "init_decisions.yml")) as init_file:
            init_decs = yaml.load(init_file, Loader=yaml.FullLoader)

        # insert the decisions
        for step in range(n_agent_steps + 1):
            ind = simulations_per * step + entry_offset
            nom = 1100 - scenarios[scenario][step]
            init_decs["entry_nom"]["S"][joiner(entries[0])][ind] = nom
            init_decs["entry_nom"]["S"][joiner(entries[1])][ind] = 1100 - nom

        # save the decisions for the agent evaluation
        temp_decisions_string = "init_decisions_temp.yml"
        with open(path.join(data_path, temp_decisions_string), "w") \
                as temp_decisions:
            yaml.dump(init_decs, temp_decisions)

        # perform the benchmark evaluation
        bench_decisions, _ = get_benchmark(simulation_steps=simulations_per,
                                           n_episodes=n_agent_steps,
                                           flow_variant=False,
                                           rounded_decimal=0,
                                           enable_idle_compressor=True,
                                           custom_init_decisions=init_decs.copy())
        benchmark_rewards[scenario] = perform_benchmark(
            decisions=bench_decisions,
            simulation_steps=simulations_per,
            n_episodes=n_agent_steps,
            custom_init_decisions=init_decs)

        # perform the agent evaluation
        for update_steps in C_to_evaluate:
            for eps in eps_to_evaluate:
                policy_dir = f"/home/adi/anaconda3/envs/pyGasnetwork/gasnet_control/" +\
                             f"instances/da2/policies/{'400kiters_allrandom'}/" + \
                             "policy_" + \
                             f"cdqn_{(22, 32, 42)}realQ_" +\
                             f"iters{400}_" + \
                             f"rate1e-2to1e-05_" + \
                             f"clipNone_" + \
                             f"update{update_steps}_" + \
                             f"epsilondecay{eps}to0001_{'sigmoid'}"
                try:
                    trained_policy = tf.compat.v2.saved_model.load(policy_dir)
                except OSError:
                    continue
                eval_py_env = evaluation_network_environment.GasNetworkEnv(
                    discretization_steps=10,
                    convert_action=True,
                    steps_per_agent_step=simulations_per,
                    max_agent_steps=n_agent_steps,
                    random_nominations=False,
                    print_actions=False,
                    decision_string=temp_decisions_string
                )

                eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
                time_step = eval_env.reset()

                for _ in range(n_agent_steps):
                    if not time_step.is_last():
                        action_step = trained_policy.action(time_step)
                        time_step = eval_env.step(action_step.action)

                with open("rewardfile.csv", "r") as rewardfile:
                    reward_string = rewardfile.read()
                rewardfile.close()
                agent_rewards[update_steps][eps][scenario] = []
                for reward in reward_string.split(";"):
                    if reward != '':
                        agent_rewards[update_steps][eps][scenario] += [float(reward)]

    #print(f"Scenario Start Nominations: {scenario_nom_starts}")
    #print(f"Scenario Step Nominations: {scenario_nom_steps}")
    # print(f"Overall Benchmark reward: \t{benchmark_rewards}")
    # print(f"Overall Agent reward: \t\t{agent_rewards}")
    if len(C_to_evaluate) == 1 and len(eps_to_evaluate) == 1:
        benchmark_y = []
        agent_y = []
        for eval_step in range(n_agent_steps):
            benchmark_y += [0]
            agent_y += [0]
            for scenario in range(n_scenarios):
                benchmark_y[-1] += benchmark_rewards[scenario][eval_step]
                agent_y[-1] += agent_rewards[C_to_evaluate[0]][eps_to_evaluate[0]][scenario][eval_step]
            benchmark_y[-1] /= n_scenarios
            agent_y[-1] /= n_scenarios

        axes[0].set_ylim(y_limits)
        axes[0].plot(x, benchmark_y)
        axes[1].set_ylim(y_limits)
        axes[1].plot(x, agent_y)
        plt.show()

        print(f"Overall benchmark reward {np.sum(benchmark_y)}")
        print(f"Overall agent reward {np.sum(agent_y)}")

        table_content = ""
        for scenario in range(n_scenarios):
            table_content += f"{scenario + 1}"
            for nomination in scenarios[scenario]:
                table_content += f"&{nomination}"
            table_content += f"\\\\ \\hline\n"
        with open("scenario_table_d.txt", "w") as scenario_table:
            scenario_table.write(table_content)
    else:
        benchmark_reward_sum = [
            np.sum(benchmark_rewards[scenario]) for scenario in range(n_scenarios)
        ]
        print(f"Overall Benchmark Reward per Scenario:\t{benchmark_reward_sum}")
        print(f"Overall Benchmark Reward: \t{np.sum(benchmark_reward_sum)}")
        for update_steps in C_to_evaluate:
            for eps in eps_to_evaluate:
                try:
                    agent_reward_sum = [
                        np.sum(agent_rewards[update_steps][eps][scenario])
                        for scenario in range(n_scenarios)
                    ]
                except KeyError:
                    continue
                #print(f"Overall Agent Reward per Scenario for {update_steps} {eps}:\t{agent_reward_sum}")
                print(f"Overall Agent Reward per Scenario for {update_steps} {eps}:\t{np.sum(agent_reward_sum)}")


if __name__ == "__main__":
    if eval_const:
        eval_constant_scenario()
    if eval_randomstart:
        eval_randomstart_scenario()
    if eval_randomstartstep:
        eval_randomstartstep_scenario()
    if eval_allrandom:
        eval_allrandom_scenario()
import time

import numpy as np
import instances.da2.connections
from urmel import *
from model import *
import functions as funcs
from params import *

max_iterations = 15


def get_decision(decisions, time_step):
    for i in range(time_step, -17, -1):
        if i in decisions:
            return decisions[i]
    return None


def get_benchmark(simulation_steps=8, n_episodes=10, flow_variant=False,
                  rounded_decimal=None):
    init_states = funcs.get_init_scenario()

    low_entry = co.special[1]
    high_entry = co.special[0]
    low_init_flow = init_states[-1]["q_out"][low_entry]
    high_init_flow = init_states[-1]["q_out"][high_entry]

    time_offset = simulation_steps*(-2)
    # force numpy to round 0.5 down
    numpy_offset = 0
    if rounded_decimal is not None:
        numpy_offset = -10**(-rounded_decimal - 2)

    with open(path.join(data_path, 'init_decisions.yml')) as init_file:
        init_decisions = yaml.load(init_file, Loader=yaml.FullLoader)

    low_exit = no.exits[0]
    high_exit = no.exits[1]

    compressor = list(init_decisions["compressor"]["CS"].keys())[0]
    resistor = list(init_decisions["zeta"]["RE"].keys())[0]

    compressor_switch = 0
    compressor_min = 0.0
    compressor_max = 1.0
    compressor_start_values = [compressor_min, compressor_max, 0.5]
    resistor_min = 30
    resistor_max = 100
    resistor_start_values = [resistor_min, resistor_max]

    decisions = {
        "zeta" : [],
        "gas" : [],
        "compressor" : []
    }
    for episode in range(n_episodes):
        time_index = episode * simulation_steps

        low_exit_nom = np.abs(get_decision(
            init_decisions["exit_nom"]["X"][low_exit], time_index
        ))
        high_exit_nom = np.abs(get_decision(
            init_decisions["exit_nom"]["X"][high_exit], time_index
        ))

        low_entry_nom = get_decision(
            init_decisions["entry_nom"]["S"][joiner(low_entry)],
            time_index + time_offset
        )
        high_entry_nom = get_decision(
            init_decisions["entry_nom"]["S"][joiner(high_entry)],
            time_index + time_offset
        )

        searching_decision = True
        iterations = 0
        compressor_case = True

        if (flow_variant and low_init_flow - low_entry_nom < -1) or \
                (not flow_variant and low_exit_nom - low_entry_nom < -1):
            compressor_case = True
        elif (flow_variant and low_init_flow - low_entry_nom > 1) or \
                (not flow_variant and low_exit_nom - low_entry_nom > 1):
            compressor_case = False

        # if in compressor case, we set resistor to close and vice versa
        resistor_value = resistor_max * compressor_case
        compressor_switch = int(compressor_case)
        compressor_gas = float(compressor_case)
        compressor_values = compressor_start_values
        resistor_values = resistor_start_values + [65]
        current_decisions = {}

        # calculate the optimal compressor/resistor efficiency
        while searching_decision:
            # extract the updated gas value
            if compressor_case:
                compressor_gas = compressor_values[-1]
            else:
                resistor_value = resistor_values[-1]

            # write the decision into init_decisions
            init_decisions["zeta"]["RE"][resistor][time_index] = \
                resistor_value
            init_decisions["compressor"]["CS"][compressor][time_index] = \
                compressor_switch
            init_decisions["gas"]["CS"][compressor][time_index] = \
                compressor_gas

            low_avg_flow = 0
            # simulator_step.counter = 0
            for step in range(simulation_steps):
                solution = simulator_step(
                    init_decisions,
                    time_index + step,
                    "sim"
                )
                low_avg_flow += \
                    solution["var_pipe_Qo_out[%s,%s]" % low_entry]
            low_avg_flow /= simulation_steps

            if compressor_case:
                search_values = compressor_values
            else:
                search_values = resistor_values

            # safe the current decisions dependent on the value of interest
            current_decisions[np.abs(low_avg_flow - low_entry_nom)] = [
                resistor_value,
                compressor_switch,
                compressor_gas
            ]

            # control the search value in dependence of the avg flow
            if np.abs(low_avg_flow - low_entry_nom) <= 1.0:
                iterations = max_iterations
                # decisions["zeta"].append(resistor_value)
                # decisions["compressor"].append(compressor_switch)
                # decisions["gas"].append(compressor_gas)
            elif low_avg_flow < low_entry_nom:
                # make a bisection to the upper bound
                new_value = search_values[2] + \
                            (search_values[1] - search_values[2])/2
                # if rounding is desired, do so and check for useless loops
                if rounded_decimal is not None:
                    if compressor_case:
                        new_value = np.round(new_value + numpy_offset,
                                             rounded_decimal)
                    else:
                        new_value = np.round(new_value + numpy_offset,
                                             rounded_decimal - 2)

                    # numpy rounds downwards -> check for same value and
                    # equality to upper bound. If not -> test upper bound
                    if new_value == search_values[0]:
                        try:
                            last_decision = list(
                                current_decisions.values()
                            )[-1]
                        except KeyError:
                            last_decision = []
                        if new_value in last_decision:
                            iterations = max_iterations
                    elif new_value == search_values[2]:
                        if new_value != search_values[1]:
                            new_value = search_values[1]
                        else:
                            iterations = max_iterations

                search_values = [
                    search_values[2],
                    search_values[1],
                    new_value]
            else:
                # make a bisection to the lower bound
                new_value = search_values[0] + \
                            (search_values[2] - search_values[0])/2
                # if rounding is desired, do so and check for useless loops
                if rounded_decimal is not None:
                    if compressor_case:
                        new_value = np.round(new_value + numpy_offset,
                                             rounded_decimal)
                    else:
                        new_value = np.round(new_value + numpy_offset,
                                             rounded_decimal - 2)

                    # terminate if no change in search values
                    if new_value == search_values[2] or\
                       new_value == search_values[1]:
                        iterations = max_iterations

                search_values = [
                    search_values[0],
                    search_values[2],
                    new_value
                ]

            if compressor_case:
                compressor_values = search_values
            else:
                resistor_values = search_values

            iterations += 1
            searching_decision = iterations < max_iterations

            if not searching_decision:
                if compressor_case \
                        and low_avg_flow - low_entry_nom > 5.0 \
                        and compressor_gas < 0.01:
                    compressor_case = False
                    compressor_gas = 0.0
                    compressor_switch = 1
                    iterations = 0
                    searching_decision = True

        if len(decisions["zeta"]) <= episode:
            best_objective_value = min(current_decisions.keys())
            best_current_decision = current_decisions[best_objective_value]
            decisions["zeta"].append(best_current_decision[0])
            decisions["compressor"].append(best_current_decision[1])
            decisions["gas"].append(best_current_decision[2])

    return decisions, init_decisions


def perform_benchmark(decisions, simulation_steps=8, n_episodes=10):
    ub_entry_violation = 1100
    n_entries = 2
    # simulator_step.counter = 0
    overall_reward = 0.0

    with open(path.join(data_path, 'init_decisions.yml')) as init_file:
        init_decisions = yaml.load(init_file, Loader=yaml.FullLoader)

    compressor = list(init_decisions["compressor"]["CS"].keys())[0]
    resistor = list(init_decisions["zeta"]["RE"].keys())[0]

    # perform the decisions that were learned by the bisection procedure
    for episode in range(n_episodes):

        # time.sleep(5)
        print("#" * 15 + f"Evaluation of step {episode}" + "#" * 15)

        time_index = episode*simulation_steps
        compr_active = decisions["compressor"][episode]
        init_decisions["zeta"]["RE"][resistor][time_index] = \
            decisions["zeta"][episode]
        init_decisions["gas"]["CS"][compressor][time_index] = \
            decisions["gas"][episode]
        init_decisions["compressor"]["CS"][compressor][time_index] = \
            compr_active

        for valve in co.valves:
            print(f"valve {valve} is activated")
        print(f"resistor {resistor} works at efficiency"
              f" of {decisions['zeta'][episode]}")
        print(f"compressor {compressor} is "
              f"{'' if compr_active == 1 else 'not '}"
              f"activated"
              f"{'' if compr_active == 0 else ' with efficiency ' + str(decisions['gas'][episode])}")

        # initialize variables for rewards calculation
        episode_flow_violation = {}
        episode_pressure_violations = []

        # simulate each step
        for step in range(simulation_steps):
            solution = simulator_step(init_decisions, time_index + step, "sim")

            # calculate the reward as in network environment
            for variable_name in solution.keys():
                # deviations from entry flows have to be summed up for each
                if variable_name.startswith("nom_entry_slack_DA"):
                    # sum up pressure violations for each entry over one
                    # agent  step (may cancel out and is intended)
                    if variable_name not in episode_flow_violation:
                        episode_flow_violation[variable_name] = 0
                    episode_flow_violation[variable_name] += \
                        solution[variable_name]

                # count deviations from exit pressures to be penalized
                elif any(map(variable_name.__contains__, no.exits)):
                    if "b_pressure_violation_DA" in variable_name:
                        # define the potential violation
                        violation = solution[variable_name]
                        # positive slacks = violation -> identify it
                        if violation > 0:
                            # extract exit name
                            violated_exit = variable_name.split("[")[1]
                            violated_exit = violated_exit.split("]")[0]
                            # mark violations via appendix to set
                            episode_pressure_violations.add(violated_exit)

        # reward calculation between [-1, 1]
        # each flow violation has an impact of max 1/n_entries and is dependent
        # on the accumulated flow violation
        flow_violation = 0
        for violation in episode_flow_violation.values():
            flow_violation += min(
                np.abs(violation/(
                        simulation_steps *
                        ub_entry_violation
                        )),
                1.0
                )/n_entries

        # a pressure violation is rated is critical -> if n = amount exits
        # the ith exit is equal to a violation of 2^(n - i)/(2^n - 1)
        n_press_viol = len(episode_pressure_violations)
        pressure_violation = np.sum([2**(n_press_viol - i - 1) /
                                     (2**n_press_viol - 1)
                                     for i in range(n_press_viol)])
        reward = 1.0 - (pressure_violation + flow_violation)
        overall_reward += reward

        print(f"This step lead to a reward of {reward}")
        print(f"The accumulated flow violations are at "
              f"{episode_flow_violation}")
        print(f"The summed up pressure violations are "
              f"{pressure_violation}")
        print(f"The nominations for the current step were "
              f" not relevant and same as for agent")

        print("#" * 15 + f"End of evaluation of step {episode}" + "#" * 9 + "\n\n")

    print(f"The overall reward is {overall_reward}")


# main program
# handle the input via a command line execution
steps_per_episode = config['nomination_freq']
try:
    time_horizon = int(sys.argv[2])
except ValueError:
    raise ValueError(f"Second argument after file name has to give the time "
                     f"horizon as integer. {sys.argv[2]} is not of type int.")
if time_horizon % steps_per_episode != 0:
    raise ValueError(f"The time horizon {time_horizon} in #steps has to be "
                     f"dividable by {steps_per_episode}.\n"
                     f"Calculation is time horizon = steps per episode (here "
                     f"{steps_per_episode}) * number of episodes.")
else:
    amount_episodes = int(time_horizon / steps_per_episode)

if len(sys.argv) > 4:
    flow_calc_for_benchmark = sys.argv[4]
    if flow_calc_for_benchmark in ["y", "f", 1, "1"]:
        flow_calc_for_benchmark = True
    else:
        flow_calc_for_benchmark = False
else:
    flow_calc_for_benchmark = False

# if given correctly, get the benchmark based on the input
print("#"*20 + " CALCULATING BENCHMARK " + "#"*20)
decision, decisions_as_yaml = get_benchmark(
    simulation_steps=steps_per_episode,
    n_episodes=amount_episodes,
    flow_variant=flow_calc_for_benchmark,
    rounded_decimal=1
)

# write the decisions into a separate yml file if wanted
with open(
        path.join(data_path, 'benchmark_decisions.yml'),
        'w'
) as benchmark_file:
    yaml.dump(decisions_as_yaml, benchmark_file)

# perform the benchmark afterwards, but wait a little for user attentiveness
print("#"*20 + " PERFORMING BENCHMARK " + "#"*20)
time.sleep(15)
perform_benchmark(
    decisions=decision,
    simulation_steps=steps_per_episode,
    n_episodes=amount_episodes
)

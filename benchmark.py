import numpy as np
import instances.da2.connections
from urmel import *
from model import *
import functions as funcs
from params import *


def getDecision(decision, timestep):
    for i in range(timestep, -17, -1):
        if i in decision:
            return decision[i]
    return None


def getBenchmark(simulation_steps=8, n_episodes=10, flow_variant=False):
    init_states = funcs.get_init_scenario()

    low_entry = co.special[1]
    high_entry = co.special[0]
    low_init_flow = init_states[-1]["q_out"][low_entry]
    high_init_flow = init_states[-1]["q_out"][high_entry]

    time_offset = simulation_steps*(-2)

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

        low_exit_nom = np.abs(getDecision(
            init_decisions["exit_nom"]["X"][low_exit], time_index
        ))
        high_exit_nom = np.abs(getDecision(
            init_decisions["exit_nom"]["X"][high_exit], time_index
        ))

        low_entry_nom = getDecision(
            init_decisions["entry_nom"]["S"][joiner(low_entry)],
            time_index + time_offset
        )
        high_entry_nom = getDecision(
            init_decisions["entry_nom"]["S"][joiner(low_entry)],
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
            init_decisions["gas"]["CS"][compressor][time_index] = \
                compressor_gas
            init_decisions["compressor"]["CS"][compressor][time_index] = \
                compressor_switch

            low_avg_flow = 0
            simulator_step.counter = 0
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

            # control the search value in dependence of the avg flow
            if np.abs(low_avg_flow - low_entry_nom) <= 50.0:
                iterations = 10
                decisions["zeta"].append(resistor_value)
                decisions["compressor"].append(compressor_switch)
                decisions["gas"].append(compressor_gas)
            elif low_avg_flow < low_entry_nom:
                # make a bisection to the upper bound
                new_value = search_values[2] + \
                            (search_values[1] - search_values[2])/2
                search_values = [
                    search_values[2],
                    search_values[1],
                    new_value]
            else:
                # make a bisection to the lower bound
                new_value = search_values[0] + \
                            (search_values[2] - search_values[0])/2
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
            searching_decision = iterations < 10

        if len(decisions["zeta"]) <= episode:
            decisions["zeta"].append(resistor_value)
            decisions["compressor"].append(compressor_switch)
            decisions["gas"].append(compressor_gas)

    return decisions


def performBenchmark(decisions, simulation_steps=8, n_episodes=10):
    ub_entry_violation = 1100
    n_entries = 2
    simulator_step.counter = 0

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

        print(f"This step lead to a reward of {reward}")
        print(f"The accumulated flow violations are at "
              f"{episode_flow_violation}")
        print(f"The summed up pressure violations are "
              f"{pressure_violation}")
        print(f"The nominations for the current step were "
              f" not relevant and same as for agent")

        print("#" * 15 + f"End of evaluation of step {episode}" + "#" * 9 + "\n\n")


decision = getBenchmark()
performBenchmark(decision)
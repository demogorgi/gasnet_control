# standard imports as suggested at
# https://www.tensorflow.org/agents/tutorials/2_environments_tutorial?hl=en#python_environments
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np
import itertools
import random

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

#tf.compat.v1.enable_v2_behavior()

# own inputs
#import gasnet_control.instances.da2.connections
import instances.da2.connections
from urmel import *
from model import *
import functions as funcs
from params import *
obs_no = importlib.import_module(wd + ".observable_nodes")
obs_co = importlib.import_module(wd + ".observable_connections")


class GasNetworkEnv(py_environment.PyEnvironment):

    def __init__(self, discretization_steps=10, convert_action=True,
                 steps_per_agent_step=8, max_agent_steps=1,
                 random_nominations=True, print_actions=False):
        ### define the action specificities
        self._convert_action = convert_action
        self._steps_per_agent_steps = steps_per_agent_step
        self._max_agent_steps = max_agent_steps
        self._random_nominations = random_nominations
        self._entry_offset = self._steps_per_agent_steps * (-2)
        self._print_actions = print_actions

        # analyse initial decisions to extract values
        with open(path.join(data_path, 'init_decisions.yml')) as init_file:
            init_decisions = yaml.load(init_file, Loader=yaml.FullLoader)

        # safe the control variable names for later mapping
        self._valves = list(init_decisions["va"]["VA"].keys())
        self._resistors = list(init_decisions["zeta"]["RE"].keys())
        self._compressors = list(init_decisions["compressor"]["CS"].keys())

        n_resistors = len(self._resistors)
        n_compressors = len(self._compressors)
        n_control_vars = n_resistors + n_compressors

        # overall minimum set to 0, maxima are the discretization step size for
        # others
        self._discretization = discretization_steps
        control_minima = [0]*n_control_vars
        # upper resistor bounds [0, discretization] (in reality, [0, 100])
        resistor_maxima = [discretization_steps]*n_resistors
        compressor_maxima = [discretization_steps]*n_compressors
        control_maxima = resistor_maxima + compressor_maxima

        # define the actual action spec
        if convert_action:
            # convert all actions to one variable
            # define a list with lists of possible values for each action
            action_list = [list(range(minimum, maximum + 1))
                           for minimum, maximum
                           in zip(control_minima, control_maxima)]
            action_combinations = list(itertools.product(*action_list))
            self._action_mapping = {}
            for i, action in enumerate(action_combinations):
                self._action_mapping[i] = action

            self._action_spec = array_spec.BoundedArraySpec(
                shape=(),
                dtype=np.int32,
                minimum=0,
                maximum=len(self._action_mapping.keys()) - 1,
                name='action'
            )
        else:
            self._action_spec = array_spec.BoundedArraySpec(
                shape=(n_control_vars,),
                dtype=np.int32,
                minimum=np.array(control_minima),
                maximum=np.array(control_maxima),
                name='action'
            )
        ### define the observations specificities
        ## extract the nomination properties
        # define the entries & exits for which nominations are to be respected
        nomination_nodes = obs_no.nodes_with_bds
        # define the respective bounds and the amount
        nomination_minima = [obs_no.nom_lb[node]
                             for node in nomination_nodes]
        nomination_maxima = [obs_no.nom_ub[node]
                             for node in nomination_nodes]
        n_nomination_nodes = len(nomination_nodes)

        ## extract the pressure violation specifities
        # define the relevant nodes
        pressure_violation_nodes = obs_no.exits
        # define their amount doubled for upper and lower deviations
        n_pressure_violations = len(pressure_violation_nodes)*2
        # define their lower and upper bounds
        pressure_violation_minima = [0]*n_pressure_violations
        pressure_violation_maxima = [10]*n_pressure_violations

        ## extract the flow specificities
        # define the relevant pipes
        flow_pipes = list(obs_co.pipes)
        # define their amount
        n_flow_pipes = len(flow_pipes)
        # define their lower and upper bounds
        flow_minima = [-100]*n_flow_pipes
        flow_maxima = [1600]*n_flow_pipes

        ## define the actual observation spec
        # doubled nomination nodes since we want to respect  nominations of the
        # current and following time step
        n_observations = 2*n_nomination_nodes + \
                         n_pressure_violations + \
                         n_flow_pipes
        observation_minima = 2*nomination_minima + \
                             pressure_violation_minima + \
                             flow_minima

        observation_maxima = 2*nomination_maxima + \
                             pressure_violation_maxima + \
                             flow_maxima

        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(n_observations,), dtype=np.float32,
            minimum=np.array(observation_minima),
            maximum=np.array(observation_maxima),
            name='observation'
        )

        self._nodes = pressure_violation_nodes
        self._pipes = flow_pipes

        ### define the initial state
        ## extract the nominations
        # extract the initial exit nominations for time step 0
        # nominations_t0 = [init_decisions["exit_nom"]["X"][ex][0]
        #                  for ex in obs_no.exits_for_nom]
        nominations_t0 = []
        # extract/compute the initial entry nominations for time step 0
        if self._random_nominations:
            nomination_sum = [init_decisions["entry_nom"]["S"][joiner(supply)]
                              [0 + self._entry_offset]
                              for supply in obs_co.special]
            nomination_sum = int(np.abs(sum(nomination_sum)))
            n_entries = len(obs_no.nodes_with_bds) - len(obs_no.exits_for_nom)
            breaks = random.choices(range(0, nomination_sum + 1, 50),
                                    k=n_entries - 1)
            breaks.sort()
            breaks = [0] + breaks + [nomination_sum]

            nominations_t0 += [breaks[break_step] - breaks[break_step - 1]
                               for break_step in range(1, n_entries + 1)]
            # # swap implementation
            # self._nom_scenario = random.randint(0, 1)
            # if self._nom_scenario == 0:
            #     nominations_t0 += [400, 700]
            # else:
            #     nominations_t0 += [700, 400]
            # start with initial nominations implementation
            # nominations_t0 += [init_decisions["entry_nom"]["S"][joiner(supply)]
            #                   [0 + self._entry_offset]
            #                   for supply in co.special]
        else:
            nominations_t0 += [init_decisions["entry_nom"]["S"][joiner(supply)]
                               [0 + self._entry_offset]
                               for supply in co.special]
        # length of nominations has to be the same as in the observation specs
        assert(len(nominations_t0) == n_nomination_nodes)

        # extract/compute the initial nominations for time step 1
        nominations_t1 = []

        if self._random_nominations:
            # for count, node in enumerate(no.exits):
            #     try:
            #         nomination = init_decisions["exit_nom"]["X"][node]\
            #             [self._steps_per_agent_steps]
            #     except KeyError:
            #         nomination = nominations_t0[count]
            #     nominations_t1 += [nomination]
            #
            # breaks = random.choices(range(0, nomination_sum + 1, 50),
            #                        k=n_entries - 1)
            # breaks.sort()
            # breaks = [0] + breaks + [nomination_sum]
            # nominations_t1 += [breaks[break_step] - breaks[break_step - 1]
            #                    for break_step in range(1, n_entries + 1)]
            # epsilon tube implementation
            scenario = random.randint(0, 2)
            for count, node in enumerate(obs_no.exits_for_nom + co.special):
                try:
                    if type(node) == str:
                        nomination = init_decisions["exit_nom"]["X"][node]\
                            [self._steps_per_agent_steps]
                    else:
                        key = joiner(node)
                        #nomination = init_decisions["entry_nom"]["S"][key]\
                        #    [self._steps_per_agent_steps + self._entry_offset]
                        nomination = nominations_t0[count]
                        # epsilon tube implementation
                        # if 'EN' in key:
                        #     if scenario == 1:
                        #         nomination += 50
                        #     elif scenario == 2:
                        #         nomination -= 50
                        # else:
                        #     if scenario == 1:
                        #         nomination -= 50
                        #     elif scenario == 2:
                        #         nomination += 50
                except KeyError:
                    nomination = nominations_t0[count]
                nominations_t1 += [nomination]
        else:
            for count, node in enumerate(obs_no.exits_for_nom + co.special):
                try:
                    if type(node) == str:
                        nomination = init_decisions["exit_nom"]["X"][node]\
                            [self._steps_per_agent_steps]
                    else:
                        key = joiner(node)
                        nomination = init_decisions["entry_nom"]["S"][key]\
                            [self._steps_per_agent_steps + self._entry_offset]
                except KeyError:
                    nomination = nominations_t0[count]
                nominations_t1 += [nomination]

        ## extract the node pressure violations and flows
        # extract all initial values
        init_states = funcs.get_init_scenario()
        # extract the pressure violations
        pressure_violations_upper = [
            np.max([
                init_states[-1]["p"][node] - no.pressure_limits_upper[node],
                0
            ]) for node in pressure_violation_nodes
        ]
        pressure_violations_lower = [
            np.max([
                no.pressure_limits_lower[node] - init_states[-1]["p"][node],
                0
            ]) for node in pressure_violation_nodes
        ]
        pressure_violations = pressure_violations_upper + \
                              pressure_violations_lower
        # extract the flow values
        flows = [init_states[-1]["q_out"][pipe]
                        if pipe[1].startswith("X")
                        else init_states[-1]["q_in"][pipe]
                        for pipe in flow_pipes]

        ## define the initial state
        self._state = np.array(
                        nominations_t0 + nominations_t1 +
                        pressure_violations +
                        flows
                        , np.float32)

        self._episode_ended = False

        # counter to come to an end
        self._action_counter = 0

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        ### reset of the self state
        ## extract initial decisions/values
        with open(path.join(data_path, 'init_decisions.yml')) as init_file:
            init_decisions = yaml.load(init_file, Loader=yaml.FullLoader)

        ## extract the nominations
        # extract the initial exit nominations for time step 0
        # nominations_t0 = [init_decisions["exit_nom"]["X"][ex][0]
        #                  for ex in obs_no.exits_for_nom]
        nominations_t0 = []
        # extract/compute the initial entry nominations for time step 0
        if self._random_nominations:
            nomination_sum = [init_decisions["entry_nom"]["S"][joiner(supply)]
                              [0 + self._entry_offset]
                              for supply in obs_co.special]
            nomination_sum = int(np.abs(sum(nomination_sum)))
            n_entries = len(obs_no.nodes_with_bds) - len(obs_no.exits_for_nom)
            breaks = random.choices(range(0, nomination_sum + 1, 50),
                                    k=n_entries - 1)
            breaks.sort()
            breaks = [0] + breaks + [nomination_sum]

            nominations_t0 += [breaks[break_step] - breaks[break_step - 1]
                               for break_step in range(1, n_entries + 1)]
            # # swap implementation
            # self._nom_scenario = random.randint(0, 1)
            # if self._nom_scenario == 0:
            #     nominations_t0 += [400, 700]
            # else:
            #     nominations_t0 += [700, 400]
            # starting at given nomination implementation
            # nominations_t0 += [init_decisions["entry_nom"]["S"][joiner(supply)]
            #                   [0 + self._entry_offset]
            #                   for supply in co.special]
        else:
            nominations_t0 += [init_decisions["entry_nom"]["S"][joiner(supply)]
                               [0 + self._entry_offset]
                               for supply in co.special]

        # extract/compute the initial nominations for time step 1
        nominations_t1 = []
        # scenario for epsilon tube implementation
        scenario = random.randint(0, 2)
        if self._random_nominations:
            # for count, node in enumerate(no.exits):
            #     try:
            #         nomination = init_decisions["exit_nom"]["X"][node]\
            #             [self._steps_per_agent_steps]
            #     except KeyError:
            #         nomination = nominations_t0[count]
            #     nominations_t1 += [nomination]
            #
            # breaks = random.choices(range(0, nomination_sum + 1, 50),
            #                        k=n_entries - 1)
            # breaks.sort()
            # breaks = [0] + breaks + [nomination_sum]
            # nominations_t1 += [breaks[break_step] - breaks[break_step - 1]
            #                    for break_step in range(1, n_entries + 1)]
            for count, node in enumerate(obs_no.exits_for_nom + co.special):
                try:
                    if type(node) == str:
                        nomination = init_decisions["exit_nom"]["X"][node]\
                            [self._steps_per_agent_steps]
                    else:
                        key = joiner(node)
                        #nomination = init_decisions["entry_nom"]["S"][key]\
                        #    [self._steps_per_agent_steps + self._entry_offset]
                        nomination = nominations_t0[count]
                        # epsilon tube implementation
                        # if 'EN' in key:
                        #     if scenario == 1:
                        #         nomination += 50
                        #     elif scenario == 2:
                        #         nomination -= 50
                        # else:
                        #     if scenario == 1:
                        #         nomination -= 50
                        #     elif scenario == 2:
                        #         nomination += 50
                except KeyError:
                    nomination = nominations_t0[count]
                nominations_t1 += [nomination]
        else:
            for count, node in enumerate(obs_no.exits_for_nom + co.special):
                try:
                    if type(node) == str:
                        nomination = init_decisions["exit_nom"]["X"][node]\
                            [self._steps_per_agent_steps]
                    else:
                        key = joiner(node)
                        nomination = init_decisions["entry_nom"]["S"][key]\
                            [self._steps_per_agent_steps + self._entry_offset]
                except KeyError:
                    nomination = nominations_t0[count]
                nominations_t1 += [nomination]

        ## extract the node pressure violations and flows
        # extract all initial values
        init_states = funcs.get_init_scenario()
        # extract the pressure violations
        pressure_violations_upper = [
            np.max([
                init_states[-1]["p"][node] - no.pressure_limits_upper[node],
                0
            ]) for node in self._nodes
        ]
        pressure_violations_lower = [
            np.max([
                no.pressure_limits_lower[node] - init_states[-1]["p"][node],
                0
            ]) for node in self._nodes
        ]
        pressure_violations = pressure_violations_upper + \
                              pressure_violations_lower
        # extract the flow values
        flows = [init_states[-1]["q_out"][pipe]
                 if pipe[1].startswith("X")
                 else init_states[-1]["q_in"][pipe]
                 for pipe in self._pipes]

        ## define the initial state
        self._state = np.array(
            nominations_t0 + nominations_t1 +
            pressure_violations +
            flows
            , np.float32)

        self._episode_ended = False
        self._action_counter = 0

        return ts.restart(self._state)

    def _step(self, actions):

        if self._episode_ended:
            # The last actions ended the episode
            # Therefore the new action can be ignored and the environment
            # is restarted
            return self.reset()

        ### simulate one step
        big_step = self._action_counter
        # convert the action vector such that urmel can use it
        # first get the necessary dictionary syntax
        with open(path.join(data_path, 'init_decisions.yml')) as dec_file:
            agent_decisions = yaml.load(dec_file, Loader=yaml.FullLoader)

        # if random nominations were required, insert it into agent_decisions
        # define the amount of entries and exit
        n_entries_exits = len(obs_no.nodes_with_bds)
        # define the number of entries and exits
        n_exits = len(obs_no.exits_for_nom)
        n_entries = n_entries_exits - n_exits
        current_entry_nominations = self._state[n_exits:n_entries_exits]
        current_exit_nominations = self._state[:n_exits]

        if self._random_nominations:
            # fill the exit nominations if not given in the file
            for count, node in enumerate(obs_no.exits_for_nom):
                try:
                    nomination = agent_decisions["exit_nom"]["X"][node]\
                        [big_step * self._steps_per_agent_steps]
                except KeyError:
                    agent_decisions["exit_nom"]["X"][node]\
                        [big_step * self._steps_per_agent_steps] = \
                        current_exit_nominations[count]
            # fill the entry nominations as calculated randomly before
            for count, node in enumerate(obs_co.special):
                key = joiner(node)
                agent_decisions["entry_nom"]["S"][key] \
                    [big_step * self._steps_per_agent_steps +
                     self._entry_offset] = \
                    current_entry_nominations[count]

        # second handle all actions and convert it to the format
        if self._convert_action:
            action_list = list(self._action_mapping[np.int(actions)])
        else:
            action_list = actions
        n_valves = len(self._valves)
        n_resistors = len(self._resistors)

        agent_step_reward = 0

        step = 0
        agent_step_flow_violation = {}
        pressure_violations = set()
        pressure_violation_values = [0.0]*(2*len(self._nodes))

        for small_step in range(self._steps_per_agent_steps):
            step = big_step * self._steps_per_agent_steps + small_step
            # apply the actions; first fix valves to 1 ( = open)
            # then apply the resistors, then compressors
            for valve_counter in range(n_valves):
                valve = self._valves[valve_counter]
                agent_decisions["va"]["VA"][valve][step] = 1
            for action_counter, action in enumerate(action_list):
                if action_counter < n_resistors:
                    # resistor action has to be converted to discrete [0, 100]
                    resistor = self._resistors[action_counter]
                    resis_value = 100/self._discretization * action
                    agent_decisions["zeta"]["RE"][resistor][step] = resis_value
                else:
                    compressor_index = action_counter - n_resistors
                    compressor = self._compressors[compressor_index]
                    if action > 10e-3:
                        activation = 1
                        efficiency = 1/self._discretization * action
                    else:
                        activation = 0
                        efficiency = 0.0
                    agent_decisions["gas"]["CS"][compressor][step] = efficiency
                    agent_decisions["compressor"]["CS"][compressor][step] = \
                        activation

            # print the actions once for each agent step if desired
            if self._print_actions:
                if step % self._steps_per_agent_steps == 0:
                    for action_counter, action in enumerate(action_list):
                        if action_counter < n_resistors:
                            resistor = self._resistors[action_counter]
                            resis_value = 100 / self._discretization * action
                            print(f"resistor {resistor} works at efficiency"
                                  f" of {resis_value}")
                        else:
                            compressor_index = action_counter - n_resistors
                            compressor = self._compressors[compressor_index]
                            if action > 10e-3:
                                activation = 1
                                efficiency = 1 / self._discretization * action
                            else:
                                activation = 0
                                efficiency = 0.0
                            print(f"compressor {compressor} is "
                                  f"{'' if activation == 1 else 'not '}"
                                  f"activated"
                                  f"{'' if activation == 0 else ' with efficiency ' + str(efficiency)}")

            solution = simulator_step(agent_decisions, step, "sim")

            # if the resulting problem is infeasible we reward -1 for the whole
            # agent step
            if solution is None:
                agent_step_reward = -1
                self._episode_ended = True
            # otherwise we calculate the reward as 1 - the weighted violations
            # divided by the amount of simulation steps per agent step
            else:
                # for norming the violations with their upper bound
                ub_entry_violation = np.abs(int(np.sum(
                    current_entry_nominations
                )))

                # iterate through variables to identify entries and exits
                for variable_name in solution.keys():
                    # deviations from entry flows have to be summed up for each
                    if variable_name.startswith("nom_entry_slack_DA"):
                        # sum up pressure violations for each entry over one
                        # agent  step (may cancel out and is intended)
                        if variable_name not in agent_step_flow_violation:
                            agent_step_flow_violation[variable_name] = 0
                        agent_step_flow_violation[variable_name] += \
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
                                pressure_violations.add(violated_exit)
                                # extract the node index
                                node_index = self._nodes.index(violated_exit)
                                # adjust the node index dependent on ub/lb
                                if variable_name.startswith("lb"):
                                    node_index += len(self._nodes)
                                # add the violation to the respective value
                                pressure_violation_values[node_index] += \
                                    violation

            if self._episode_ended:
                break

        # reward calculation between [-1, 1]
        # each flow violation has an impact of max 1/n_entries and is dependent
        # on the accumulated flow violation
        flow_violation = 0
        for violation in agent_step_flow_violation.values():
            flow_violation += min(
                np.abs(violation/(
                        self._steps_per_agent_steps *
                        ub_entry_violation
                        )),
                1.0
                )/n_entries

        # a pressure violation is rated as critical -> if n = amount exits
        # the ith exit is equal to a violation of 2^(n - i)/(2^n - 1)
        n_press_viol = len(pressure_violations)
        pressure_violation = np.sum([2**(len(no.exits) - i - 1) /
                                     (2**len(no.exits) - 1)
                                     for i in range(n_press_viol)])
        if not self._episode_ended:
            agent_step_reward = 1.0 - (pressure_violation + flow_violation)
            with open("rewardfile.csv", "a+") as rewardcsv:
                rewardcsv.write(str(agent_step_reward) + ";")
        if self._print_actions:
            print(f"This step lead to a reward of {agent_step_reward}")
            print(f"The accumulated flow violations are at "
                  f"{agent_step_flow_violation}")
            print(f"The summed up pressure violations are "
                  f"{pressure_violation}")
            print(f"The nominations for the current step were "
                  f"{self._state[:n_entries_exits]}")

        ## update the state
        # extract the next 'current nominations'
        nominations_t0 = self._state[n_entries_exits:2*n_entries_exits]

        # extract/compute the next 'next time step nominations'
        nominations_t1 = []
        if self._random_nominations:
            # TODO: add random interchange of two scenarios
            for count, node in enumerate(obs_no.exits_for_nom):
                try:
                    nomination = agent_decisions["exit_nom"]["X"][node] \
                        [(self._action_counter + 2) *
                         self._steps_per_agent_steps]
                except KeyError:
                    nomination = nominations_t0[count]
                nominations_t1 += [nomination]

            scenario = random.randint(0, 2)
            # for count, node in enumerate(obs_co.special):
            #     key = joiner(node)
            #     nomination = nominations_t0[n_exits + count]
            #     # if ((('EN' in key and nomination == config["upper_nom_EN"]) or
            #     #     ('EH' in key and nomination == config["upper_nom_EH"]))
            #     #         and scenario == 0):
            #     #     nomination -= 50
            #     # at EN we can reach the upper/lower nomination bound
            #     # second implementation:
            #     # if 'EN' in key:
            #     #     if nomination == config["upper_nom_EN"]:
            #     #         if scenario == 0:
            #     #             nomination -= 50
            #     #     elif nomination == config["lower_nom_EN"]:
            #     #         if scenario == 0:
            #     #             nomination += 50
            #     #     else:
            #     #         if scenario == 1:
            #     #             nomination += 50
            #     #         elif scenario == 2:
            #     #             nomination -= 50
            #     # else:
            #     #     if nomination == config["upper_nom_EH"]:
            #     #         if scenario == 0:
            #     #             nomination -= 50
            #     #     elif nomination == config["lower_nom_EH"]:
            #     #         if scenario == 0:
            #     #             nomination += 50
            #     #     else:
            #     #         if scenario == 1:
            #     #             nomination -= 50
            #     #         elif scenario == 2:
            #     #             nomination += 50
            #
            #     nominations_t1 += [nomination]

            # for count, node in enumerate(no.exits):
            #     try:
            #         nomination = agent_decisions["exit_nom"]["X"][node]\
            #             [(self._action_counter + 1) *
            #              self._steps_per_agent_steps]
            #     except KeyError:
            #         nomination = nominations_t0[count]
            #     nominations_t1 += [nomination]
            #
            change_step = self._max_agent_steps/2 - 2
            if self._action_counter == change_step:
                nomination_sum = [agent_decisions["entry_nom"]["S"][joiner(supply)]
                                  [0 + self._entry_offset]
                                  for supply in obs_co.special]
                nomination_sum = int(np.abs(sum(nomination_sum)))
                n_entries = len(obs_no.nodes_with_bds) - len(obs_no.exits_for_nom)
                breaks = random.choices(range(0, nomination_sum + 1, 50),
                                        k=n_entries - 1)
                breaks.sort()
                breaks = [0] + breaks + [nomination_sum]

                nominations_t1 += [breaks[break_step] - breaks[break_step - 1]
                                   for break_step in range(1, n_entries + 1)]
            else:
                for count, node in enumerate(obs_co.special):
                    nomination = nominations_t0[n_exits + count]
                    nominations_t1 += [nomination]
        else:
            for count, node in enumerate(obs_no.exits_for_nom + co.special):
                try:
                    if type(node) == str:
                        nomination = agent_decisions["exit_nom"]["X"][node]\
                            [(self._action_counter + 2) *
                             self._steps_per_agent_steps]
                    else:
                        key = joiner(node)
                        nomination = agent_decisions["entry_nom"]["S"][key]\
                            [(self._action_counter + 2) *
                             self._steps_per_agent_steps + self._entry_offset]
                except KeyError:
                    nomination = nominations_t0[count]
                nominations_t1 += [nomination]

        # calculate the average pressure violation
        pressure_violation_values = np.divide(pressure_violation_values,
                                              small_step + 1)

        # update the state variables
        if solution is None:
            flows = list(self._state[-len(self._pipes):])
        else:
            flows = [solution["var_pipe_Qo_out[%s,%s]" % pipe]
                            if pipe[1].startswith("X")
                            else solution["var_pipe_Qo_in[%s,%s]" % pipe]
                            for pipe in self._pipes]

        self._state = np.array(
            list(nominations_t0) + nominations_t1 +
            list(pressure_violation_values) +
            flows
            , np.float32)

        self._action_counter += 1

        # test if the episode has ended by infeasibility or maximal amount of
        # steps
        if self._action_counter >= self._max_agent_steps \
                or self._episode_ended:

            self._episode_ended = True
            return ts.termination(self._state, reward=agent_step_reward)
        else:
            return ts.transition(self._state, reward=agent_step_reward,
                                 discount=1.0)

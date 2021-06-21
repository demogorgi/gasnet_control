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

        n_valves = len(self._valves)
        n_resistors = len(self._resistors)
        n_compressors = len(self._compressors)
        n_control_vars = n_valves + n_resistors + n_compressors

        # overall minimum set to 0, maxima are 1 for valves and the
        # discretization step size for others
        self._discretization = discretization_steps
        control_minima = [0]*n_control_vars
        valve_maxima = [1]*n_valves
        # upper resistor bounds [0, discretization] (in reality, [0, 100])
        resistor_maxima = [discretization_steps]*n_resistors
        compressor_maxima = [discretization_steps]*n_compressors
        control_maxima = valve_maxima + resistor_maxima + compressor_maxima

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
        ## extract the nominations
        entries_exits_list = no.nodes_with_bds
        entries_exits_minima = [no.q_lb[node] for node in no.nodes_with_bds]
        entries_exits_maxima = [no.q_ub[node] for node in no.nodes_with_bds]
        n_entries_exits = len(entries_exits_list)

        ## extract the network state specifities
        # get all nodes and pipes but exclude helper elements
        # all nodes in the relevant network
        nodes_list = [node for node in no.nodes if node.startswith('N')]
        # entries and exits of the relevant network
        nodes_list += no.nodes_with_bds
        # only (non-)pipes where one element starts with 'N' are observable
        pipes_list = [pipe for pipe in co.pipes
                      if pipe[0].startswith('N') or pipe[1].startswith('N')]
        non_pipes_list = [non_pipe for non_pipe in co.non_pipes
                          if non_pipe[0].startswith('N') or
                          non_pipe[1].startswith('N')]

        n_nodes = len(nodes_list)
        n_pipes = len(pipes_list)
        n_non_pipes = len(non_pipes_list)

        # extract the pressure ranges
        node_pressure_minima = [no.pressure_limits_lower[node] for node in
                                nodes_list]
        # TODO: check why initial pressures are higher than upper bound
        node_pressure_maxima = [no.pressure_limits_upper[node] + 3.0 for node in
                                nodes_list]
        # set the inflow ranges, TODO: ask if necessary and extract from file
        #node_inflow_minima = [-10000]*n_nodes
        #node_inflow_maxima = [10000]*n_nodes
        # extract in and non pipe infos, TODO: extract from file?
        pipe_in_minima = [-10000]*n_pipes
        pipe_in_maxima = [10000]*n_pipes
        non_pipe_minima = [-10000]*n_non_pipes
        non_pipe_maxima = [10000]*n_non_pipes

        # define the actual observation spec
        # 2 times entries and exits since we want to have nominations of time 0
        # and time 1
        n_observations = 2*n_entries_exits + n_nodes + n_pipes + n_non_pipes
        observation_minima = 2*entries_exits_minima + node_pressure_minima + \
                             pipe_in_minima + non_pipe_minima
                             #node_inflow_minima to be added on third position

        observation_maxima = 2*entries_exits_maxima + node_pressure_maxima + \
                             pipe_in_maxima + non_pipe_maxima
                             #node_inflow_maxima to be added on third position

        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(n_observations,), dtype=np.float32,
            minimum=np.array(observation_minima),
            maximum=np.array(observation_maxima),
            name='observation'
        )

        self._nodes = nodes_list
        self._pipes = pipes_list
        self._non_pipes = non_pipes_list
        # define the initial state (initial network + nominations)
        # extract the initial nominations and if given for the next time step
        nominations_t0 = [init_decisions["exit_nom"]["X"][ex][0]
                          for ex in no.exits]
        if self._random_nominations:
            nomination_sum = int(np.abs(sum(nominations_t0)))
            n_entries = len(no.nodes_with_bds) - len(no.exits)
            breaks = random.choices(range(0, nomination_sum + 1, 50),
                                    k=n_entries - 1)
            breaks.sort()
            breaks = [0] + breaks + [nomination_sum]

            nominations_t0 += [breaks[break_step] - breaks[break_step - 1]
                               for break_step in range(1, n_entries + 1)]
        else:
            nominations_t0 += [init_decisions["entry_nom"]["S"][joiner(supply)]
                               [0 + self._entry_offset]
                               for supply in co.special]
        # length of nominations has to be the same as in the observation specs
        assert(len(nominations_t0) == n_entries_exits)

        nominations_t1 = []

        if self._random_nominations:
            for count, node in enumerate(no.exits):
                try:
                    nomination = init_decisions["exit_nom"]["X"][node]\
                        [self._steps_per_agent_steps]
                except KeyError:
                    nomination = nominations_t0[count]
                nominations_t1 += [nomination]

            breaks = random.choices(range(0, nomination_sum + 1, 50),
                                   k=n_entries - 1)
            breaks.sort()
            breaks = [0] + breaks + [nomination_sum]
            nominations_t1 += [breaks[break_step] - breaks[break_step - 1]
                               for break_step in range(1, n_entries + 1)]
        else:
            for count, node in enumerate(no.exits + co.special):
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

        # extract the initial node pressure and pipe inflow as well as the
        # initial values for non pipe elements
        init_states = funcs.get_init_scenario()
        node_pressures = [init_states[-1]["p"][node] for node in nodes_list]
        pipe_inflows = [init_states[-1]["q_in"][pipe] for pipe in pipes_list]
        non_pipe_values = [init_states[-1]["q"][non_pipe]
                           for non_pipe in non_pipes_list]

        self._state = np.array(
                        nominations_t0 + nominations_t1 + node_pressures + \
                        pipe_inflows + non_pipe_values
                        , np.float32)

        # ## debugging
        # for i, val in enumerate(self._state):
        #     if val > observation_maxima[i]:
        #         print(f"error detected at index {i}")
        #     if val < observation_minima[i]:
        #         print(f"error detected at index {i}")
        # ## end of debugging
        self._episode_ended = False

        # counter to come to an end
        self._action_counter = 0
        # set simulation_step counter in urmel to 0
        #simulator_step.counter = 0

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        # TODO: check functionality in learning
        # extract initial decisions/values
        with open(path.join(data_path, 'init_decisions.yml')) as init_file:
            init_decisions = yaml.load(init_file, Loader=yaml.FullLoader)
        # extract the initial nominations and if given for the next time step
        nominations_t0 = [init_decisions["exit_nom"]["X"][ex][0]
                          for ex in no.exits]

        if self._random_nominations:
            nomination_sum = int(np.abs(sum(nominations_t0)))
            n_entries = len(no.nodes_with_bds) - len(no.exits)
            breaks = random.choices(range(0, nomination_sum + 1, 50),
                                    k=n_entries - 1)
            breaks.sort()
            breaks = [0] + breaks + [nomination_sum]

            nominations_t0 += [breaks[break_step] - breaks[break_step - 1]
                               for break_step in range(1, n_entries + 1)]
        else:
            nominations_t0 += [init_decisions["entry_nom"]["S"][joiner(supply)]
                               [0 + self._entry_offset]
                               for supply in co.special]

        nominations_t1 = []

        if self._random_nominations:
            for count, node in enumerate(no.exits):
                try:
                    nomination = init_decisions["exit_nom"]["X"][node]\
                        [self._steps_per_agent_steps]
                except KeyError:
                    nomination = nominations_t0[count]
                nominations_t1 += [nomination]

            breaks = random.choices(range(0, nomination_sum + 1, 50),
                                   k=n_entries - 1)
            breaks.sort()
            breaks = [0] + breaks + [nomination_sum]
            nominations_t1 += [breaks[break_step] - breaks[break_step - 1]
                               for break_step in range(1, n_entries + 1)]
        else:
            for count, node in enumerate(no.exits + co.special):
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

        # extract the initial node pressure and inflow as well as the
        # initial values for non pipe elements
        init_states = funcs.get_init_scenario()
        node_pressures = [init_states[-1]["p"][node] for node in self._nodes]
        pipe_inflows = [init_states[-1]["q_in"][pipe] for pipe in self._pipes]
        non_pipe_values = [init_states[-1]["q"][non_pipe]
                           for non_pipe in self._non_pipes]

        self._state = np.array(
                        nominations_t0 + nominations_t1 + node_pressures + \
                        pipe_inflows + non_pipe_values
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

        #print(f"performing an agent step number {self._action_counter}")
        ### simulate one step
        big_step = self._action_counter
        # convert the action vector such that urmel can use it
        # first get the necessary dictionary syntax
        with open(path.join(data_path, 'init_decisions.yml')) as dec_file:
            agent_decisions = yaml.load(dec_file, Loader=yaml.FullLoader)

        # if random nominations were required, insert it into agent_decisions
        # define the amount of entries and exit
        n_entries_exits = len(no.nodes_with_bds)
        # define the number of entries and exits
        n_exits = len(no.exits)
        n_entries = n_entries_exits - n_exits
        current_entry_nominations = self._state[n_exits:n_entries_exits]
        current_exit_nominations = self._state[:n_exits]

        if self._random_nominations:
            # fill the exit nominations if not given in the file
            for count, node in enumerate(no.exits):
                try:
                    nomination = agent_decisions["exit_nom"]["X"][node]\
                        [big_step * self._steps_per_agent_steps]
                except KeyError:
                    agent_decisions["exit_nom"]["X"][node]\
                        [big_step * self._steps_per_agent_steps] = \
                        current_exit_nominations[count]
            # fill the entry nominations as calculated randomly before
            for count, node in enumerate(co.special):
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

        for small_step in range(self._steps_per_agent_steps):
            step = big_step * self._steps_per_agent_steps + small_step
            # apply the actions; first valves, then resistors, then compressors
            for action_counter, action in enumerate(action_list):
                if action_counter < n_valves:
                    # valves can only be 0 or 1
                    valve = self._valves[action_counter]
                    agent_decisions["va"]["VA"][valve][step] = action
                elif action_counter < n_valves + n_resistors:
                    # resistor action has to be converted to discrete [0, 100]
                    resistor = self._resistors[action_counter - n_valves]
                    resis_value = 100/self._discretization * action
                    agent_decisions["zeta"]["RE"][resistor][step] = resis_value
                else:
                    compressor_index = action_counter - n_valves - n_resistors
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
                        if action_counter < n_valves:
                            valve = self._valves[action_counter]
                            print(f"valve {valve} is "
                                  f"{'' if action == 1 else 'not '}activated")
                        elif action_counter < n_valves + n_resistors:
                            resistor = self._resistors[
                                action_counter - n_valves]
                            resis_value = 100 / self._discretization * action
                            print(f"resistor {resistor} works at efficiency"
                                  f" of {resis_value}")
                        else:
                            compressor_index = action_counter - \
                                               n_valves - n_resistors
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
                step_reward = 0
                self._episode_ended = True
            # otherwise we calculate the reward as 1 - the weighted violations
            # divided by the amount of simulation steps per agent step
            else:
                # if nothing is violated, we reward 1 for the simulation step
                step_reward = 1.0

                # one can weigh the impact of exit to entry violations
                # ratio of 10 means an exit violation has 10 times more
                # percentage impact than an entry violation of the same perc.
                exit_entry_impact_ratio = 2

                # for norming the violations with their upper bound
                ub_entry_violation = np.abs(int(np.sum(
                    current_entry_nominations
                )))
                ub_exit_violation = 430

                # define the resulting multipliers (see thesis)
                entry_multiplier = 1 / (n_entries +
                                        exit_entry_impact_ratio * n_exits)
                exit_multiplier = 1 / (n_entries / exit_entry_impact_ratio +
                                       n_exits)

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

        # a pressure violation is rated is critical -> if n = amount exits
        # the ith exit is equal to a violation of 2^(n - i)/(2^n - 1)
        n_press_viol = len(pressure_violations)
        pressure_violation = np.sum([2**(n_press_viol - i - 1) /
                                     (2**n_press_viol - 1)
                                     for i in range(n_press_viol)])
        if not self._episode_ended:
            agent_step_reward = 1.0 - (pressure_violation + flow_violation)
        if self._print_actions:
            print(f"This step lead to a reward of {agent_step_reward}")
            print(f"The accumulated flow violations are at "
                  f"{agent_step_flow_violation}")
            print(f"The summed up pressure violations are "
                  f"{pressure_violation}")
            print(f"The nominations for the current step were "
                  f"{self._state[:n_entries_exits]}")
        # extract the nominations for updating the state
        nominations_t0 = self._state[n_entries_exits:2*n_entries_exits]

        nominations_t1 = []
        if self._random_nominations:
            for count, node in enumerate(no.exits):
                try:
                    nomination = agent_decisions["exit_nom"]["X"][node]\
                        [(self._action_counter + 1) *
                         self._steps_per_agent_steps]
                except KeyError:
                    nomination = nominations_t0[count]
                nominations_t1 += [nomination]

            nomination_sum = int(np.abs(sum(nominations_t1)))
            n_entries = n_entries_exits - len(no.exits)
            breaks = random.choices(range(0, nomination_sum + 1, 50),
                                   k=n_entries - 1)
            breaks.sort()
            breaks = [0] + breaks + [nomination_sum]
            nominations_t1 += [breaks[break_step] - breaks[break_step - 1]
                               for break_step in range(1, n_entries + 1)]
        else:
            for count, node in enumerate(no.exits + co.special):
                try:
                    if type(node) == str:
                        nomination = agent_decisions["exit_nom"]["X"][node]\
                            [(self._action_counter + 1) *
                             self._steps_per_agent_steps]
                    else:
                        key = joiner(node)
                        nomination = agent_decisions["entry_nom"]["S"][key]\
                            [(self._action_counter + 1) *
                             self._steps_per_agent_steps]
                except KeyError:
                    nomination = nominations_t0[count]
                nominations_t1 += [nomination]

        # update the state variables
        if solution is None:
            node_pressures = list(self._state[
                2*n_entries_exits:2*n_entries_exits + len(self._nodes)
            ])
            pipe_inflows = list(self._state[
                -len(self._pipes) - len(self._non_pipes) : -len(self._non_pipes)
            ])
            non_pipe_values = list(self._state[
                -len(self._non_pipes):
            ])
        else:
            node_pressures = [solution["var_node_p[%s]" % node]
                              for node in self._nodes]
            pipe_inflows = [solution["var_pipe_Qo_in[%s,%s]" % pipe]
                            for pipe in self._pipes]
            non_pipe_values = [solution["var_non_pipe_Qo[%s,%s]" % non_pipe]
                               for non_pipe in self._non_pipes]

        self._state = np.array(
            list(nominations_t0) + nominations_t1 + node_pressures +
            pipe_inflows + non_pipe_values
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

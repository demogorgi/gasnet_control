# standard imports as suggested at
# https://www.tensorflow.org/agents/tutorials/2_environments_tutorial?hl=en#python_environments
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np

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
from urmel import *
from model import *
import functions as funcs
from params import *


class GasNetworkEnv(py_environment.PyEnvironment):

    def __init__(self, discretization_steps):
        ### define the action specificities
        # analyse initial decisions to extract values
        with open(path.join(data_path, 'init_decisions.yml')) as file:
            init_decisions = yaml.load(file, Loader=yaml.FullLoader)

        # safe the control variable names for later mapping
        self._valves = init_decisions["va"]["VA"].keys
        self._resistors = init_decisions["zeta"]["RE"].keys
        self._compressors = init_decisions["compressor"]["CS"].keys

        n_valves = len(self._valves)
        n_resistors = len(self._resistors)
        n_compressors = len(self._compressors)
        n_control_vars = n_valves + n_resistors + n_compressors

        # overall minimum set to 0, maxima are 1 for valves and the
        # discretization step size for others
        self._discretization = discretization_steps
        control_minima = 0
        valve_maxima = [1]*n_valves
        resistor_maxima = [discretization_steps]*n_resistors
        compressor_maxima = [discretization_steps]*n_compressors
        control_maxima = valve_maxima + resistor_maxima + compressor_maxima

        # define the actual action spec
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(n_control_vars,),
            dtype=np.int32,
            minimum=control_minima,
            maximum=control_maxima,
            name='action'
        )
        ### define the observations specificities
        ## extract the nominations
        entries_exits_list = no.nodes_with_bds
        entries_exits_minima = [no.q_lb[node] for node in no.nodes_with_bds]
        entries_exits_maxima = [no.q_ub[node] for node in no.nodes_with_bds]
        n_entries_exits = len(entries_exits_list)

        ## extract the network state specifities
        # get all nodes and pipes
        nodes_list = no.nodes
        pipes_list = co.pipes
        non_pipes_list = co.non_pipes
        # nodes_list = [node for node in no.nodes if 'aux' not in node]
        n_nodes = len(nodes_list)
        n_pipes = len(pipes_list)
        n_non_pipes = len(non_pipes_list)

        # extract the pressure ranges
        node_pressure_minima = [no.pressure_limits_lower[node] for node in
                                nodes_list]
        node_pressure_maxima = [no.pressure_limits_upper[node] for node in
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
            minimum=observation_minima,
            maximum=observation_maxima,
            name='observation'
        )
        # define the initial state (initial network + nominations)
        # TODO: insert a vector with initial values for all observations
        # extract the initial nominations and if given for the next time step
        nominations_t0 = [init_decisions["exit_nom"]["X"][ex][0]
                          for ex in no.exits]
        nominations_t0 += [init_decisions["entry_nom"]["S"][joiner(supply)][0]
                           for supply in co.special]
        # length of nominations has to be the same as in the observation specs
        assert(len(nominations_t0) == n_entries_exits)

        nominations_t1 = []
        for count, node in enumerate(no.exits + co.special):
            try:
                if len(node) == 1:
                    nomination = init_decisions["exit_nom"]["X"][node][1]
                else:
                    key = joiner(node)
                    nomination = init_decisions["entry_nom"]["S"][key][1]
            except KeyError:
                nomination = nominations_t0[count]
            nominations_t1 += [nomination]

        # extract the initial node pressure and inflow as well as the
        # initial values for non pipe elements
        states = funcs.get_init_scenario()
        node_pressures = [states[-1]["p"][node] for node in no.nodes]
        pipe_inflows = [states[-1]["q_in"][pipe] for pipe in co.pipes]
        non_pipe_values = [states[-1]["q"][non_pipe]
                           for non_pipe in co.non_pipes]

        self._state = nominations_t0 + nominations_t1 + node_pressures + \
                      pipe_inflows + non_pipe_values

        self._episode_ended = False

        # counter to come to an end
        self._action_counter = 0

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        # TODO: check functionality in learning
        # extract initial decisions/values
        with open(path.join(data_path, 'init_decisions.yml')) as file:
            init_decisions = yaml.load(file, Loader=yaml.FullLoader)
        # extract the initial nominations and if given for the next time step
        nominations_t0 = [init_decisions["exit_nom"]["X"][ex][0]
                          for ex in no.exits]
        nominations_t0 += [init_decisions["entry_nom"]["S"][joiner(supply)][0]
                           for supply in co.special]

        nominations_t1 = []
        for count, node in enumerate(no.exits + co.special):
            try:
                if len(node) == 1:
                    nomination = init_decisions["exit_nom"]["X"][node][1]
                else:
                    key = joiner(node)
                    nomination = init_decisions["entry_nom"]["S"][key][1]
            except KeyError:
                nomination = nominations_t0[count]
            nominations_t1 += [nomination]

        # extract the initial node pressure and inflow as well as the
        # initial values for non pipe elements
        states = funcs.get_init_scenario()
        node_pressures = [states[-1]["p"][node] for node in no.nodes]
        pipe_inflows = [states[-1]["q_in"][pipe] for pipe in co.pipes]
        non_pipe_values = [states[-1]["q"][non_pipe]
                           for non_pipe in co.non_pipes]

        self._state = nominations_t0 + nominations_t1 + node_pressures + \
                      pipe_inflows + non_pipe_values

        self._episode_ended = False

        return ts.restart(self._state)

    def _step(self, action):

        if self._episode_ended:
            # The last actions ended the episode
            # Therefore the new action can be ignored and the environment
            # is restarted
            return self.reset()

        # TODO: Implement a reasonable time step counter for the max amount of
        # TODO: steps
        if self._action_counter > 8:
            self._episode_ended = True
            # TODO: calculate reward, maybe dependent on earlier steps
            reward = 0
            return ts.termination(self._state, reward)
        else:
            ### simulate one step
            # convert the action vector such that urmel can use it
            # first get the necessary dictionary syntax
            with open(path.join(data_path, 'init_decisions.yml')) as file:
                agent_decisions = yaml.load(file, Loader=yaml.FullLoader)

            # second handle all actions and convert it to the format
            step = self._action_counter
            n_valves = len(self._valves)
            n_resistors = len(self._resistors)
            for action_counter, action in enumerate(action):
                if action_counter < n_valves:
                    valve = self._valves[action_counter]
                    agent_decisions["va"]["VA"][valve][step] = action
                elif action_counter < n_valves + n_resistors:
                    resistor = self._resistors[action_counter - n_valves]
                    # TODO: clarify resistor handling and value calculation
                    agent_decisions["zeta"]["RE"][resistor][step] = action
                else:
                    compressor_index = action_counter - n_valves - n_resistors
                    compressor = self._compressors[compressor_index]
                    if action > 10e-3:
                        activation = 1
                        efficiency = 100/self._discretization * action
                    else:
                        activation = 0
                        efficiency = 0.0
                    agent_decisions["gas"]["CS"][compressor][step] = efficiency
                    agent_decisions["gas"]["CS"][compressor][step] = activation

            solution = simulator_step(agent_decisions, step, "sim")
            if solution is None:
                reward = 0
            else:
                reward = 100
                for variable_name in solution.keys():
                    if variable_name == "scenario_balance_TA":
                        reward -= solution[variable_name]
                    elif variable_name.endswith("b_pressure_violation_DA"):
                        reward -= solution[variable_name]

            self._action_counter += 1
            return ts.transition(self._state, reward=reward, discount=1.0)

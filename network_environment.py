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
from model import *
#import functions as funcs
from params import *

class GasNetworkEnv(py_environment.PyEnvironment):

    def __init__(self, discretization_steps):
        ### define the action specificities
        # analyse initial decisions to extract action types and amount
        with open(path.join(data_path, 'init_decisions.yml')) as file:
            agent_decisions = yaml.load(file, Loader=yaml.FullLoader)

        n_valves = len(agent_decisions["va"]["VA"])
        n_compressors = len(agent_decisions["compressor"]["CS"])
        n_resistors = len(agent_decisions["zeta"]["RE"])
        n_control_vars = n_valves + n_compressors + n_resistors

        # overall minimum set to 0, maxima are 1 for valves and the
        # discretization step size for others
        control_minima = 0
        valve_maxima = [1]*n_valves
        compressor_maxima = [discretization_steps]*n_compressors
        resistor_maxima = [discretization_steps]*n_resistors
        control_maxima = valve_maxima + compressor_maxima + resistor_maxima

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
        node_inflow_minima = [-10000]*n_nodes
        node_inflow_maxima = [10000]*n_nodes
        # extract in and non pipe infos, TODO: extract from file?
        pipe_in_minima = [-10000]*n_pipes
        pipe_in_maxima = [10000]*n_pipes
        non_pipe_minima = [-10000]*n_non_pipes
        non_pipe_maxima = [10000]*n_non_pipes

        # define the actual observation spec
        n_observations = n_entries_exits + n_nodes + n_pipes + n_non_pipes
        observation_minima = entries_exits_minima + node_pressure_minima + \
                             node_inflow_minima + pipe_in_minima + \
                             non_pipe_minima
        observation_maxima = entries_exits_maxima + node_pressure_maxima + \
                             node_inflow_maxima + pipe_in_maxima + \
                             non_pipe_maxima
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(n_observations,), dtype=np.float32,
            minimum=observation_minima,
            maximum=observation_maxima,
            name='observation'
        )
        # define the initial state (initial network + nominations)
        states = funcs.get_init_scenario()

        self._episode_ended = False

    def action_spec(self):# -> types.NestedArraySpec:
        pass

    def observation_spec(self):# -> types.NestedArraySpec:
        pass

    def _reset(self) -> ts.TimeStep:
        pass

    def _step(self, action):
        pass
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
#import model
#import functions as funcs
from params import *

class GasNetworkEnv(py_environment.PyEnvironment):

    def __init__(self, discretization_steps):
        ### define the action specificities
        # analyse initial decisions to extract action types and amount
        with open(path.join(data_path, 'init_decisions.yml')) as file:
            agent_decisions = yaml.load(file, Loader=yaml.FullLoader)

        n_valves = len(agent_decisions["va"])
        n_compressors = len(agent_decisions["compressor"])
        n_resistors = len(agent_decisions["zeta"])
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
        # define the observations specificities

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
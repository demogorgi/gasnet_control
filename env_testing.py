from tf_agents.environments import utils

import network_environment

environment = network_environment.GasNetworkEnv(10)
utils.validate_py_environment(environment)

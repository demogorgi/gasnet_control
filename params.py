#File contains global variables, values and functions
import sys
from os import path
import os
import yaml
import re
import csv

from datetime import datetime, timedelta

data_path = sys.argv[1]
numSteps  = int(sys.argv[2])
dt        = int(sys.argv[3])

timestep = datetime.now()

# default configs which are merged with instance configuration
config = {
    # prefix for output filenames
    "name": "urmel",
    # debug mode with more output
    "debug": False,
    # write new initial state
    "new_init_scenario": False,
    # write problem files in the lp-format?
    "write_lp": False,
    # write solution files in the sol-format?
    "write_sol": False,
    # write irreducible infeasibility set if problem is infeasible?
    "write_ilp": True,
    # write wheel maps with gnuplot?
    "gnuplot": False,
    # console output?
    "urmel_console_output": False,
    # gurobi logfile
    "grb_logfile": "gurobi.log",
    # gurobi console output
    "grb_console": False,
    # contour output (net- and state-files in contour folder)
    "contour_output": False,
    # how often new trader nomination comes (number of timesteps)
    "nomination_freq": 8,
    # controlling random nomination values [start(opt),stop,step(opt)]
    "randrange": [0,1101,50],
    # Is AI control active?
    "ai": False
}

# read manual file with configs
# the dictionary does not change during the process
if os.path.exists(os.path.join(data_path, "config.yml")):
    with open(os.path.join(data_path, 'config.yml')) as file:
        ymlConfig = yaml.load(file, Loader=yaml.FullLoader)
        merged = {**config, **ymlConfig}
        config = merged
        print(config)

# read manual file with compressor data
# the dictionary does not change during the process
with open(path.join(data_path, 'compressors.yml')) as file:
    compressors = yaml.load(file, Loader=yaml.FullLoader)
    #print(compressors)

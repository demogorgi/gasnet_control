#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# this file manages the iterative process
# it works for example with python3, gurobi 8.0.1, yaml 5.3
# >python3 simulate.py path numIterations lengthTimestep

import importlib
import re
import shutil
import gurobipy as gp
from gurobipy import GRB
from constants import *
from functions import *
from minlp import *
from params import *
import pprint
from deepmerge import always_merger
import random

# basic argument check
rem = int(sys.argv[2]) % config['nomination_freq']
if rem:
    raise ValueError(
        "Nomination frequency is not a divisor of the number of iterations ({} % {} = {})".format(
            sys.argv[2], config['nomination_freq'], rem))

# read manual file with initial gas network control
# the dictionary changes with every new control
with open(path.join(data_path, 'init_decisions.yml')) as file:
    agent_decisions = yaml.load(file, Loader=yaml.FullLoader)
    print(agent_decisions)

process_type = "sim"

#### START OF URMEL PART

# m ist the simulator model with agent decisisons, compressor specs and timestep length incorporated
m = simulate(agent_decisions, compressors, dt, 10)
# control output
m.params.logToConsole = config['grb_console']
m.params.logfile = config['grb_logfile']
# tuned parameters
m.params.heuristics = 0
m.params.cuts = 0
m.optimize()
# get the model status
status = m.status
# if solved to optimallity
if config['urmel_console_output']:
    print("model status: ", status)
if status == GRB.OPTIMAL:  # == 2
    # store solution in dictionary
    sol = {}
    for v in m.getVars():
        sol[v.varName] = v.x
        if config['urmel_console_output']:
            print('%s %g' % (v.varName, v.x))

# if infeasible write IIS for analysis and debugging
elif status == GRB.INFEASIBLE:
    if config['write_ilp'] and (process_type == "sim" or config["debug"]):
        if config['urmel_console_output']:
            print("Model is infeasible. %s.lp/ilp written." % config['name'])
    else:
        if config['urmel_console_output']:
            print("Model is infeasible.")

# don't know yet, what else
else:
    if config['urmel_console_output']:
        print("Solution status is %d, don't know what to do." % status)

# END OF URMEL PART

# generate contour output
if config["contour_output"]:
    if not config["write_sol"]:
        print(
            "WARNING: Config parameter \"write_sol\" needs to be True if contour_output is True.")
    else:
        os.system("ruby sol2state.rb {} {}".format(data_path, dt))

# concat all compressor pdfs to a single one
if compressors and config["gnuplot"]:
    p = path.join(data_path, "output/")
    os.system("pdftk " + p + "*.pdf cat output " + p + "all.pdf")
    print("pdftk " + path.join(data_path,
                               "output/*.pdf") + " cat output all.pdf")

print("\n\n>> finished")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# this file manages the iterative process
# it works for example with python3, gurobi 8.0.1, yaml 5.3
# >python3 main.py path numIterations lengthTimestep

from urmel import *
import pprint
from deepmerge import always_merger
import random

# basic argument check
rem = int(sys.argv[2]) % config['nomination_freq']
if rem:
	raise ValueError("Nomination frequency is not a divisor of the number of iterations ({} % {} = {})".format(sys.argv[2],config['nomination_freq'],rem))

# read manual file with initial gas network control
# the dictionary changes with every new control
with open(path.join(data_path, 'init_decisions.yml')) as file:
    agent_decisions = yaml.load(file, Loader=yaml.FullLoader)
    print(agent_decisions)

simulator_step.counter = 0
for i in range(numSteps):
    print("step %d" % i)

    # dirty hack to randomly generate nominations
    if i > 0 and (i+1) % config['nomination_freq'] == 0 and (i+1) < numSteps:
        a = random.randrange(*config["randrange"])
        agent_decisions["entry_nom"]["S"]["EN_aux0^EN"][i+1] = a
        agent_decisions["entry_nom"]["S"]["EH_aux0^EH"][i+1] = 1100 - a

    # for every i in numSteps a simulator step is performed.
    # agent_decisions (init_decisions.yml in scenario folder for the first step) delivers the agents decisions to the simulator and can be modified for every step.
    # i is the step number (neccessary for naming output files if any).
    # If the last argument "porcess_type" is "sim" files (sol, lp, ... ) will be written if their option is set.
    # If the last argument "porcess_type" is not "sim" files will only be written if their option is set and if config["debug"] is True.
    solution = simulator_step(agent_decisions, i, "sim")

    ##############################################################################
    # This is the place where the AI comes into play.
    # The solution should contain all information to compute penalties.
    # The agent_decisions-dictionary can be adjusted here.
    ##############################################################################

# generate contour output
if config["contour_output"]:
    if not config["write_sol"]:
        print("WARNING: Config parameter \"write_sol\" needs to be True if contour_output is True.")
    else:
        os.system("ruby sol2state.rb " + data_path)

# concat all compressor pdfs to a single one
if config["gnuplot"]:
    p = path.join(data_path, "output/")
    os.system("pdftk " + p + "*.pdf cat output " + p + "all.pdf")
    print("pdftk " + path.join(data_path, "output/*.pdf") + " cat output all.pdf")

print("\n\n>> finished")

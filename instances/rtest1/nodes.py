""" auto-generated python script by create_netstate """
  
import gurobipy as gp
from gurobipy import GRB

# set up Q boundaries
nodes_with_bds, q_lb, q_ub = gp.multidict({
	'START': [0, 10000],
	'END': [-1000, 0]
})

# set up entries
entries, entry_flow_bound, pressure = gp.multidict({
	'START_ND': [1000, 181.01325],
	'START_HD': [1, 181.01325],
	'START_1_ND': [1000, 51.01325],
	'START_1_HD': [1000, 51.01325]
})

# set up exits
exits = ['END_1', 'END']

# set up innodes
innodes = ['N363_1', 'N366_1', 'N365_1', 'N364_1', 'N376_1', 'N380_1', 'N379_1', 'N378_1', 'N377_1', 'N381_1', 'B_1', 'N382_1', 'N383_1', 'N384_1', 'N322_1', 'A_1', 'N384', 'N383', 'N382', 'N381', 'N380', 'N379', 'N378', 'N377', 'N376', 'N366', 'N365', 'N364', 'N363', 'N322', 'B', 'A', 'START', 'START_aux0', 'START_aux1', 'START_NDin1', 'START_NDin2', 'START_aux2', 'START_aux3', 'START_HDin1', 'START_HDin2', 'START_1', 'START_1_aux0', 'START_1_aux1', 'START_1_NDin1', 'START_1_NDin2', 'START_1_aux2', 'START_1_aux3', 'START_1_HDin1', 'START_1_HDin2', 'B_aux']

# set up nodes heights and pressure limits
nodes, heights, pressure_limits_lower, pressure_limits_upper = gp.multidict({
	# innodes
	'N363_1': [0.0, 1.01325, 105.01325],
	'N366_1': [0.0, 1.01325, 105.01325],
	'N365_1': [0.0, 1.01325, 105.01325],
	'N364_1': [0.0, 1.01325, 105.01325],
	'N376_1': [0.0, 1.01325, 105.01325],
	'N380_1': [0.0, 1.01325, 105.01325],
	'N379_1': [0.0, 1.01325, 105.01325],
	'N378_1': [0.0, 1.01325, 105.01325],
	'N377_1': [0.0, 1.01325, 105.01325],
	'N381_1': [0.0, 1.01325, 105.01325],
	'B_1': [0.0, 1.01325, 105.01325],
	'N382_1': [0.0, 1.01325, 105.01325],
	'N383_1': [0.0, 1.01325, 105.01325],
	'N384_1': [0.0, 1.01325, 105.01325],
	'N322_1': [0.0, 1.01325, 105.01325],
	'A_1': [0.0, 1.01325, 105.01325],
	'N384': [0.0, 1.01325, 105.01325],
	'N383': [0.0, 1.01325, 105.01325],
	'N382': [0.0, 1.01325, 105.01325],
	'N381': [0.0, 1.01325, 105.01325],
	'N380': [0.0, 1.01325, 105.01325],
	'N379': [0.0, 1.01325, 105.01325],
	'N378': [0.0, 1.01325, 105.01325],
	'N377': [0.0, 1.01325, 105.01325],
	'N376': [0.0, 1.01325, 105.01325],
	'N366': [0.0, 1.01325, 105.01325],
	'N365': [0.0, 1.01325, 105.01325],
	'N364': [0.0, 1.01325, 105.01325],
	'N363': [0.0, 1.01325, 105.01325],
	'N322': [0.0, 1.01325, 105.01325],
	'B': [0.0, 1.01325, 105.01325],
	'A': [0.0, 1.01325, 105.01325],
	'START': [0.0, 1.01325, 105.01325],
	'START_aux0': [0.0, 1.01325, 81.01325],
	'START_aux1': [0.0, 1.01325, 81.01325],
	'START_NDin1': [0.0, 1.01325, 81.01325],
	'START_NDin2': [0.0, 1.01325, 81.01325],
	'START_aux2': [0.0, 1.01325, 81.01325],
	'START_aux3': [0.0, 1.01325, 81.01325],
	'START_HDin1': [0.0, 1.01325, 81.01325],
	'START_HDin2': [0.0, 1.01325, 81.01325],
	'START_1': [0.0, 1.01325, 105.01325],
	'START_1_aux0': [0.0, 1.01325, 51.01325],
	'START_1_aux1': [0.0, 1.01325, 51.01325],
	'START_1_NDin1': [0.0, 1.01325, 51.01325],
	'START_1_NDin2': [0.0, 1.01325, 51.01325],
	'START_1_aux2': [0.0, 1.01325, 51.01325],
	'START_1_aux3': [0.0, 1.01325, 51.01325],
	'START_1_HDin1': [0.0, 1.01325, 51.01325],
	'START_1_HDin2': [0.0, 1.01325, 51.01325],
	'B_aux': [0, 1.01325, 105.01325],
	# boundary nodes
	'END_1': [0.0, 1.01325, 105.01325],
	'END': [0.0, 1.01325, 105.01325],
	'START_ND': [0.0, 1.01325, 81.01325],
	'START_HD': [0.0, 1.01325, 81.01325],
	'START_1_ND': [0.0, 1.01325, 51.01325],
	'START_1_HD': [0.0, 1.01325, 51.01325]
})

# all nodes
nodes = entries + exits + innodes
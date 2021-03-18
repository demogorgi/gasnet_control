""" auto-generated python script by create_netstate """
  
import gurobipy as gp
from gurobipy import GRB

# set up Q boundaries
nodes_with_bds, q_lb, q_ub = gp.multidict({
	'END': [-1000, 0],
	'START': [0, 10000]
})

# set up entries
entries, entry_flow_bound, pressure = gp.multidict({
	'START_ND': [200, 70.01325],
	'START_HD': [1000, 61.01325]
})

# set up exits
exits = ['END']

# set up innodes
innodes = ['B', 'A', 'START', 'START_aux0', 'START_aux1', 'START_NDin1', 'START_NDin2', 'START_aux2', 'START_aux3', 'START_HDin1', 'START_HDin2', 'B_aux']

# set up nodes heights and pressure limits
nodes, heights, pressure_limits_lower, pressure_limits_upper = gp.multidict({
	# innodes
	'B': [0.0, 1.01325, 105.01325],
	'A': [0.0, 1.01325, 105.01325],
	'START': [0.0, 1.01325, 105.01325],
	'START_aux0': [0.0, 1.01325, 70.01325],
	'START_aux1': [0.0, 1.01325, 70.01325],
	'START_NDin1': [0.0, 1.01325, 70.01325],
	'START_NDin2': [0.0, 1.01325, 70.01325],
	'START_aux2': [0.0, 1.01325, 70.01325],
	'START_aux3': [0.0, 1.01325, 70.01325],
	'START_HDin1': [0.0, 1.01325, 70.01325],
	'START_HDin2': [0.0, 1.01325, 70.01325],
	'B_aux': [0, 1.01325, 105.01325],
	# boundary nodes
	'END': [0.0, 31.01325, 71.01325],
	'START_ND': [0.0, 1.01325, 70.01325],
	'START_HD': [0.0, 1.01325, 61.01325]
})

# all nodes
nodes = entries + exits + innodes

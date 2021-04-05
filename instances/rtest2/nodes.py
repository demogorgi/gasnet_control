""" auto-generated python script by create_netstate """
  
import gurobipy as gp
from gurobipy import GRB

# set up Q boundaries
nodes_with_bds, q_lb, q_ub = gp.multidict({
	'END': [-1100, -900],
	'START': [0, 10000]
})

# set up entries
entries, entry_flow_bound, pressure = gp.multidict({
	'START_ND': [800, 81.01325],
	'START_HD': [1, 81.01325]
})

# set up exits
exits = ['END']

# set up innodes
innodes = ['N407', 'N366', 'N409', 'N408', 'N433', 'N376', 'N432', 'N431', 'N430', 'N436', 'N435', 'N434', 'N418', 'B', 'N428', 'N429', 'N410', 'N322', 'N415', 'N416', 'A', 'N417', 'START', 'START_aux0', 'START_aux1', 'START_NDin1', 'START_NDin2', 'START_aux2', 'START_aux3', 'START_HDin1', 'START_HDin2', 'B_aux']

# set up nodes heights and pressure limits
nodes, heights, pressure_limits_lower, pressure_limits_upper = gp.multidict({
	# innodes
	'N407': [0.0, 1.01325, 105.01325],
	'N366': [0.0, 1.01325, 105.01325],
	'N409': [0.0, 1.01325, 105.01325],
	'N408': [0.0, 1.01325, 105.01325],
	'N433': [0.0, 1.01325, 105.01325],
	'N376': [0.0, 1.01325, 105.01325],
	'N432': [0.0, 1.01325, 105.01325],
	'N431': [0.0, 1.01325, 105.01325],
	'N430': [0.0, 1.01325, 105.01325],
	'N436': [0.0, 1.01325, 105.01325],
	'N435': [0.0, 1.01325, 105.01325],
	'N434': [0.0, 1.01325, 105.01325],
	'N418': [0.0, 1.01325, 105.01325],
	'B': [0.0, 1.01325, 105.01325],
	'N428': [0.0, 1.01325, 105.01325],
	'N429': [0.0, 1.01325, 105.01325],
	'N410': [0.0, 1.01325, 105.01325],
	'N322': [0.0, 1.01325, 105.01325],
	'N415': [0.0, 1.01325, 105.01325],
	'N416': [0.0, 1.01325, 105.01325],
	'A': [0.0, 1.01325, 105.01325],
	'N417': [0.0, 1.01325, 105.01325],
	'START': [0.0, 1.01325, 105.01325],
	'START_aux0': [0.0, 1.01325, 81.01325],
	'START_aux1': [0.0, 1.01325, 81.01325],
	'START_NDin1': [0.0, 1.01325, 81.01325],
	'START_NDin2': [0.0, 1.01325, 81.01325],
	'START_aux2': [0.0, 1.01325, 81.01325],
	'START_aux3': [0.0, 1.01325, 81.01325],
	'START_HDin1': [0.0, 1.01325, 81.01325],
	'START_HDin2': [0.0, 1.01325, 81.01325],
	'B_aux': [0, 1.01325, 105.01325],
	# boundary nodes
	'END': [0.0, 1.01325, 105.01325],
	'START_ND': [0.0, 1.01325, 81.01325],
	'START_HD': [0.0, 1.01325, 81.01325]
})

# all nodes
nodes = entries + exits + innodes

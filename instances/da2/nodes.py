""" auto-generated python script by create_netstate """
  
import gurobipy as gp
from gurobipy import GRB

# set up Q boundaries
nodes_with_bds, q_lb, q_ub = gp.multidict({
	'XN': [-510, -490],
	'XH': [-610, -590],
	'EH': [0, 1600],
	'EN': [0, 1600]
})

# set up entries
entries, entry_flow_bound, pressure = gp.multidict({
	'EH_ND': [1600, 84.01325],
	'EH_HD': [100, 86.01325],
	'EN_ND': [1600, 71.01325],
	'EN_HD': [100, 72.01325]
})

# set up exits
exits = ['XN', 'XH']

# set up innodes
innodes = ['N23_1', 'N13', 'N14', 'N26', 'N17', 'N17_1', 'N11', 'N12', 'N20', 'N19', 'N18', 'N25', 'N23', 'EH', 'EH_aux0', 'EH_aux1', 'EH_NDin1', 'EH_NDin2', 'EH_aux2', 'EH_aux3', 'EH_HDin1', 'EH_HDin2', 'EN', 'EN_aux0', 'EN_aux1', 'EN_NDin1', 'EN_NDin2', 'EN_aux2', 'EN_aux3', 'EN_HDin1', 'EN_HDin2', 'N26_aux', 'N22_aux', 'N23_aux', 'N22']

# set up nodes heights and pressure limits
nodes, heights, pressure_limits_lower, pressure_limits_upper = gp.multidict({
	# innodes
	'N23_1': [0.0, 1.01325, 105.01325],
	'N13': [0.0, 1.01325, 105.01325],
	'N14': [0.0, 1.01325, 105.01325],
	'N26': [0.0, 1.01325, 105.01325],
	'N17': [0.0, 1.01325, 105.01325],
	'N17_1': [0.0, 1.01325, 105.01325],
	'N11': [0.0, 1.01325, 105.01325],
	'N12': [0.0, 1.01325, 105.01325],
	'N20': [0.0, 1.01325, 105.01325],
	'N19': [0.0, 1.01325, 105.01325],
	'N18': [0.0, 1.01325, 105.01325],
	'N25': [0.0, 1.01325, 105.01325],
	'N23': [0.0, 1.01325, 105.01325],
	'EH': [0.0, 1.01325, 105.01325],
	'EH_aux0': [0.0, 1.01325, 84.01325],
	'EH_aux1': [0.0, 1.01325, 84.01325],
	'EH_NDin1': [0.0, 1.01325, 84.01325],
	'EH_NDin2': [0.0, 1.01325, 84.01325],
	'EH_aux2': [0.0, 1.01325, 84.01325],
	'EH_aux3': [0.0, 1.01325, 84.01325],
	'EH_HDin1': [0.0, 1.01325, 84.01325],
	'EH_HDin2': [0.0, 1.01325, 84.01325],
	'EN': [0.0, 1.01325, 105.01325],
	'EN_aux0': [0.0, 1.01325, 71.01325],
	'EN_aux1': [0.0, 1.01325, 71.01325],
	'EN_NDin1': [0.0, 1.01325, 71.01325],
	'EN_NDin2': [0.0, 1.01325, 71.01325],
	'EN_aux2': [0.0, 1.01325, 71.01325],
	'EN_aux3': [0.0, 1.01325, 71.01325],
	'EN_HDin1': [0.0, 1.01325, 71.01325],
	'EN_HDin2': [0.0, 1.01325, 71.01325],
	'N26_aux': [0, 1.01325, 105.01325],
	'N22_aux': [0.0, 1.01325, 105.01325],
	'N23_aux': [0.0, 1.01325, 105.01325],
	'N22': [0.0, 1.01325, 105.01325],
	# boundary nodes
	'XN': [0.0, 56.01325, 73.01325],
	'XH': [0.0, 67.01325, 86.01325],
	'EH_ND': [0.0, 1.01325, 84.01325],
	'EH_HD': [0.0, 1.01325, 86.01325],
	'EN_ND': [0.0, 1.01325, 71.01325],
	'EN_HD': [0.0, 1.01325, 72.01325]
})

# all nodes
nodes = entries + exits + innodes

""" auto-generated python script by create_netstate """
  
import gurobipy as gp
from gurobipy import GRB

# set up Q boundaries
nodes_with_bds, q_lb, q_ub = gp.multidict({
	'START_1': [0, 10000],
	'START': [0, 10000],
	'END': [-1000, 0]
})

# set up entries
entries, entry_flow_bound, pressure = gp.multidict({
	'START_2_ND': [0, 51.01325],
	'START_2_HD': [100, 51.01325],
	'START_1_ND': [200, 70.01325],
	'START_1_HD': [1000, 61.01325],
	'START_ND': [200, 70.01325],
	'START_HD': [1000, 61.01325]
})

# set up exits
exits = ['END_2', 'END_1', 'END']

# set up innodes
innodes = ['N39', 'N38', 'N37', 'N36', 'N358', 'N357', 'N356', 'N355', 'N354', 'N353', 'N352', 'N351', 'N350', 'N35', 'N340', 'N34', 'N339', 'N338', 'N337', 'N336', 'N335', 'N334', 'N333', 'N332', 'N33', 'N322_1', 'N322', 'N32', 'N31', 'N21', 'N20', 'N19', 'N18', 'N17', 'N16', 'N15', 'N14', 'N13', 'B_2', 'B_1', 'B', 'A_2', 'A_1', 'A', 'START_2', 'START_2_aux0', 'START_2_aux1', 'START_2_NDin1', 'START_2_NDin2', 'START_2_aux2', 'START_2_aux3', 'START_2_HDin1', 'START_2_HDin2', 'START_1', 'START_1_aux0', 'START_1_aux1', 'START_1_NDin1', 'START_1_NDin2', 'START_1_aux2', 'START_1_aux3', 'START_1_HDin1', 'START_1_HDin2', 'START', 'START_aux0', 'START_aux1', 'START_NDin1', 'START_NDin2', 'START_aux2', 'START_aux3', 'START_HDin1', 'START_HDin2', 'B_2_aux', 'B_1_aux', 'B_aux']

# set up nodes heights and pressure limits
nodes, heights, pressure_limits_lower, pressure_limits_upper = gp.multidict({
	# innodes
	'N39': [0.0, 1.01325, 105.01325],
	'N38': [0.0, 1.01325, 105.01325],
	'N37': [0.0, 1.01325, 105.01325],
	'N36': [0.0, 1.01325, 105.01325],
	'N358': [0.0, 1.01325, 105.01325],
	'N357': [0.0, 1.01325, 105.01325],
	'N356': [0.0, 1.01325, 105.01325],
	'N355': [0.0, 1.01325, 105.01325],
	'N354': [0.0, 1.01325, 105.01325],
	'N353': [0.0, 1.01325, 105.01325],
	'N352': [0.0, 1.01325, 105.01325],
	'N351': [0.0, 1.01325, 105.01325],
	'N350': [0.0, 1.01325, 105.01325],
	'N35': [0.0, 1.01325, 105.01325],
	'N340': [0.0, 1.01325, 105.01325],
	'N34': [0.0, 1.01325, 105.01325],
	'N339': [0.0, 1.01325, 105.01325],
	'N338': [0.0, 1.01325, 105.01325],
	'N337': [0.0, 1.01325, 105.01325],
	'N336': [0.0, 1.01325, 105.01325],
	'N335': [0.0, 1.01325, 105.01325],
	'N334': [0.0, 1.01325, 105.01325],
	'N333': [0.0, 1.01325, 105.01325],
	'N332': [0.0, 1.01325, 105.01325],
	'N33': [0.0, 1.01325, 105.01325],
	'N322_1': [0.0, 1.01325, 105.01325],
	'N322': [0.0, 1.01325, 105.01325],
	'N32': [0.0, 1.01325, 105.01325],
	'N31': [0.0, 1.01325, 105.01325],
	'N21': [0.0, 1.01325, 105.01325],
	'N20': [0.0, 1.01325, 105.01325],
	'N19': [0.0, 1.01325, 105.01325],
	'N18': [0.0, 1.01325, 105.01325],
	'N17': [0.0, 1.01325, 105.01325],
	'N16': [0.0, 1.01325, 105.01325],
	'N15': [0.0, 1.01325, 105.01325],
	'N14': [0.0, 1.01325, 105.01325],
	'N13': [0.0, 1.01325, 105.01325],
	'B_2': [0.0, 1.01325, 105.01325],
	'B_1': [0.0, 1.01325, 105.01325],
	'B': [0.0, 1.01325, 105.01325],
	'A_2': [0.0, 1.01325, 105.01325],
	'A_1': [0.0, 1.01325, 105.01325],
	'A': [0.0, 1.01325, 105.01325],
	'START_2': [0.0, 1.01325, 105.01325],
	'START_2_aux0': [0.0, 1.01325, 51.01325],
	'START_2_aux1': [0.0, 1.01325, 51.01325],
	'START_2_NDin1': [0.0, 1.01325, 51.01325],
	'START_2_NDin2': [0.0, 1.01325, 51.01325],
	'START_2_aux2': [0.0, 1.01325, 51.01325],
	'START_2_aux3': [0.0, 1.01325, 51.01325],
	'START_2_HDin1': [0.0, 1.01325, 51.01325],
	'START_2_HDin2': [0.0, 1.01325, 51.01325],
	'START_1': [0.0, 1.01325, 105.01325],
	'START_1_aux0': [0.0, 1.01325, 70.01325],
	'START_1_aux1': [0.0, 1.01325, 70.01325],
	'START_1_NDin1': [0.0, 1.01325, 70.01325],
	'START_1_NDin2': [0.0, 1.01325, 70.01325],
	'START_1_aux2': [0.0, 1.01325, 70.01325],
	'START_1_aux3': [0.0, 1.01325, 70.01325],
	'START_1_HDin1': [0.0, 1.01325, 70.01325],
	'START_1_HDin2': [0.0, 1.01325, 70.01325],
	'START': [0.0, 1.01325, 105.01325],
	'START_aux0': [0.0, 1.01325, 70.01325],
	'START_aux1': [0.0, 1.01325, 70.01325],
	'START_NDin1': [0.0, 1.01325, 70.01325],
	'START_NDin2': [0.0, 1.01325, 70.01325],
	'START_aux2': [0.0, 1.01325, 70.01325],
	'START_aux3': [0.0, 1.01325, 70.01325],
	'START_HDin1': [0.0, 1.01325, 70.01325],
	'START_HDin2': [0.0, 1.01325, 70.01325],
	'B_2_aux': [0, 1.01325, 105.01325],
	'B_1_aux': [0, 1.01325, 105.01325],
	'B_aux': [0, 1.01325, 105.01325],
	# boundary nodes
	'END_2': [0.0, 1.01325, 105.01325],
	'END_1': [0.0, 1.01325, 105.01325],
	'END': [0.0, 1.01325, 105.01325],
	'START_2_ND': [0.0, 1.01325, 51.01325],
	'START_2_HD': [0.0, 1.01325, 51.01325],
	'START_1_ND': [0.0, 1.01325, 70.01325],
	'START_1_HD': [0.0, 1.01325, 61.01325],
	'START_ND': [0.0, 1.01325, 70.01325],
	'START_HD': [0.0, 1.01325, 61.01325]
})

# all nodes
nodes = entries + exits + innodes

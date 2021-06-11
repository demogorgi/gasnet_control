""" auto-generated python script by create_netstate """
  
import gurobipy as gp
from gurobipy import GRB

# set up Q boundaries
nodes_with_bds, q_lb, q_ub = gp.multidict({
	# entries
	'S2P_ND': [0, 0],
	'S2P_HD': [0, 0],
	'S1P_ND': [0, 0],
	'S1P_HD': [0, 0],
	'E1_ND':  [0, 1200],
	'E1_HD':  [0, 100],
	'E2_ND':  [0, 0],
	'E2_HD':  [0, 0],
	# exits
	'X1': [-110, -90],
	'X2': [-110, -90],
	'S1I': [-510, -490],
	'S2I': [-510, -490]
	
})

# set up entries
entries, entry_flow_bound, pressure = gp.multidict({
	'S2P_ND': [0, 40],
	'S2P_HD': [0, 40],
	'S1P_ND': [0, 40],
	'S1P_HD': [0, 40],
	'E1_ND': [1100, 60],
	'E1_HD': [200, 80],
	'E2_ND': [1100, 60],
	'E2_HD': [200, 80]
})

# set up exits
exits = ['X1', 'X2', 'S1I', 'S2I']

# set up innodes
innodes = ['A9', 'A14', 'A12', 'A7', 'A8', 'A6', 'A4', 'A5', 'A13', 'A2', 'A11', 'A1', 'A3', 'A10', 'A15', 'S2P', 'S2P_aux0', 'S2P_aux1', 'S2P_NDin1', 'S2P_NDin2', 'S2P_aux2', 'S2P_aux3', 'S2P_HDin1', 'S2P_HDin2', 'S1P', 'S1P_aux0', 'S1P_aux1', 'S1P_NDin1', 'S1P_NDin2', 'S1P_aux2', 'S1P_aux3', 'S1P_HDin1', 'S1P_HDin2', 'E1', 'E1_aux0', 'E1_aux1', 'E1_NDin1', 'E1_NDin2', 'E1_aux2', 'E1_aux3', 'E1_HDin1', 'E1_HDin2', 'E2', 'E2_aux0', 'E2_aux1', 'E2_NDin1', 'E2_NDin2', 'E2_aux2', 'E2_aux3', 'E2_HDin1', 'E2_HDin2', 'A7_aux', 'A10_aux', 'A15_aux', 'A14_aux', 'A2_aux', 'A3_aux']

# set up nodes heights and pressure limits
nodes, heights, pressure_limits_lower, pressure_limits_upper = gp.multidict({
	# innodes
	'A9': [0.0, 1.01325, 105.01325],
	'A14': [0.0, 1.01325, 105.01325],
	'A12': [0.0, 1.01325, 105.01325],
	'S1I': [0.0, 1.01325, 105.01325],
	'A7': [0.0, 1.01325, 105.01325],
	'A8': [0.0, 1.01325, 105.01325],
	'X2': [0.0, 1.01325, 105.01325],
	'A6': [0.0, 1.01325, 105.01325],
	'A4': [0.0, 1.01325, 105.01325],
	'S2I': [0.0, 1.01325, 105.01325],
	'A5': [0.0, 1.01325, 105.01325],
	'A13': [0.0, 1.01325, 105.01325],
	'A2': [0.0, 1.01325, 105.01325],
	'A11': [0.0, 1.01325, 105.01325],
	'A1': [0.0, 1.01325, 105.01325],
	'A3': [0.0, 1.01325, 105.01325],
	'A10': [0.0, 1.01325, 105.01325],
	'A15': [0.0, 1.01325, 105.01325],
	'S2P': [0.0, 1.01325, 105.01325],
	'S2P_aux0': [0.0, 1.01325, 105.01325],
	'S2P_aux1': [0.0, 1.01325, 105.01325],
	'S2P_NDin1': [0.0, 1.01325, 105.01325],
	'S2P_NDin2': [0.0, 1.01325, 105.01325],
	'S2P_aux2': [0.0, 1.01325, 105.01325],
	'S2P_aux3': [0.0, 1.01325, 105.01325],
	'S2P_HDin1': [0.0, 1.01325, 105.01325],
	'S2P_HDin2': [0.0, 1.01325, 105.01325],
	'S1P': [0.0, 1.01325, 105.01325],
	'S1P_aux0': [0.0, 1.01325, 105.01325],
	'S1P_aux1': [0.0, 1.01325, 105.01325],
	'S1P_NDin1': [0.0, 1.01325, 105.01325],
	'S1P_NDin2': [0.0, 1.01325, 105.01325],
	'S1P_aux2': [0.0, 1.01325, 105.01325],
	'S1P_aux3': [0.0, 1.01325, 105.01325],
	'S1P_HDin1': [0.0, 1.01325, 105.01325],
	'S1P_HDin2': [0.0, 1.01325, 105.01325],
	'E1': [0.0, 1.01325, 105.01325],
	'E1_aux0': [0.0, 1.01325, 105.01325],
	'E1_aux1': [0.0, 1.01325, 105.01325],
	'E1_NDin1': [0.0, 1.01325, 105.01325],
	'E1_NDin2': [0.0, 1.01325, 105.01325],
	'E1_aux2': [0.0, 1.01325, 105.01325],
	'E1_aux3': [0.0, 1.01325, 105.01325],
	'E1_HDin1': [0.0, 1.01325, 105.01325],
	'E1_HDin2': [0.0, 1.01325, 105.01325],
	'E2': [0.0, 1.01325, 105.01325],
	'E2_aux0': [0.0, 1.01325, 105.01325],
	'E2_aux1': [0.0, 1.01325, 105.01325],
	'E2_NDin1': [0.0, 1.01325, 105.01325],
	'E2_NDin2': [0.0, 1.01325, 105.01325],
	'E2_aux2': [0.0, 1.01325, 105.01325],
	'E2_aux3': [0.0, 1.01325, 105.01325],
	'E2_HDin1': [0.0, 1.01325, 105.01325],
	'E2_HDin2': [0.0, 1.01325, 105.01325],
	'A7_aux': [0, 1.01325, 105.01325],
	'A10_aux': [0, 1.01325, 105.01325],
	'A15_aux': [0.0, 1.01325, 105.01325],
	'A14_aux': [0.0, 1.01325, 105.01325],
	'A2_aux': [0.0, 1.01325, 105.01325],
	'A3_aux': [0.0, 1.01325, 105.01325],
	# boundary nodes
	'X1': [0.0, 1.01325, 105.01325],
	'S2P_ND': [0.0, 1.01325, 106],
	'S2P_HD': [0.0, 1.01325, 106],
	'S1P_ND': [0.0, 1.01325, 106],
	'S1P_HD': [0.0, 1.01325, 106],
	'E1_ND': [0.0, 1.01325, 106],
	'E1_HD': [0.0, 1.01325, 106],
	'E2_ND': [0.0, 1.01325, 106],
	'E2_HD': [0.0, 1.01325, 106]
})

# all nodes
nodes = entries + exits + innodes

""" auto-generated python script by create_netstate """

import gurobipy as gp
from gurobipy import GRB

valves = gp.tuplelist([
  
])

flap_traps = gp.tuplelist([
	('S2P_NDin2', 'S2P_NDin1'),
	('S2P_HDin1', 'S2P_aux3'),
	('S1P_NDin2', 'S1P_NDin1'),
	('S1P_HDin1', 'S1P_aux3'),
	('E1_NDin2', 'E1_NDin1'),
	('E1_HDin1', 'E1_aux3'),
	('E2_NDin2', 'E2_NDin1'),
	('E2_HDin1', 'E2_aux3'),
	('A7_aux', 'A7'),
	('A10_aux', 'A10'),
	('A15_aux', 'A14_aux'),
	('A2_aux', 'A3_aux')
])

resistors, diameter = gp.multidict({
	('A6', 'A7_aux'): 0.4,
	('A11', 'A10_aux'): 0.35000000000000003
})

# compressors tuples
compressors = gp.tuplelist([
	('A15', 'A14'),
	('A2', 'A3')
])

pipes, length, roughness = gp.multidict({
	('A8', 'A9'): [20000, 0.000006],
	('A9', 'X1'): [10000, 0.00001],
	('A14', 'A12'): [30000, 0.000006],
	('A5', 'S1I'): [1000, 0.000006],
	('A7', 'A8'): [25000, 0.000006],
	('A8', 'X2'): [5000, 0.000006],
	('A10', 'A9'): [35000, 0.000006],
	('A4', 'A6'): [15000, 0.000006],
	('A13', 'S2P'): [30000, 0.000006],
	('A3', 'A4'): [30000, 0.000006],
	('A13', 'S2I'): [30000, 0.000006],
	('A5', 'S1P'): [1000, 0.000006],
	('E1', 'A1'): [5000, 0.000006],
	('A4', 'A5'): [15000, 0.000006],
	('A12', 'A13'): [8000, 0.000006],
	('A1', 'A2'): [30000, 0.000006],
	('A12', 'A11'): [30000, 0.000006],
	('A11', 'E2'): [30000, 0.000006],
	('A1', 'A15'): [30000, 0.000006],
	('S2P_aux0', 'S2P'): [100, 0.000012],
	('S2P_aux1', 'S2P_aux0'): [10000, 0.000012],
	('S2P_NDin1', 'S2P_aux1'): [10000, 0.000012],
	('S2P_ND', 'S2P_NDin2'): [10000, 0.000012],
	('S2P_aux2', 'S2P_aux1'): [10000, 0.000012],
	('S2P_aux3', 'S2P_aux2'): [1000, 0.000012],
	('S2P_HDin2', 'S2P_HDin1'): [1000, 0.000012],
	('S2P_HD', 'S2P_HDin2'): [10000, 0.000012],
	('S1P_aux0', 'S1P'): [100, 0.000012],
	('S1P_aux1', 'S1P_aux0'): [10000, 0.000012],
	('S1P_NDin1', 'S1P_aux1'): [10000, 0.000012],
	('S1P_ND', 'S1P_NDin2'): [10000, 0.000012],
	('S1P_aux2', 'S1P_aux1'): [10000, 0.000012],
	('S1P_aux3', 'S1P_aux2'): [1000, 0.000012],
	('S1P_HDin2', 'S1P_HDin1'): [1000, 0.000012],
	('S1P_HD', 'S1P_HDin2'): [10000, 0.000012],
	('E1_aux0', 'E1'): [100, 0.000012],
	('E1_aux1', 'E1_aux0'): [10000, 0.000012],
	('E1_NDin1', 'E1_aux1'): [10000, 0.000012],
	('E1_ND', 'E1_NDin2'): [10000, 0.000012],
	('E1_aux2', 'E1_aux1'): [10000, 0.000012],
	('E1_aux3', 'E1_aux2'): [1000, 0.000012],
	('E1_HDin2', 'E1_HDin1'): [1000, 0.000012],
	('E1_HD', 'E1_HDin2'): [10000, 0.000012],
	('E2_aux0', 'E2'): [100, 0.000012],
	('E2_aux1', 'E2_aux0'): [10000, 0.000012],
	('E2_NDin1', 'E2_aux1'): [10000, 0.000012],
	('E2_ND', 'E2_NDin2'): [10000, 0.000012],
	('E2_aux2', 'E2_aux1'): [10000, 0.000012],
	('E2_aux3', 'E2_aux2'): [1000, 0.000012],
	('E2_HDin2', 'E2_HDin1'): [1000, 0.000012],
	('E2_HD', 'E2_HDin2'): [10000, 0.000012],
	('A15', 'A15_aux'): [1, 0.000012],
	('A14_aux', 'A14'): [1, 0.000012],
	('A2', 'A2_aux'): [1, 0.000012],
	('A3_aux', 'A3'): [1, 0.000012]
})

# this cannot be put into the multidicts for pipes and resistors, as no keys could be appended then
diameter = {
	#pipes
	('A8', 'A9'): 0.35000000000000003,
	('A9', 'X1'): 0.4,
	('A14', 'A12'): 0.6,
	('A5', 'S1I'): 0.9,
	('A7', 'A8'): 0.4,
	('A8', 'X2'): 0.4,
	('A10', 'A9'): 0.35000000000000003,
	('A4', 'A6'): 0.9,
	('A13', 'S2P'): 0.9,
	('A3', 'A4'): 0.9,
	('A13', 'S2I'): 0.9,
	('A5', 'S1P'): 0.9,
	('E1', 'A1'): 0.9,
	('A4', 'A5'): 0.9,
	('A12', 'A13'): 0.4,
	('A1', 'A2'): 0.9,
	('A12', 'A11'): 0.4,
	('A11', 'E2'): 0.4,
	('A1', 'A15'): 0.9,
	('S2P_aux0', 'S2P'): 0.5,
	('S2P_aux1', 'S2P_aux0'): 2,
	('S2P_NDin1', 'S2P_aux1'): 2,
	('S2P_ND', 'S2P_NDin2'): 2,
	('S2P_aux2', 'S2P_aux1'): 2,
	('S2P_aux3', 'S2P_aux2'): 0.3,
	('S2P_HDin2', 'S2P_HDin1'): 0.3,
	('S2P_HD', 'S2P_HDin2'): 2,
	('S1P_aux0', 'S1P'): 0.5,
	('S1P_aux1', 'S1P_aux0'): 2,
	('S1P_NDin1', 'S1P_aux1'): 2,
	('S1P_ND', 'S1P_NDin2'): 2,
	('S1P_aux2', 'S1P_aux1'): 2,
	('S1P_aux3', 'S1P_aux2'): 0.3,
	('S1P_HDin2', 'S1P_HDin1'): 0.3,
	('S1P_HD', 'S1P_HDin2'): 2,
	('E1_aux0', 'E1'): 0.5,
	('E1_aux1', 'E1_aux0'): 2,
	('E1_NDin1', 'E1_aux1'): 2,
	('E1_ND', 'E1_NDin2'): 2,
	('E1_aux2', 'E1_aux1'): 2,
	('E1_aux3', 'E1_aux2'): 0.3,
	('E1_HDin2', 'E1_HDin1'): 0.3,
	('E1_HD', 'E1_HDin2'): 2,
	('E2_aux0', 'E2'): 0.5,
	('E2_aux1', 'E2_aux0'): 2,
	('E2_NDin1', 'E2_aux1'): 2,
	('E2_ND', 'E2_NDin2'): 2,
	('E2_aux2', 'E2_aux1'): 2,
	('E2_aux3', 'E2_aux2'): 0.3,
	('E2_HDin2', 'E2_HDin1'): 0.3,
	('E2_HD', 'E2_HDin2'): 2,
	('A15', 'A15_aux'): 1,
	('A14_aux', 'A14'): 1,
	('A2', 'A2_aux'): 1,
	('A3_aux', 'A3'): 1,
	# resistors
	('A6', 'A7_aux'): 0.4,
	('A11', 'A10_aux'): 0.35000000000000003
}

# special pipes
special = gp.tuplelist([
	('S2P_aux0', 'S2P'),
	('S1P_aux0', 'S1P'),
	('E1_aux0', 'E1'),
	('E2_aux0', 'E2')
])

connections = pipes + resistors + valves + flap_traps + compressors

non_pipes = [x for x in connections if x not in pipes]


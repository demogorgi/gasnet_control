""" auto-generated python script by create_netstate """

import gurobipy as gp
from gurobipy import GRB

valves = gp.tuplelist([
  
])

flap_traps = gp.tuplelist([
	('START_2_NDin2', 'START_2_NDin1'),
	('START_2_HDin1', 'START_2_aux3'),
	('START_1_NDin2', 'START_1_NDin1'),
	('START_1_HDin1', 'START_1_aux3'),
	('START_NDin2', 'START_NDin1'),
	('START_HDin1', 'START_aux3'),
	('B_2_aux', 'B_2'),
	('B_1_aux', 'B_1'),
	('B_aux', 'B')
])

resistors = gp.tuplelist({
	('A_2', 'B_2_aux'),
	('A_1', 'B_1_aux'),
	('A', 'B_aux')
})

# compressors tuples
compressors = gp.tuplelist([
	
])

pipes, length, roughness = gp.multidict({
	('START_2', 'N322_1'): [500, 0.000012],
	('START_1', 'N31'): [100, 0.000012],
	('START', 'N322'): [500, 0.000012],
	('N39', 'A_1'): [100, 0.000012],
	('N38', 'N39'): [100, 0.000012],
	('N37', 'N38'): [100, 0.000012],
	('N36', 'N37'): [100, 0.000012],
	('N35', 'N36'): [100, 0.000012],
	('N358', 'END_2'): [10000, 0.000012],
	('N357', 'N358'): [10000, 0.000012],
	('N356', 'N357'): [10000, 0.000012],
	('N355', 'N356'): [10000, 0.000012],
	('N354', 'N355'): [10000, 0.000012],
	('N353', 'N354'): [10000, 0.000012],
	('N352', 'N353'): [10000, 0.000012],
	('N351', 'N352'): [10000, 0.000012],
	('N350', 'N351'): [10000, 0.000012],
	('N34', 'N35'): [100, 0.000012],
	('N340', 'A_2'): [10000, 0.000012],
	('N33', 'N34'): [100, 0.000012],
	('N339', 'N340'): [10000, 0.000012],
	('N338', 'N339'): [10000, 0.000012],
	('N337', 'N338'): [10000, 0.000012],
	('N336', 'N337'): [10000, 0.000012],
	('N335', 'N336'): [10000, 0.000012],
	('N334', 'N335'): [10000, 0.000012],
	('N333', 'N334'): [10000, 0.000012],
	('N332', 'N333'): [10000, 0.000012],
	('N32', 'N33'): [100, 0.000012],
	('N322_1', 'N332'): [10000, 0.000012],
	('N322', 'A'): [500, 0.000012],
	('N31', 'N32'): [100, 0.000012],
	('N21', 'END_1'): [100, 0.000012],
	('N20', 'N21'): [100, 0.000012],
	('N19', 'N20'): [100, 0.000012],
	('N18', 'N19'): [100, 0.000012],
	('N17', 'N18'): [100, 0.000012],
	('N16', 'N17'): [100, 0.000012],
	('N15', 'N16'): [100, 0.000012],
	('N14', 'N15'): [100, 0.000012],
	('N13', 'N14'): [100, 0.000012],
	('B_2', 'N350'): [10000, 0.000012],
	('B_1', 'N13'): [100, 0.000012],
	('B', 'END'): [1000, 0.000012],
	('START_2_aux0', 'START_2'): [100, 0.000012],
	('START_2_aux1', 'START_2_aux0'): [10000, 0.000012],
	('START_2_NDin1', 'START_2_aux1'): [10000, 0.000012],
	('START_2_ND', 'START_2_NDin2'): [10000, 0.000012],
	('START_2_aux2', 'START_2_aux1'): [10000, 0.000012],
	('START_2_aux3', 'START_2_aux2'): [1000, 0.000012],
	('START_2_HDin2', 'START_2_HDin1'): [1000, 0.000012],
	('START_2_HD', 'START_2_HDin2'): [10000, 0.000012],
	('START_1_aux0', 'START_1'): [100, 0.000012],
	('START_1_aux1', 'START_1_aux0'): [10000, 0.000012],
	('START_1_NDin1', 'START_1_aux1'): [10000, 0.000012],
	('START_1_ND', 'START_1_NDin2'): [10000, 0.000012],
	('START_1_aux2', 'START_1_aux1'): [10000, 0.000012],
	('START_1_aux3', 'START_1_aux2'): [1000, 0.000012],
	('START_1_HDin2', 'START_1_HDin1'): [1000, 0.000012],
	('START_1_HD', 'START_1_HDin2'): [10000, 0.000012],
	('START_aux0', 'START'): [100, 0.000012],
	('START_aux1', 'START_aux0'): [10000, 0.000012],
	('START_NDin1', 'START_aux1'): [10000, 0.000012],
	('START_ND', 'START_NDin2'): [10000, 0.000012],
	('START_aux2', 'START_aux1'): [10000, 0.000012],
	('START_aux3', 'START_aux2'): [1000, 0.000012],
	('START_HDin2', 'START_HDin1'): [1000, 0.000012],
	('START_HD', 'START_HDin2'): [10000, 0.000012]
})

# this cannot be put into the multidicts for pipes and resistors, as no keys could be appended then
diameter = {
	#pipes
	('START_2', 'N322_1'): 1,
	('START_1', 'N31'): 1,
	('START', 'N322'): 1,
	('N39', 'A_1'): 1,
	('N38', 'N39'): 1,
	('N37', 'N38'): 1,
	('N36', 'N37'): 1,
	('N35', 'N36'): 1,
	('N358', 'END_2'): 0.5,
	('N357', 'N358'): 0.5,
	('N356', 'N357'): 0.5,
	('N355', 'N356'): 0.5,
	('N354', 'N355'): 0.5,
	('N353', 'N354'): 0.5,
	('N352', 'N353'): 0.5,
	('N351', 'N352'): 0.5,
	('N350', 'N351'): 0.5,
	('N34', 'N35'): 1,
	('N340', 'A_2'): 0.5,
	('N33', 'N34'): 1,
	('N339', 'N340'): 0.5,
	('N338', 'N339'): 0.5,
	('N337', 'N338'): 0.5,
	('N336', 'N337'): 0.5,
	('N335', 'N336'): 0.5,
	('N334', 'N335'): 0.5,
	('N333', 'N334'): 0.5,
	('N332', 'N333'): 0.5,
	('N32', 'N33'): 1,
	('N322_1', 'N332'): 0.5,
	('N322', 'A'): 1,
	('N31', 'N32'): 1,
	('N21', 'END_1'): 1,
	('N20', 'N21'): 1,
	('N19', 'N20'): 1,
	('N18', 'N19'): 1,
	('N17', 'N18'): 1,
	('N16', 'N17'): 1,
	('N15', 'N16'): 1,
	('N14', 'N15'): 1,
	('N13', 'N14'): 1,
	('B_2', 'N350'): 0.5,
	('B_1', 'N13'): 1,
	('B', 'END'): 1,
	('START_2_aux0', 'START_2'): 0.5,
	('START_2_aux1', 'START_2_aux0'): 2,
	('START_2_NDin1', 'START_2_aux1'): 2,
	('START_2_ND', 'START_2_NDin2'): 2,
	('START_2_aux2', 'START_2_aux1'): 2,
	('START_2_aux3', 'START_2_aux2'): 0.3,
	('START_2_HDin2', 'START_2_HDin1'): 0.3,
	('START_2_HD', 'START_2_HDin2'): 2,
	('START_1_aux0', 'START_1'): 0.5,
	('START_1_aux1', 'START_1_aux0'): 2,
	('START_1_NDin1', 'START_1_aux1'): 2,
	('START_1_ND', 'START_1_NDin2'): 2,
	('START_1_aux2', 'START_1_aux1'): 2,
	('START_1_aux3', 'START_1_aux2'): 0.3,
	('START_1_HDin2', 'START_1_HDin1'): 0.3,
	('START_1_HD', 'START_1_HDin2'): 2,
	('START_aux0', 'START'): 0.5,
	('START_aux1', 'START_aux0'): 2,
	('START_NDin1', 'START_aux1'): 2,
	('START_ND', 'START_NDin2'): 2,
	('START_aux2', 'START_aux1'): 2,
	('START_aux3', 'START_aux2'): 0.3,
	('START_HDin2', 'START_HDin1'): 0.3,
	('START_HD', 'START_HDin2'): 2
}

# special pipes
special = gp.tuplelist([
	('START_2_aux0', 'START_2'),
	('START_1_aux0', 'START_1'),
	('START_aux0', 'START')
])

connections = pipes + resistors + valves + flap_traps + compressors

non_pipes = [x for x in connections if x not in pipes and x not in resistors]


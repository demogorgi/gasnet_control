""" auto-generated python script by create_netstate """

import gurobipy as gp
from gurobipy import GRB

valves = gp.tuplelist([
  
])

check_valves = gp.tuplelist([
	('START_NDin2', 'START_NDin1'),
	('START_HDin1', 'START_aux3'),
	('START_1_NDin2', 'START_1_NDin1'),
	('START_1_HDin1', 'START_1_aux3'),
	('B_aux', 'B')
])

resistors, diameter = gp.multidict({
	('A', 'B_aux'): 0.9
})

# compressors tuples
compressors = gp.tuplelist([
	
])

pipes, length, roughness = gp.multidict({
	('N363_1', 'N364_1'): [100, 0.000012],
	('N365_1', 'N366_1'): [100, 0.000012],
	('N364_1', 'N365_1'): [100, 0.000012],
	('START_1', 'N322_1'): [500, 0.000012],
	('N376_1', 'N377_1'): [100, 0.000012],
	('B_1', 'N376_1'): [100, 0.000012],
	('N377_1', 'N378_1'): [100, 0.000012],
	('N378_1', 'N379_1'): [100, 0.000012],
	('N379_1', 'N380_1'): [100, 0.000012],
	('N380_1', 'N381_1'): [100, 0.000012],
	('N322_1', 'N363_1'): [100, 0.000012],
	('N381_1', 'N382_1'): [100, 0.000012],
	('N382_1', 'N383_1'): [100, 0.000012],
	('N383_1', 'N384_1'): [100, 0.000012],
	('N384_1', 'END_1'): [100, 0.000012],
	('N366_1', 'A_1'): [100, 0.000012],
	('N384', 'END'): [100, 0.000012],
	('N383', 'N384'): [100, 0.000012],
	('N382', 'N383'): [100, 0.000012],
	('N381', 'N382'): [100, 0.000012],
	('N380', 'N381'): [100, 0.000012],
	('N379', 'N380'): [100, 0.000012],
	('N378', 'N379'): [100, 0.000012],
	('N377', 'N378'): [100, 0.000012],
	('N376', 'N377'): [100, 0.000012],
	('N366', 'A'): [100, 0.000012],
	('N365', 'N366'): [100, 0.000012],
	('N364', 'N365'): [100, 0.000012],
	('N363', 'N364'): [100, 0.000012],
	('N322', 'N363'): [100, 0.000012],
	('B', 'N376'): [100, 0.000012],
	('START', 'N322'): [500, 0.000012],
	('A_1', 'B_1'): [1000, 0.000012],
	('START_aux0', 'START'): [100, 0.000012],
	('START_aux1', 'START_aux0'): [10000, 0.000012],
	('START_NDin1', 'START_aux1'): [10000, 0.000012],
	('START_ND', 'START_NDin2'): [10000, 0.000012],
	('START_aux2', 'START_aux1'): [10000, 0.000012],
	('START_aux3', 'START_aux2'): [1000, 0.000012],
	('START_HDin2', 'START_HDin1'): [1000, 0.000012],
	('START_HD', 'START_HDin2'): [10000, 0.000012],
	('START_1_aux0', 'START_1'): [100, 0.000012],
	('START_1_aux1', 'START_1_aux0'): [10000, 0.000012],
	('START_1_NDin1', 'START_1_aux1'): [10000, 0.000012],
	('START_1_ND', 'START_1_NDin2'): [10000, 0.000012],
	('START_1_aux2', 'START_1_aux1'): [10000, 0.000012],
	('START_1_aux3', 'START_1_aux2'): [1000, 0.000012],
	('START_1_HDin2', 'START_1_HDin1'): [1000, 0.000012],
	('START_1_HD', 'START_1_HDin2'): [10000, 0.000012]
})

# this cannot be put into the multidicts for pipes and resistors, as no keys could be appended then
diameter = {
	#pipes
	('N363_1', 'N364_1'): 1,
	('N365_1', 'N366_1'): 1,
	('N364_1', 'N365_1'): 1,
	('START_1', 'N322_1'): 1,
	('N376_1', 'N377_1'): 1,
	('B_1', 'N376_1'): 1,
	('N377_1', 'N378_1'): 1,
	('N378_1', 'N379_1'): 1,
	('N379_1', 'N380_1'): 1,
	('N380_1', 'N381_1'): 1,
	('N322_1', 'N363_1'): 1,
	('N381_1', 'N382_1'): 1,
	('N382_1', 'N383_1'): 1,
	('N383_1', 'N384_1'): 1,
	('N384_1', 'END_1'): 1,
	('N366_1', 'A_1'): 1,
	('N384', 'END'): 1,
	('N383', 'N384'): 1,
	('N382', 'N383'): 1,
	('N381', 'N382'): 1,
	('N380', 'N381'): 1,
	('N379', 'N380'): 1,
	('N378', 'N379'): 1,
	('N377', 'N378'): 1,
	('N376', 'N377'): 1,
	('N366', 'A'): 1,
	('N365', 'N366'): 1,
	('N364', 'N365'): 1,
	('N363', 'N364'): 1,
	('N322', 'N363'): 1,
	('B', 'N376'): 1,
	('START', 'N322'): 1,
	('A_1', 'B_1'): 0.271,
	('START_aux0', 'START'): 0.5,
	('START_aux1', 'START_aux0'): 2,
	('START_NDin1', 'START_aux1'): 2,
	('START_ND', 'START_NDin2'): 2,
	('START_aux2', 'START_aux1'): 2,
	('START_aux3', 'START_aux2'): 0.3,
	('START_HDin2', 'START_HDin1'): 0.3,
	('START_HD', 'START_HDin2'): 2,
	('START_1_aux0', 'START_1'): 0.5,
	('START_1_aux1', 'START_1_aux0'): 2,
	('START_1_NDin1', 'START_1_aux1'): 2,
	('START_1_ND', 'START_1_NDin2'): 2,
	('START_1_aux2', 'START_1_aux1'): 2,
	('START_1_aux3', 'START_1_aux2'): 0.3,
	('START_1_HDin2', 'START_1_HDin1'): 0.3,
	('START_1_HD', 'START_1_HDin2'): 2,
	# resistors
	('A', 'B_aux'): 0.9
}

# special pipes
special = gp.tuplelist([
	('START_aux0', 'START'),
	('START_1_aux0', 'START_1')
])

connections = pipes + resistors + valves + check_valves + compressors

non_pipes = [x for x in connections if x not in pipes]


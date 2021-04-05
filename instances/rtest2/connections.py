""" auto-generated python script by create_netstate """

import gurobipy as gp
from gurobipy import GRB

valves = gp.tuplelist([
  
])

flap_traps = gp.tuplelist([
	('START_NDin2', 'START_NDin1'),
	('START_HDin1', 'START_aux3'),
	('B_aux', 'B')
])

resistors, diameter = gp.multidict({
	('A', 'B_aux'): 0.9
})

# compressors tuples
compressors = gp.tuplelist([
	
])

pipes, length, roughness = gp.multidict({
	('N432', 'N433'): [20000, 0.000012],
	('N431', 'N432'): [20000, 0.000012],
	('N430', 'N431'): [20000, 0.000012],
	('N429', 'N430'): [20000, 0.000012],
	('N436', 'N376'): [20000, 0.000012],
	('N435', 'N436'): [20000, 0.000012],
	('N322', 'N407'): [20000, 0.000012],
	('N434', 'N435'): [20000, 0.000012],
	('N433', 'N434'): [20000, 0.000012],
	('N366', 'A'): [20000, 0.000012],
	('N428', 'N429'): [20000, 0.000012],
	('N415', 'N416'): [20000, 0.000012],
	('START', 'N322'): [500, 0.000012],
	('N416', 'N417'): [20000, 0.000012],
	('N417', 'N418'): [20000, 0.000012],
	('N418', 'END'): [20000, 0.000012],
	('N366', 'N428'): [20000, 0.000012],
	('B', 'N376'): [20000, 0.000012],
	('N407', 'N408'): [20000, 0.000012],
	('N408', 'N409'): [20000, 0.000012],
	('N409', 'N410'): [20000, 0.000012],
	('N376', 'N415'): [20000, 0.000012],
	('N410', 'N366'): [20000, 0.000012],
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
	('N432', 'N433'): 1,
	('N431', 'N432'): 1,
	('N430', 'N431'): 1,
	('N429', 'N430'): 1,
	('N436', 'N376'): 1,
	('N435', 'N436'): 1,
	('N322', 'N407'): 1,
	('N434', 'N435'): 1,
	('N433', 'N434'): 1,
	('N366', 'A'): 1,
	('N428', 'N429'): 1,
	('N415', 'N416'): 1,
	('START', 'N322'): 1,
	('N416', 'N417'): 1,
	('N417', 'N418'): 1,
	('N418', 'END'): 1,
	('N366', 'N428'): 1,
	('B', 'N376'): 1,
	('N407', 'N408'): 1,
	('N408', 'N409'): 1,
	('N409', 'N410'): 1,
	('N376', 'N415'): 1,
	('N410', 'N366'): 1,
	('START_aux0', 'START'): 0.5,
	('START_aux1', 'START_aux0'): 2,
	('START_NDin1', 'START_aux1'): 2,
	('START_ND', 'START_NDin2'): 2,
	('START_aux2', 'START_aux1'): 2,
	('START_aux3', 'START_aux2'): 0.3,
	('START_HDin2', 'START_HDin1'): 0.3,
	('START_HD', 'START_HDin2'): 2,
	# resistors
	('A', 'B_aux'): 0.9
}

# special pipes
special = gp.tuplelist([
	('START_aux0', 'START')
])

connections = pipes + resistors + valves + flap_traps + compressors

non_pipes = [x for x in connections if x not in pipes]


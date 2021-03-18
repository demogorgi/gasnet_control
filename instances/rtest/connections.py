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
	('B', 'END'): [1000, 0.000012],
	('START', 'A'): [1000, 0.000012],
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
	('B', 'END'): 1,
	('START', 'A'): 1,
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


""" auto-generated python script by create_netstate """

import gurobipy as gp
from gurobipy import GRB

valves = gp.tuplelist([
  ('N17', 'N17_1'),
	('N18', 'N23_1')
])

flap_traps = gp.tuplelist([
	('EH_NDin2', 'EH_NDin1'),
	('EH_HDin1', 'EH_aux3'),
	('EN_NDin2', 'EN_NDin1'),
	('EN_HDin1', 'EN_aux3'),
	('N26_aux', 'N26'),
	('N22_aux', 'N23_aux')
])

resistors, diameter = gp.multidict({
	('N25', 'N26_aux'): 0.9
})

# compressors tuples
compressors = gp.tuplelist([
	('N22', 'N23')
])

pipes, length, roughness = gp.multidict({
	('N12', 'N13'): [20000, 0.000012],
	('N13', 'N14'): [20000, 0.000012],
	('N26', 'N13'): [20000, 0.000012],
	('N14', 'XN'): [20000, 0.000012],
	('N19', 'N25'): [20000, 0.000012],
	('EN', 'N11'): [20000, 0.000012],
	('N11', 'N12'): [20000, 0.000012],
	('N19', 'N20'): [20000, 0.000012],
	('N18', 'N19'): [20000, 0.000012],
	('N17_1', 'N18'): [20000, 0.000012],
	('EH', 'N17'): [20000, 0.000012],
	('N23', 'N23_1'): [20000, 0.000012],
	('N12', 'N22'): [20000, 0.000012],
	('N20', 'XH'): [20000, 0.000012],
	('EH_aux0', 'EH'): [100, 0.000012],
	('EH_aux1', 'EH_aux0'): [10000, 0.000012],
	('EH_NDin1', 'EH_aux1'): [10000, 0.000012],
	('EH_ND', 'EH_NDin2'): [10000, 0.000012],
	('EH_aux2', 'EH_aux1'): [10000, 0.000012],
	('EH_aux3', 'EH_aux2'): [1000, 0.000012],
	('EH_HDin2', 'EH_HDin1'): [1000, 0.000012],
	('EH_HD', 'EH_HDin2'): [10000, 0.000012],
	('EN_aux0', 'EN'): [100, 0.000012],
	('EN_aux1', 'EN_aux0'): [10000, 0.000012],
	('EN_NDin1', 'EN_aux1'): [10000, 0.000012],
	('EN_ND', 'EN_NDin2'): [10000, 0.000012],
	('EN_aux2', 'EN_aux1'): [10000, 0.000012],
	('EN_aux3', 'EN_aux2'): [1000, 0.000012],
	('EN_HDin2', 'EN_HDin1'): [1000, 0.000012],
	('EN_HD', 'EN_HDin2'): [10000, 0.000012],
	('N22', 'N22_aux'): [1, 0.000012],
	('N23_aux', 'N23'): [1, 0.000012]
})

# this cannot be put into the multidicts for pipes and resistors, as no keys could be appended then
diameter = {
	#pipes
	('N12', 'N13'): 0.9,
	('N13', 'N14'): 0.9,
	('N26', 'N13'): 0.9,
	('N14', 'XN'): 0.9,
	('N19', 'N25'): 0.9,
	('EN', 'N11'): 0.9,
	('N11', 'N12'): 0.9,
	('N19', 'N20'): 0.9,
	('N18', 'N19'): 0.9,
	('N17_1', 'N18'): 0.9,
	('EH', 'N17'): 0.9,
	('N23', 'N23_1'): 0.9,
	('N12', 'N22'): 0.9,
	('N20', 'XH'): 0.9,
	('EH_aux0', 'EH'): 0.5,
	('EH_aux1', 'EH_aux0'): 2,
	('EH_NDin1', 'EH_aux1'): 2,
	('EH_ND', 'EH_NDin2'): 2,
	('EH_aux2', 'EH_aux1'): 2,
	('EH_aux3', 'EH_aux2'): 0.3,
	('EH_HDin2', 'EH_HDin1'): 0.3,
	('EH_HD', 'EH_HDin2'): 2,
	('EN_aux0', 'EN'): 0.5,
	('EN_aux1', 'EN_aux0'): 2,
	('EN_NDin1', 'EN_aux1'): 2,
	('EN_ND', 'EN_NDin2'): 2,
	('EN_aux2', 'EN_aux1'): 2,
	('EN_aux3', 'EN_aux2'): 0.3,
	('EN_HDin2', 'EN_HDin1'): 0.3,
	('EN_HD', 'EN_HDin2'): 2,
	('N22', 'N22_aux'): 1,
	('N23_aux', 'N23'): 1,
	# resistors
	('N25', 'N26_aux'): 0.9
}

# special pipes
special = gp.tuplelist([
	('EH_aux0', 'EH'),
	('EN_aux0', 'EN')
])

connections = pipes + resistors + valves + flap_traps + compressors

non_pipes = [x for x in connections if x not in pipes]


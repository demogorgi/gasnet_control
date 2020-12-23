#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# this file contains the simulator model

from constants import *
from functions import *
import gurobipy as gp
from gurobipy import GRB

import importlib
import sys
import re
wd = sys.argv[1].replace("/",".")
wd = re.sub(r'\.$', '', wd)
#print(wd + ".init_scenario")

init_s = importlib.import_module(wd + ".init_scenario")
no = importlib.import_module(wd + ".nodes")
co = importlib.import_module(wd + ".connections")

def joiner(s):
    return '^'.join(map(str,s))

def get_agent_decision(deep_agent_decision,i):
    r = range(i,-1,-1)
    for i in r:
        if i in deep_agent_decision:
            #print("---------------------------------------------------------")
            #print("%s[%d] = %f" % (deep_agent_decision,i,deep_agent_decision[i]))
            #print("---------------------------------------------------------")
            return deep_agent_decision[i]

def simulate(agent_decisions,compressors,t,dt):
    # Model
    m = gp.Model()
    
    ## Node variables
    # pressure for every node
    var_node_p = m.addVars(no.nodes, lb=1.01325, ub=501.01325, name="var_node_p")
    # flow slack variables for exits, with obj coefficient
    var_boundary_node_flow_slack_positive = m.addVars(no.exits, obj=1, name="var_boundary_node_flow_slack_positive");
    var_boundary_node_flow_slack_negative = m.addVars(no.exits, obj=1, name="var_boundary_node_flow_slack_negative");
    # pressure slack variables for entries, with obj coefficient
    var_boundary_node_pressure_slack_positive = m.addVars(no.entries, obj=10, name="var_boundary_node_pressure_slack_positive");
    var_boundary_node_pressure_slack_negative = m.addVars(no.entries, obj=10, name="var_boundary_node_pressure_slack_negative");
    # node inflow for entries and exits (inflow is negative for exits)
    var_node_Qo_in = m.addVars(no.nodes, lb=-10000, ub=10000, name="var_node_Qo_in")
    
    ## Pipe variables
    var_pipe_Qo_in = m.addVars(co.pipes, lb=-10000, ub=10000, name="var_pipe_Qo_in")
    var_pipe_Qo_out = m.addVars(co.pipes, lb=-10000, ub=10000, name="var_pipe_Qo_out")
    
    ## Non pipe connections variables
    var_non_pipe_Qo = m.addVars(co.non_pipes, lb=-10000, ub=10000, name="var_non_pipe_Qo")
    
    ## Flap trap variables
    flaptrap = m.addVars(co.flap_traps, vtype=GRB.BINARY, name="flaptrap")
    
    ## Auxilary variables
    # v * Q for pressure drop for pipes ...
    vQp = m.addVars(co.pipes, lb=-GRB.INFINITY, name="vQp") #:= ( vi(l,r) * var_pipe_Qo_in[l,r] + vo(l,r) * var_pipe_Qo_out[l,r] ) * rho / 3.6;
    # ... and resistors
    vQr = m.addVars(co.resistors, lb=-GRB.INFINITY, name="vQr") #:= vm(l,r) * var_non_pipe_Qo[l,r] * rho / 3.6;
    
    # Pressure difference p_out minus p_in
    delta_p = m.addVars(co.connections, lb=-Mp, ub=Mp, name="delta_p") #:= var_node_p[l] - var_node_p[r];
    
    ## Auxiliary variables to track dispatcher agent decisions
    va_DA = m.addVars(co.valves, name="va_DA");
    zeta_DA = m.addVars(co.resistors, name="zeta_DA");
    gas_DA = m.addVars(co.compressors, name="gas_DA");
    compressor_DA = m.addVars(co.compressors, name="compressor_DA");
    
    ## Auxiliary variables to track trader agent decisions
    exit_nom_TA = m.addVars(no.exits, lb=-GRB.INFINITY, name="exit_nom_TA")
    entry_nom_TA = m.addVars(co.special, name="entry_nom_TA")
    
    ## Auxiliary variable to track deviations from entry nominations ...
    nom_entry_slack_DA = m.addVars(co.special, lb=-GRB.INFINITY, name="nom_entry_slack_DA")
    # ... and from exit nominations
    nom_exit_slack_DA = m.addVars(no.exits, lb=-GRB.INFINITY, name="nom_exit_slack_DA")
    
    ## Auxiliary variable to track balances
    scenario_balance_TA = m.addVar(lb=-GRB.INFINITY, name="scenario_balance_TA")
    
    ## Auxiliary variable to track pressure violations
    ub_pressure_violation_DA = m.addVars(no.nodes, lb=-GRB.INFINITY, name="ub_pressure_violation_DA")
    lb_pressure_violation_DA = m.addVars(no.nodes, lb=-GRB.INFINITY, name="lb_pressure_violation_DA")
    
    ## Auxiliary variable to track smoothed flow over S-pipes
    smoothed_special_pipe_flow_DA = m.addVars(co.special, lb=-GRB.INFINITY, name="smoothed_special_pipe_flow_DA")

    # From here on the constraints have to be added.
    #m.addConstr(va_DA[("N23","N23_1")] == -1, "test")

    #### auxilary constraints
    ## v * Q for pressure drop (for pipes and resistors)
    #subto vxQp:
    #      forall <l,r> in P: vQp[l,r] == ( vi(l,r) * var_pipe_Qo_in[l,r] + vo(l,r) * var_pipe_Qo_out[l,r] ) * rho / 3.6;
    m.addConstrs((vQp[p] == ( vi(t,*p) * var_pipe_Qo_in[p] + vo(t,*p) * var_pipe_Qo_out[p] ) * rho / 3.6 for p in co.pipes), name='vxQp')
    #
    #subto vxQr:
    #      forall <l,r> in RE: vQr[l,r] == vm(l,r) * var_non_pipe_Qo[l,r] * rho / 3.6;
    m.addConstrs((vQr[r] == vm(t,*r) * var_non_pipe_Qo[r] * rho / 3.6 for r in co.resistors), name='vxQr')
    #
    ## constraints to track trader agent's decisions
    #subto nomx:
    #      forall <x> in X: exit_nom_TA[x] == exit_nom[x];
    m.addConstrs((exit_nom_TA[x] == get_agent_decision(agent_decisions["exit_nom"]["X"][x],t) for x in no.exits), name='nomx')
    #
    #subto nome:
    #      forall <l,r> in S: entry_nom_TA[l,r] == entry_nom[l,r];
    m.addConstrs((entry_nom_TA[s] == get_agent_decision(agent_decisions["entry_nom"]["S"][joiner(s)],t) for s in co.special), name='nome')
    #
    ## constraints to track dispatcher agent's decisions
    #subto va_mode:
    #      forall <l,r> in VA: va_DA[l,r] == va[l,r];
    m.addConstrs((va_DA[v] == get_agent_decision(agent_decisions["va"]["VA"][joiner(v)],t) for v in co.valves), name='va_mode')
    #
    #subto re_drag:
    #      forall <l,r> in RE: zeta_DA[l,r] == zeta[l,r];
    m.addConstrs((zeta_DA[r] == get_agent_decision(agent_decisions["zeta"]["RE"][joiner(r)],t) for r in co.resistors), name='re_drag')
    #
    #subto cs_fuel:
    #      forall <l,r> in CS: gas_DA[l,r] == gas[l,r];
    m.addConstrs((gas_DA[cs] == get_agent_decision(agent_decisions["gas"]["CS"][joiner(cs)],t) for cs in co.compressors), name='cs_fuel')
    #
    #subto cs_mode:
    #      forall <l,r> in CS: compressor_DA[l,r] == compressor[l,r];
    m.addConstrs((compressor_DA[cs] == get_agent_decision(agent_decisions["compressor"]["CS"][joiner(cs)],t) for cs in co.compressors), name='cs_mode')
    #
    ## constraints to track smoothing of special pipe flows
    m.addConstrs((smoothed_special_pipe_flow_DA[s] == ( q_in_old(t,s) + q_out_old(t,s) + var_pipe_Qo_in[s] + var_pipe_Qo_out[s] ) / 4 for s in co.special), name='special_pipe_smoothing')
    #
    ## pressure difference p_out minus p_in
    #subto dp:
    #      forall <l,r> in CN: delta_p[l,r] == var_node_p[l] - var_node_p[r];
    m.addConstrs((delta_p[c] == var_node_p[c[0]] - var_node_p[c[1]] for c in co.connections), name='dp')
    #
    #
    #### node and pipe model ###
    #
    ## forall nodes: connection_inflow - connection_outflow = node_inflow
    #subto c_e_cons_conserv_Qo:
    #      forall <n> in NO: var_node_Qo_in[n] - sum <n,i> in P: var_pipe_Qo_in[n,i] + sum <i,n> in P: var_pipe_Qo_out[i,n] - sum <n,i> in CN without P: var_non_pipe_Qo[n,i] + sum <i,n> in CN without P: var_non_pipe_Qo[i,n] == 0;
    m.addConstrs((var_node_Qo_in[n] - var_pipe_Qo_in.sum(n, '*') +  var_pipe_Qo_out.sum('*', n) - var_non_pipe_Qo.sum(n, '*') + var_non_pipe_Qo.sum('*', n) == 0 for n in no.nodes), name='c_e_cons_conserv_Qo')
    #
    ## forall inner nodes: node_inflow = 0
    #subto null_setzen:
    #      forall <n> in N: var_node_Qo_in[n] == 0;
    m.addConstrs((var_node_Qo_in[n] == 0 for n in no.innodes), name='innode_flow')
    #
    ## flow slack for boundary nodes
    #subto c_u_cons_boundary_node_wflow_slack_1:
    #      forall <x> in X: - var_boundary_node_flow_slack_positive[x] + var_node_Qo_in[x] <= + exit_nom[x];
    m.addConstrs((- var_boundary_node_flow_slack_positive[x] + var_node_Qo_in[x] <= get_agent_decision(agent_decisions["exit_nom"]["X"][x],t) for x in no.exits), name='c_u_cons_boundary_node_wflow_slack_1')
    #
    #subto c_u_cons_boundary_node_wflow_slack_2:
    #      forall <x> in X: - var_boundary_node_flow_slack_negative[x] - var_node_Qo_in[x] <= - exit_nom[x];
    m.addConstrs((- var_boundary_node_flow_slack_negative[x] - var_node_Qo_in[x] <= - get_agent_decision(agent_decisions["exit_nom"]["X"][x],t) for x in no.exits), name='c_u_cons_boundary_node_wflow_slack_2')
    #
    ## pressure slack for boundary nodes
    #subto c_u_cons_boundary_node_wpressure_slack_1:
    #      forall <e> in E: - var_boundary_node_pressure_slack_positive[e] + var_node_p[e] <= + pressure[e];
    m.addConstrs((- var_boundary_node_pressure_slack_positive[e] + var_node_p[e] <= no.pressure[e] for e in no.entries), name='c_u_cons_boundary_node_wpressure_slack_1')
    #
    #subto c_u_cons_boundary_node_wpressure_slack_2:
    #      forall <e> in E: - var_boundary_node_pressure_slack_negative[e] - var_node_p[e] <= - pressure[e];
    m.addConstrs((- var_boundary_node_pressure_slack_negative[e] - var_node_p[e] <= - no.pressure[e] for e in no.entries), name='c_u_cons_boundary_node_wpressure_slack_2')
    #
    ## continuity equation
    #subto c_e_cons_pipe_continuity: forall <l,r> in P:
    #      b2p * ( var_node_p[l] + var_node_p[r] - p_old(l) - p_old(r) ) + rho / 3.6 * ( 2 * rtza(l,r) * dt ) / L[l,r] * ( var_pipe_Qo_out[l,r] - var_pipe_Qo_in[l,r] ) == 0; # Felix bar -> Pa und 1000 m³/h -> kg/s --> rho / 3.6
    m.addConstrs(( b2p * ( var_node_p[p[0]] + var_node_p[p[1]] - p_old(t,p[0]) - p_old(t,p[1]) ) + rho / 3.6 * ( 2 * rtza(t,*p) * dt ) / co.length[p] * ( var_pipe_Qo_out[p] - var_pipe_Qo_in[p] ) == 0 for p in co.pipes), name='c_e_cons_pipe_continuity')
    #
    ## pressure drop equation
    #subto c_e_cons_pipe_momentum: forall <l,r> in P:
    #      b2p * delta_p[l,r] == xip(l, r) * vQp[l,r];
    m.addConstrs(( b2p * delta_p[p] == xip(p) * vQp[p] for p in co.pipes), name='c_e_cons_pipe_momentum')
    #
    #
    #### valve model ###
    #
    #subto valve_eq_one: forall <l,r> in VA:
    #      var_node_p[l] - var_node_p[r] <= Mp * ( 1 - va[l,r] ); # Eq. (10) station model paper
    m.addConstrs((var_node_p[v[0]] - var_node_p[v[1]] <= Mp * ( 1 - get_agent_decision(agent_decisions["va"]["VA"][joiner(v)],t) ) for v in co.valves), name='valve_eq_one')
    #
    #subto valve_eq_two: forall <l,r> in VA:
    #      var_node_p[l] - var_node_p[r] >= - Mp * ( 1 - va[l,r] ); # Eq. (11) station model paper
    m.addConstrs((var_node_p[v[0]] - var_node_p[v[1]] >= - Mp * ( 1 - get_agent_decision(agent_decisions["va"]["VA"][joiner(v)],t) ) for v in co.valves), name='valve_eq_two')
    #
    #subto valve_eq_three: forall <l,r> in VA:
    #      var_non_pipe_Qo[l,r] <= Mq * va[l,r]; # Eq. (12) station model paper
    m.addConstrs((var_non_pipe_Qo[v] <= Mq * get_agent_decision(agent_decisions["va"]["VA"][joiner(v)],t) for v in co.valves), name='valve_eq_three')
    #
    #subto valve_eq_four: forall <l,r> in VA:
    #      var_non_pipe_Qo[l,r] >= - Mq * va[l,r]; # Eq. (13) station model paper
    m.addConstrs((var_non_pipe_Qo[v] >= - Mq * get_agent_decision(agent_decisions["va"]["VA"][joiner(v)],t) for v in co.valves), name='valve_eq_four')
    #
    #### resistor model ###
    ## pressure drop equation
    #subto resistor_eq: forall <l,r> in RE:
    #      b2p * delta_p[l,r] == xir(l,r) * vQr[l,r];
    #m.addConstrs(( b2p * delta_p[r] == xir(r,agent_decisions["zeta"]["RE"][joiner(r)]) * vQr[r] for r in co.resistors), name='resistor_eq')
    n = 5 # parameter to form the zeta-curve
    m.addConstrs(( b2p * delta_p[r] == xir(r, 10 ** 8 * get_agent_decision(agent_decisions["zeta"]["RE"][joiner(r)],t) ** n * 10 ** ( 4 * ( 1 - n ) ) ) * vQr[r] for r in co.resistors), name='resistor_eq')
#    m.addConstrs(( b2p * delta_p[r] == xir(r,100000000 * zeta_DA[r]) * vQr[r] for r in co.resistors), name='resistor_eq')
    #
    #### flap trap model ###
    #
    #subto flap_trap_eq_one: forall <l,r> in FT:
    #      var_non_pipe_Qo[l,r] >= 0;
    m.addConstrs((var_non_pipe_Qo[f] >= 0 for f in co.flap_traps), name='flap_trap_eq_one')
    #
    #subto flap_trap_eq_two: forall <l,r> in FT:
    #      var_non_pipe_Qo[l,r] <= Mq * flaptrap[l,r];
    m.addConstrs((var_non_pipe_Qo[f] <= Mq * flaptrap[f] for f in co.flap_traps), name='flap_trap_eq_two')
    #
    #subto flap_trap_eq_three: forall <l,r> in FT:
    #      delta_p[l,r] <= Mp * (1 - flaptrap[l,r] ) + 0.2 * b2p;
    m.addConstrs((delta_p[f] <= Mq * ( 1 - flaptrap[f] ) + 0.2 * b2p for f in co.flap_traps), name='flap_trap_eq_three')
    #
    #subto flap_trap_eq_four: forall <l,r> in FT:
    #      - delta_p[l,r] <= Mp * (1 - flaptrap[l,r] );
    m.addConstrs(( - delta_p[f] <= Mq * ( 1 - flaptrap[f] ) for f in co.flap_traps), name='flap_trap_eq_four')
    #
    #subto flap_trap_eq_five: forall <l,r> in FT:
    #      delta_p[l,r] <= 0;
    m.addConstrs((delta_p[f] <= 0 for f in co.flap_traps), name='flap_trap_eq_five')
    #
    #### compressor model ###
    #subto compressor_eq_one: forall <l,r> in CS:
    #      var_non_pipe_Qo[l,r] >= 0;
    m.addConstrs((var_non_pipe_Qo[cs] >= 0 for cs in co.compressors), name='compressor_eq_one')
    #
    #subto compressor_eq_two: forall <l,r> in CS:
    m.addConstrs((var_non_pipe_Qo[cs] == get_agent_decision(agent_decisions["compressor"]["CS"][joiner(cs)],t) * 3.6 * p_old(t,cs[0]) * phi_new(get_agent_decision(agent_decisions["compressor"]["CS"][joiner(cs)],t),compressors[joiner(cs)]["phi_min"],compressors[joiner(cs)]["phi_max"],compressors[joiner(cs)]["pi_1"],compressors[joiner(cs)]["pi_2"],compressors[joiner(cs)]["L_min_pi"],compressors[joiner(cs)]["L_max_pi"],compressors[joiner(cs)]["L_min_phi"],compressors[joiner(cs)]["p_i_min"],compressors[joiner(cs)]["p_i_max"],compressors[joiner(cs)]["L_max_pi"],compressors[joiner(cs)]["eta"],get_agent_decision(agent_decisions["gas"]["CS"][joiner(cs)],t),p_old(t,cs[0]),p_old(t,cs[1])) for cs in co.compressors), name='compressor_eq_two')
    #
    #### Entrymodellierung ###
    #subto entry_flow_model:
    #      forall <e> in E: var_node_Qo_in[e] <= entry_flow_bound[e];
    m.addConstrs((var_node_Qo_in[e] <= no.entry_flow_bound[e] for e in no.entries), name='entry_flow_model')
    #
    #subto nomination_check:
    #      forall <l,r> in S: var_pipe_Qo_out[l,r] + nom_entry_slack_DA[l,r] == entry_nom[l,r];
    m.addConstrs((var_pipe_Qo_out[s] + nom_entry_slack_DA[s] == get_agent_decision(agent_decisions["entry_nom"]["S"][joiner(s)],t) for s in co.special), name='nomination_check')
    #
    #subto track_scenario_balance:
    #      sum <l,r> in S: entry_nom[l,r] + sum <x> in X: exit_nom[x] == scenario_balance_TA;
    #m.addConstr((sum(agent_decisions["entry_nom"]["S"].values()) + sum(agent_decisions["exit_nom"]["X"].values()) == scenario_balance_TA), name='track_scenario_balance')
    print("--->")
    print([get_agent_decision(agent_decisions["exit_nom"]["X"][x],t) for x in no.exits])
    print("--->")
    print([joiner(s) for s in co.special])
    print([get_agent_decision(agent_decisions["entry_nom"]["S"][joiner(s)],t) for s in co.special])
    print("--->")
    m.addConstr((sum([get_agent_decision(agent_decisions["entry_nom"]["S"][joiner(s)],t) for s in co.special]) + sum([get_agent_decision(agent_decisions["exit_nom"]["X"][x],t) for x in no.exits]) == scenario_balance_TA), 'track_scenario_balance')
    #
    #subto track_exit_nomination_slack:
    #      forall <x> in X: nom_exit_slack_DA[x] == var_boundary_node_flow_slack_positive[x] - var_boundary_node_flow_slack_negative[x];
    m.addConstrs((nom_exit_slack_DA[x] == var_boundary_node_flow_slack_positive[x] - var_boundary_node_flow_slack_negative[x] for x in no.exits), name='track_exit_nomination_slack')
    #
    #subto track_ub_pressure_violation:
    #      forall <n> in NO: var_node_p[n] - pressureLimitsUpper[n] == ub_pressure_violation_DA[n];
    m.addConstrs((var_node_p[n] - no.pressure_limits_upper[n] == ub_pressure_violation_DA[n] for n in no.nodes), name='track_ub_pressure_violation')
    #
    #subto track_lb_pressure_violation:
    #      forall <n> in NO: pressureLimitsLower[n] - var_node_p[n] == lb_pressure_violation_DA[n];
    m.addConstrs((no.pressure_limits_lower[n] - var_node_p[n] == lb_pressure_violation_DA[n] for n in no.nodes), name='track_lb_pressure_violation')
    
    return m

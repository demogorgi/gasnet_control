#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# this file contains the simulator model

from constants import *
from functions import *
import gurobipy as gp
from gurobipy import GRB
from params import *

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
    r = range(i,-2*config['nomination_freq']-1,-1)
    for i in r:
        if i in deep_agent_decision:
            #print("---------------------------------------------------------")
            #print("%s[%d] = %f" % (deep_agent_decision,i,deep_agent_decision[i]))
            #print("---------------------------------------------------------")
            return deep_agent_decision[i]

def simulate(agent_decisions,compressors,dt):
    # Model
    m = gp.Model()
    m.setParam("LogToConsole", 0)

    #### From here on the variables have to be added. ####
    var_node_p = {}
    var_boundary_node_flow_slack_positive = {}
    var_boundary_node_flow_slack_negative = {}
    var_boundary_node_pressure_slack_positive = {}
    var_boundary_node_pressure_slack_negative = {}
    var_node_Qo_in = {}
    var_pipe_Qo_in = {}
    var_pipe_Qo_out = {}
    var_non_pipe_Qo = {}
    checkvalve = {}
    vQp = {}
    vQr = {}
    delta_p = {}
    va_DA = {}
    zeta_DA = {}
    gas_DA = {}
    compressor_DA = {}
    nom_entry_slack_DA = {}
    ## Node variables
    for tstep in range(numSteps):
        # pressure for every node
        var_node_p[tstep] = m.addVars(no.nodes, lb=1.01325, ub=501.01325, name=f"var_node_p_{tstep}")
        # flow slack variables for exits, with obj coefficient
        var_boundary_node_flow_slack_positive[tstep] = m.addVars(no.exits, obj=1, name=f"var_boundary_node_flow_slack_positive_{tstep}");
        var_boundary_node_flow_slack_negative[tstep] = m.addVars(no.exits, obj=1, name=f"var_boundary_node_flow_slack_negative_{tstep}");
        # pressure slack variables for entries, with obj coefficient
        var_boundary_node_pressure_slack_positive[tstep] = m.addVars(no.entries, obj=10, name=f"var_boundary_node_pressure_slack_positive_{tstep}");
        var_boundary_node_pressure_slack_negative[tstep] = m.addVars(no.entries, obj=10, name=f"var_boundary_node_pressure_slack_negative_{tstep}");
        # node inflow for entries and exits (inflow is negative for exits)
        var_node_Qo_in[tstep] = m.addVars(no.nodes, lb=-10000, ub=10000, name=f"var_node_Qo_in_{tstep}")

        ## Pipe variables
        var_pipe_Qo_in[tstep] = m.addVars(co.pipes, lb=-10000, ub=10000, name=f"var_pipe_Qo_in_{tstep}")
        var_pipe_Qo_out[tstep] = m.addVars(co.pipes, lb=-10000, ub=10000, name=f"var_pipe_Qo_out_{tstep}")

        ## Non pipe connections variables
        var_non_pipe_Qo[tstep] = m.addVars(co.non_pipes, lb=-10000, ub=10000, name=f"var_non_pipe_Qo_{tstep}")

        ## Flap trap variables
        checkvalve[tstep] = m.addVars(co.check_valves, vtype=GRB.BINARY, name=f"checkvalve_{tstep}")

        ## Auxiliary variables v * Q for pressure drop for pipes ...
        vQp[tstep] = m.addVars(co.pipes, lb=-GRB.INFINITY, name=f"vQp_{tstep}") #:= ( vi(l,r) * var_pipe_Qo_in[l,r] + vo(l,r) * var_pipe_Qo_out[l,r] ) * rho / 3.6;
        # ... and resistors
        vQr[tstep] = m.addVars(co.resistors, lb=-GRB.INFINITY, name=f"vQr_{tstep}") #:= vm(l,r) * var_non_pipe_Qo[l,r] * rho / 3.6;

        ## Auxiliary variable pressure difference p_out minus p_in
        delta_p[tstep] = m.addVars(co.connections, lb=-Mp, ub=Mp, name=f"delta_p_{tstep}") #:= var_node_p[l] - var_node_p[r];

        ## Auxiliary variables to track dispatcher agent decisions
        va_DA[tstep] = m.addVars(co.valves, name=f"va_DA_{tstep}", vtype=GRB.BINARY)
        zeta_DA[tstep] = m.addVars(co.resistors, name=f"zeta_DA_{tstep}");
        gas_DA[tstep] = m.addVars(co.compressors, name=f"gas_DA_{tstep}")
        compressor_DA[tstep] = m.addVars(co.compressors, name=f"compressor_DA_{tstep}")

        ## Auxiliary variables to track trader agent decisions
        exit_nom_TA = m.addVars(no.exits, lb=-GRB.INFINITY, name=f"exit_nom_TA_{tstep}")
        entry_nom_TA = m.addVars(co.special, name=f"entry_nom_TA_{tstep}")

        ## Auxiliary variable to track deviations from entry nominations ...
        nom_entry_slack_DA[tstep] = m.addVars(co.special, lb=-GRB.INFINITY, name=f"nom_entry_slack_DA_{tstep}")
        # ... and from exit nominations
        nom_exit_slack_DA = m.addVars(no.exits, lb=-GRB.INFINITY, name=f"nom_exit_slack_DA_{tstep}")

        ## Auxiliary variable to track balances
        scenario_balance_TA = m.addVar(lb=-GRB.INFINITY, name=f"scenario_balance_TA_{tstep}")

        ## Auxiliary variable to track pressure violations at exits
        ub_pressure_violation_DA = m.addVars(no.exits, lb=-GRB.INFINITY, name=f"ub_pressure_violation_DA_{tstep}")
        lb_pressure_violation_DA = m.addVars(no.exits, lb=-GRB.INFINITY, name=f"lb_pressure_violation_DA_{tstep}")

        ## Auxiliary variable to track smoothed flow over S-pipes
        smoothed_special_pipe_flow_DA = m.addVars(co.special, lb=-GRB.INFINITY, name=f"smoothed_special_pipe_flow_DA_{tstep}")

    #### From here on the constraints have to be added. ####

    ### AUXILIARY CONSTRAINTS ###
    #
    m.addConstrs((var_node_p[0][n] == states[-1]["p"][n] for n in no.nodes), name='node_init')
    m.addConstrs((var_pipe_Qo_in[0][p] == states[-1]["q_in"][p] for p in co.pipes), name='pipes_in_init')
    m.addConstrs((var_pipe_Qo_out[0][p] == states[-1]["q_out"][p] for p in co.pipes), name='pipes_out_init')
    m.addConstrs((var_non_pipe_Qo[0][np] == states[-1]["q"][np] for np in co.non_pipes), name='compressor_init')
    ## v * Q for pressure drop for pipes ...
    m.addConstrs((vQp[0][p] == (((Rs * Tm * zm(states[-2]["p"][p[0]],states[-2]["p"][p[1]]) / A(co.diameter[p]))
                                     * rho / 3.6 * states[-2]["q_in"][p] / (b2p * states[-2]["p"][p[0]]))
                                    * var_pipe_Qo_in[0][p] +
                                    ((Rs * Tm * zm(states[-2]["p"][p[0]],states[-2]["p"][p[1]]) / A(co.diameter[p]))
                                     * rho / 3.6 * states[-2]["q_out"][p] / (b2p * states[-2]["p"][p[1]]))
                                    * var_pipe_Qo_out[0][p])
                  * rho / 3.6 for p in co.pipes), name=f'vxQp_{tstep}')
    for tstep in range(1, numSteps):
        m.addConstrs((vQp[tstep][p] == ( ((Rs * Tm * zm(var_node_p[tstep - 1][p[0]],var_node_p[tstep - 1][p[1]]) / A(co.diameter[p]))
                                         * rho / 3.6 * var_pipe_Qo_in[tstep - 1][p] / ( b2p * var_node_p[tstep - 1][p[0]] ))
                                         * var_pipe_Qo_in[tstep][p] +
                                         ((Rs * Tm * zm(var_node_p[tstep - 1][p[0]],var_node_p[tstep - 1][p[1]]) / A(co.diameter[p]))
                                         * rho / 3.6 * var_pipe_Qo_out[tstep - 1][p] / ( b2p * var_node_p[tstep - 1][p[1]] ))
                                         * var_pipe_Qo_out[tstep][p] )
                      * rho / 3.6 for p in co.pipes), name=f'vxQp_{tstep}')
        # original constraint:
        #vQp[p] == (vi(t, *p) * var_pipe_Qo_in[p] + vo(t, *p) * var_pipe_Qo_out[
        #    p]) * rho / 3.6
        # vi(t,i,o) = rtza(t,i,o) * rho / 3.6 * q_in_old(t,(i,o)) / ( b2p * p_old(t,i) )
        # vo(t,i,o) = rtza(t,i,o) * rho / 3.6 * q_out_old(t,(i,o)) / ( b2p * p_old(t,o) )
        # rtza(t,i,o) = Rs * Tm * zm(p_old(t,i),p_old(t,o)) / A(co.diameter[(i,o)])
        # (the obvious 'divided by two' is carried out in the function xip (in fuctions.py) according to eqn. 18 in the Station_Model_Paper.pdf (docs))
    # ... and resistors
    m.addConstrs((vQr[0][r] == (rho / 3.6 * ((Rs * Tm * zm(states[-2]["p"][r[0]],states[-2][r[1]]) / A(co.diameter[r]))))
                  * var_non_pipe_Qo[0][r] * rho / 3.6 for r in co.resistors), name=f'vxQr_{0}')
    for tstep in range(1, numSteps):

        m.addConstrs((vQr[tstep][r] == (rho / 3.6 * ( (Rs * Tm * zm(var_node_p[tstep - 1][r[0]],var_node_p[tstep - 1][r[1]]) / A(co.diameter[r]))))
                      * var_non_pipe_Qo[tstep][r] * rho / 3.6 for r in co.resistors), name=f'vxQr_{tstep}')
        # original constraint:
        # vQr[r] == vm(t,*r) * var_non_pipe_Qo[r] * rho / 3.6
        # vm(t,i,o) =  max(rho / 3.6 * ( rtza(t,i,o) * q_old(t,(i,o)) ) / 2 * 1 / b2p * ( 1 / p_old(t,i) + 1 / p_old(t,o) ), 2)

    ## constraints only to track trader agent's decisions
    m.addConstrs((exit_nom_TA[x] == get_agent_decision(agent_decisions["exit_nom"]["X"][x],t) for x in no.exits), name='nomx')
    m.addConstrs((entry_nom_TA[s] == get_agent_decision(agent_decisions["entry_nom"]["S"][joiner(s)],(t//config['nomination_freq']-2)*config['nomination_freq']) for s in co.special), name='nome')
    #m.addConstrs((entry_nom_TA[s] == get_agent_decision(agent_decisions["entry_nom"]["S"][joiner(s)],t//config['nomination_freq']) for s in co.special), name='nome')
    #
    ## constraints only to track dispatcher agent's decisions
    m.addConstrs((va_DA[v] == get_agent_decision(agent_decisions["va"]["VA"][joiner(v)],t) for v in co.valves), name='va_mode')
    m.addConstrs((zeta_DA[r] == get_agent_decision(agent_decisions["zeta"]["RE"][joiner(r)],t) for r in co.resistors), name='re_drag')
    m.addConstrs((gas_DA[cs] == get_agent_decision(agent_decisions["gas"]["CS"][joiner(cs)],t) for cs in co.compressors), name='cs_fuel')
    m.addConstrs((compressor_DA[cs] == get_agent_decision(agent_decisions["compressor"]["CS"][joiner(cs)],t) for cs in co.compressors), name='cs_mode')
    #
    ## constraints only to track smoothing of special pipe flows
    m.addConstrs((smoothed_special_pipe_flow_DA[s] == ( q_in_old(t,s) + q_out_old(t,s) + var_pipe_Qo_in[s] + var_pipe_Qo_out[s] ) / 4 for s in co.special), name='special_pipe_smoothing')
    #
    for tstep in range(numSteps):
        ## pressure difference p_out minus p_in
        m.addConstrs((delta_p[tstep][c] == var_node_p[tstep][c[0]] - var_node_p[tstep][c[1]] for c in co.connections), name=f'dp_{tstep}')
    #
    #
    ### NODE AND PIPE MODEL ###
    # This part is inspired by section "2.2 Pipes" of https://opus4.kobv.de/opus4-zib/frontdoor/index/index/docId/7364.
	# In our simulator setting we do not have to compute the full time horizon at once. We can do it step by step.
	# This allows us to use variables from the previous step to approximate nonlinear terms in the current step.
	# This is impossible in the opimization setting in the paper.
    #TODO: check if flow conservation is true for initial scenario
    for tstep in range(1, numSteps):
        ## forall nodes: connection_inflow - connection_outflow = node_inflow
        m.addConstrs((var_node_Qo_in[tstep][n] - var_pipe_Qo_in[tstep].sum(n, '*') +  var_pipe_Qo_out[tstep].sum('*', n) - var_non_pipe_Qo[tstep].sum(n, '*') + var_non_pipe_Qo[tstep].sum('*', n) == 0 for n in no.nodes), name=f'c_e_cons_conserv_Qo_{tstep}')
        #
        ## forall inner nodes: node_inflow = 0
        m.addConstrs((var_node_Qo_in[tstep][n] == 0 for n in no.innodes), name=f'innode_flow_{tstep}')
        #
    for tstep in range(numSteps):
        ## flow slack for boundary nodes
        m.addConstrs((- var_boundary_node_flow_slack_positive[tstep][x] + var_node_Qo_in[tstep][x] <= get_agent_decision(agent_decisions["exit_nom"]["X"][x],tstep) for x in no.exits), name=f'c_u_cons_boundary_node_wflow_slack_1_{tstep}')
        m.addConstrs((- var_boundary_node_flow_slack_negative[tstep][x] - var_node_Qo_in[tstep][x] <= - get_agent_decision(agent_decisions["exit_nom"]["X"][x],tstep) for x in no.exits), name=f'c_u_cons_boundary_node_wflow_slack_2_{tstep}')
        #
        ## pressure slack for boundary nodes
        m.addConstrs((- var_boundary_node_pressure_slack_positive[tstep][e] + var_node_p[tstep][e] <= no.pressure[e] for e in no.entries), name=f'c_u_cons_boundary_node_wpressure_slack_1_{tstep}')
        m.addConstrs((- var_boundary_node_pressure_slack_negative[tstep][e] - var_node_p[tstep][e] <= - no.pressure[e] for e in no.entries), name=f'c_u_cons_boundary_node_wpressure_slack_2_{tstep}')
        #
    ## continuity equation and ## pressure drop equation (eqn. 20 without gravitational term from Station_Model_Paper.pdf)
    m.addConstrs((b2p * (var_node_p[0][p[0]] + var_node_p[0][p[1]] - states[-2]["p"][p[0]] - states[-2]["p"][p[1]]) + rho / 3.6 * (2 * (Rs * Tm * zm(var_node_p[0][p[0]],var_node_p[0][p[1]]) / A(co.diameter[p])) * dt) / co.length[p] * (var_pipe_Qo_out[0][p] - var_pipe_Qo_in[0][p]) == 0 for p in co.pipes), name=f'c_e_cons_pipe_continuity_{0}')
    m.addConstrs(( b2p * delta_p[0][p] == xip(p) * vQp[0][p] for p in co.pipes), name=f'c_e_cons_pipe_momentum_{0}')
    for tstep in range(1, numSteps):
        m.addConstrs(( b2p * ( var_node_p[tstep][p[0]] + var_node_p[tstep][p[1]] - var_node_p[tstep - 1][p[0]] - var_node_p[tstep - 1][p[1]] ) + rho / 3.6 * ( 2 * (Rs * Tm * zm(var_node_p[tstep][p[0]],var_node_p[tstep][p[1]]) / A(co.diameter[p])) * dt ) / co.length[p] * ( var_pipe_Qo_out[tstep][p] - var_pipe_Qo_in[tstep][p] ) == 0 for p in co.pipes), name=f'c_e_cons_pipe_continuity_{tstep}')
        #  rtza = Rs * Tm * zm(p_old(t,i),p_old(t,o)) / A(co.diameter[(i,o)])
        #
        m.addConstrs(( b2p * delta_p[tstep][p] == xip(p) * vQp[tstep][p] for p in co.pipes), name=f'c_e_cons_pipe_momentum_{tstep}')
    #
    #
    ### VALVE MODEL ###
    for tstep in range(numSteps):
        # As in section "2.4 Valves" of https://opus4.kobv.de/opus4-zib/frontdoor/index/index/docId/7364
        #
        m.addConstrs((var_node_p[tstep][v[0]] - var_node_p[tstep][v[1]] <= Mp * ( 1 - va_DA[tstep][v] ) for v in co.valves), name=f'valve_eq_one_{tstep}')
        m.addConstrs((var_node_p[tstep][v[0]] - var_node_p[tstep][v[1]] >= - Mp * ( 1 - va_DA[tstep][v] ) for v in co.valves), name=f'valve_eq_two_{tstep}')
        m.addConstrs((var_non_pipe_Qo[tstep][v] <= Mq * va_DA[tstep][v] for v in co.valves), name=f'valve_eq_three_{tstep}')
        m.addConstrs((var_non_pipe_Qo[tstep][v] >= - Mq * va_DA[tstep][v] for v in co.valves), name=f'valve_eq_four_{tstep}')
        #
        #
    ### CONTROL VALVE MODEL ###
        # Suggested by Klaus and inspired by section "2.3 Resistors" of https://opus4.kobv.de/opus4-zib/frontdoor/index/index/docId/7364.
        # We use resistors as control valves by controlling the resistors drag factor from outside
        #
        ## pressure drop equation (eqn. 21 Station_Model_Paper.pdf)
        # we use 2 ** (zeta/3) to map the "original" interval of relevant zeta values to the interaval [0,100]
        m.addConstrs(( b2p * delta_p[tstep][r] == xir(r, 2 ** ( zeta_DA[tstep][r] / 3 )) * vQr[tstep][r] for r in co.resistors), name='resistor_eq')
        #
        #
    ### CHECK VALVE MODEL ###
        #
        m.addConstrs((var_non_pipe_Qo[tstep][f] >= 0 for f in co.check_valves), name='check_valve_eq_one')
        m.addConstrs((var_non_pipe_Qo[tstep][f] <= Mq * checkvalve[tstep][f] for f in co.check_valves), name='check_valve_eq_two')
        m.addConstrs((delta_p[tstep][f] <= Mq * ( 1 - checkvalve[tstep][f] ) + 0.2 * b2p for f in co.check_valves), name='check_valve_eq_three')
        m.addConstrs(( - delta_p[tstep][f] <= Mq * ( 1 - checkvalve[tstep][f] ) for f in co.check_valves), name='check_valve_eq_four')
        m.addConstrs((delta_p[tstep][f] <= 0 for f in co.check_valves), name='check_valve_eq_five')
        #
    ### COMPRESSOR MODEL ###
    # Suggested by Klaus and described in gasnet_control/docs/Verdichterregeln.txt and gasnet_control/docs/Example_Compressor_Wheel_Map.pdf
    #
    m.addConstrs((var_non_pipe_Qo[0][cs] >= 0 for cs in co.compressors), name='compressor_eq_one_0')
    m.addConstrs((var_non_pipe_Qo[0][cs] == compressor_DA[0][cs] * 3.6 * states[-2]["p"][cs[0]] * phi_new(compressor_DA[0][cs],compressors[joiner(cs)]["phi_min"],compressors[joiner(cs)]["phi_max"],compressors[joiner(cs)]["pi_1"],compressors[joiner(cs)]["pi_2"],compressors[joiner(cs)]["pi_MIN"],compressors[joiner(cs)]["phi_MIN"],compressors[joiner(cs)]["p_in_min"],compressors[joiner(cs)]["p_in_max"],compressors[joiner(cs)]["pi_MAX"],compressors[joiner(cs)]["eta"], gas_DA[0][cs],states[-2]["p"][cs[0]],states[-2]["p"][cs[1]]) for cs in co.compressors), name=f'compressor_eq_two_{0}')
    for tstep in range(1, numSteps):
        m.addConstrs((var_non_pipe_Qo[tstep][cs] >= 0 for cs in co.compressors), name=f'compressor_eq_one_{tstep}')
        m.addConstrs((var_non_pipe_Qo[tstep][cs] == compressor_DA[tstep][cs] * 3.6 * var_node_p[tstep - 1][cs[0]] * phi_new(compressor_DA[tstep][cs],compressors[joiner(cs)]["phi_min"],compressors[joiner(cs)]["phi_max"],compressors[joiner(cs)]["pi_1"],compressors[joiner(cs)]["pi_2"],compressors[joiner(cs)]["pi_MIN"],compressors[joiner(cs)]["phi_MIN"],compressors[joiner(cs)]["p_in_min"],compressors[joiner(cs)]["p_in_max"],compressors[joiner(cs)]["pi_MAX"],compressors[joiner(cs)]["eta"],gas_DA[tstep][cs],var_node_p[tstep - 1][cs[0]],var_node_p[tstep - 1][cs[1]]) for cs in co.compressors), name=f'compressor_eq_two_{tstep}')
    #
    ### ENTRY MODEL ###
    for tstep in range(numSteps):
        # Suggested by Klaus and described in gasnet_control/docs/urmel-entry.pdf
        #
        m.addConstrs((var_node_Qo_in[tstep][e] <= no.entry_flow_bound[e] for e in no.entries), name=f'entry_flow_model_{tstep}')
        m.addConstrs((var_pipe_Qo_out[tstep][s] + nom_entry_slack_DA[tstep][s] == get_agent_decision(agent_decisions["entry_nom"]["S"][joiner(s)],(tstep//config['nomination_freq']-2)*config['nomination_freq']) for s in co.special), name=f'nomination_check_{tstep}')

    #m.addConstrs((var_pipe_Qo_out[s] + nom_entry_slack_DA[s] == get_agent_decision(agent_decisions["entry_nom"]["S"][joiner(s)],t//config['nomination_freq']) for s in co.special), name='nomination_check')
    #print("Exit nominations: {}".format([get_agent_decision(agent_decisions["exit_nom"]["X"][x],t) for x in no.exits]))
    #print("Entry nominations decision for this step ({}) was taken in step {}: ".format(t, (t//config['nomination_freq']-2)*config['nomination_freq']), [get_agent_decision(agent_decisions["entry_nom"]["S"][joiner(s)],(t//config['nomination_freq']-2)*config['nomination_freq']) for s in co.special])
    #print(agent_decisions["entry_nom"]["S"])
    #
    #
    ### TRACKING OF RELEVANT VALUES ###
    #
    m.addConstr((sum([get_agent_decision(agent_decisions["entry_nom"]["S"][joiner(s)],(t//config['nomination_freq']-2)*config['nomination_freq']) for s in co.special]) + sum([get_agent_decision(agent_decisions["exit_nom"]["X"][x],t) for x in no.exits]) == scenario_balance_TA), 'track_scenario_balance')
    #m.addConstr((sum([get_agent_decision(agent_decisions["entry_nom"]["S"][joiner(s)],t//config['nomination_freq']) for s in co.special]) + sum([get_agent_decision(agent_decisions["exit_nom"]["X"][x],t) for x in no.exits]) == scenario_balance_TA), 'track_scenario_balance')
    m.addConstrs((nom_exit_slack_DA[x] == var_boundary_node_flow_slack_positive[x] - var_boundary_node_flow_slack_negative[x] for x in no.exits), name='track_exit_nomination_slack')
    m.addConstrs((var_node_p[n] - no.pressure_limits_upper[n] == ub_pressure_violation_DA[n] for n in no.exits), name='track_ub_pressure_violation')
    m.addConstrs((no.pressure_limits_lower[n] - var_node_p[n] == lb_pressure_violation_DA[n] for n in no.exits), name='track_lb_pressure_violation')
    #
    #
    ### TESTS ###
    #m.addConstr( var_node_p['START_ND'] == 43.5, 'set_p_ND')

    return m

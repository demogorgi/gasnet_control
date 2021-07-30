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


def simulate(agent_decisions,compressors,dt,discretization):
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
    vQp_vi = {}
    vQp_vi_rhs = {}
    vQp_vo = {}
    vQp_vo_rhs = {}
    vQp_zm = {}
    vQr = {}
    vQr_zm = {}
    delta_p = {}
    comp_phi_new = {}
    comp_phi = {}
    comp_phi_frac = {}
    comp_phi_aux = {}
    comp_i = {}
    var_node_p_squ = {}
    comp_gas_aux = {}
    va_DA = {}
    zeta_DA = {}
    zeta_aux = {}
    gas_DA = {}
    compressor_DA = {}
    nom_entry_slack_DA = {}
    nom_entry_dev_pos = {}
    nom_entry_dev_neg = {}
    nom_exit_slack_DA = {}
    scenario_balance_TA = {}
    ub_pressure_violation_DA = {}
    lb_pressure_violation_DA = {}
    exit_viol = {}
    double_exit_viol = {}
    smoothed_special_pipe_flow_DA = {}
    ## Node variables
    var_mom_slack = {}
    var_cont_slack = {}
    var_resis_slack = {}
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
        vQp_vi[tstep] = m.addVars(co.pipes, lb=-GRB.INFINITY, name=f"vQp_vi_{tstep}")
        vQp_vi_rhs[tstep] = m.addVars(co.pipes, lb=-GRB.INFINITY, name=f"vQp_vi_rhs_{tstep}")
        vQp_vo[tstep] = m.addVars(co.pipes, lb=-GRB.INFINITY, name=f"vQp_vo_{tstep}")
        vQp_vo_rhs[tstep] = m.addVars(co.pipes, lb=-GRB.INFINITY, name=f"vQp_vo_rhs_{tstep}")
        vQp_zm[tstep] = m.addVars(co.pipes, lb=-GRB.INFINITY, name=f"vQp_zm_{tstep}")
        if tstep == 0:
            var_mom_slack = m.addVars(co.pipes, name=f"var_mom_slack_0")
            var_cont_slack = m.addVars(co.pipes, name=f"var_cont_slack_0")
        # ... and resistors
        vQr[tstep] = m.addVars(co.resistors, lb=-GRB.INFINITY, name=f"vQr_{tstep}") #:= vm(l,r) * var_non_pipe_Qo[l,r] * rho / 3.6;
        vQr_zm[tstep] = m.addVars(co.resistors, lb=-GRB.INFINITY, name=f"vQr_zm_{tstep}")
        if tstep == 0:
            var_resis_slack = m.addVars(co.resistors, name=f"var_resis_slack_0")

        ## Auxiliary variable pressure difference p_out minus p_in
        delta_p[tstep] = m.addVars(co.connections, lb=-Mp, ub=Mp, name=f"delta_p_{tstep}") #:= var_node_p[l] - var_node_p[r];

        ## Auxiliary variables to model the compressor calculation
        comp_phi_new[tstep] = m.addVars(co.compressors, name=f"comp_phi_new_{tstep}")
        comp_phi[tstep] = m.addVars(co.compressors, name=f"comp_phi_{tstep}")
        comp_phi_frac[tstep] = m.addVars(co.compressors, name=f"comp_phi_frac_{tstep}")
        comp_phi_aux[tstep] = m.addVars(co.compressors, name=f"comp_phi_aux_{tstep}")
        comp_i[tstep] = m.addVars(co.compressors, name=f"comp_i_{tstep}")
        comp_gas_aux[tstep] = m.addVars(co.compressors, name=f"comp_gas_aux_{tstep}")
        var_node_p_squ[tstep] = m.addVars(no.nodes, name=f"var_node_p_squ_{tstep}")

        ## Auxiliary variables to track dispatcher agent decisions
        va_DA[tstep] = m.addVars(co.valves, name=f"va_DA_{tstep}", vtype=GRB.BINARY)
        if tstep % config["nomination_freq"] == 0:
            zeta_aux[tstep] = {}
            for d in range(discretization + 1):
                zeta_aux[tstep][d] = m.addVars(co.resistors, name=f"zeta_aux_no{d}_{tstep}", vtype=GRB.BINARY)
        zeta_DA[tstep] = m.addVars(co.resistors, name=f"zeta_DA_{tstep}")
        gas_DA[tstep] = m.addVars(co.compressors, name=f"gas_DA_{tstep}")
        compressor_DA[tstep] = m.addVars(co.compressors, name=f"compressor_DA_{tstep}", vtype=GRB.BINARY)


        ## Auxiliary variable to track deviations from entry nominations ...
        nom_entry_slack_DA[tstep] = m.addVars(co.special, lb=-GRB.INFINITY, name=f"nom_entry_slack_DA_{tstep}")
        if tstep % config["nomination_freq"] == 0:
            nom_entry_dev_pos[tstep] = m.addVars(co.special, obj=1, name=f"nom_entry_dev_pos_{tstep}")
            nom_entry_dev_neg[tstep] = m.addVars(co.special, obj=1, name=f"nom_entry_dev_neg_{tstep}")
        # ... and from exit nominations
        nom_exit_slack_DA[tstep] = m.addVars(no.exits, lb=-GRB.INFINITY, name=f"nom_exit_slack_DA_{tstep}")

        ## Auxiliary variable to track balances
        scenario_balance_TA[tstep] = m.addVar(lb=-GRB.INFINITY, name=f"scenario_balance_TA_{tstep}")

        ## Auxiliary variable to track pressure violations at exits
        ub_pressure_violation_DA[tstep] = m.addVars(no.exits, lb=-GRB.INFINITY, name=f"ub_pressure_violation_DA_{tstep}")
        lb_pressure_violation_DA[tstep] = m.addVars(no.exits, lb=-GRB.INFINITY, name=f"lb_pressure_violation_DA_{tstep}")
        if tstep % config["nomination_freq"] == 0:
            exit_viol[tstep] = m.addVars(no.exits, obj=2/3, vtype=GRB.BINARY, name=f"exit_viol_{tstep}")
            double_exit_viol[tstep] = m.addVar(obj=-1/3, vtype=GRB.BINARY, name=f"double_exit_viol_{tstep}")

        ## Auxiliary variable to track smoothed flow over S-pipes
        smoothed_special_pipe_flow_DA[tstep] = m.addVars(co.special, lb=-GRB.INFINITY, name=f"smoothed_special_pipe_flow_DA_{tstep}")

    #### From here on the constraints have to be added. ####

    ### AUXILIARY CONSTRAINTS ###
    #
    m.addConstrs((var_node_p[0][n] == states[-1]["p"][n] for n in no.nodes), name='node_init')
    m.addConstrs((var_pipe_Qo_in[0][p] == states[-1]["q_in"][p] for p in co.pipes), name='pipes_in_init')
    m.addConstrs((var_pipe_Qo_out[0][p] == states[-1]["q_out"][p] for p in co.pipes), name='pipes_out_init')
    #m.addConstrs((var_non_pipe_Qo[0][np] == states[-1]["q"][np] for np in co.non_pipes), name='compressor_init')
    ## v * Q for pressure drop for pipes ...

    m.addConstrs((vQp[0][p] == (((Rs * Tm * zm(states[-2]["p"][p[0]],states[-2]["p"][p[1]]) / A(co.diameter[p]))
                                     * rho / 3.6 * states[-2]["q_in"][p] / (b2p * states[-2]["p"][p[0]]))
                                    * var_pipe_Qo_in[0][p] +
                                    ((Rs * Tm * zm(states[-2]["p"][p[0]],states[-2]["p"][p[1]]) / A(co.diameter[p]))
                                     * rho / 3.6 * states[-2]["q_out"][p] / (b2p * states[-2]["p"][p[1]]))
                                    * var_pipe_Qo_out[0][p])
                  * rho / 3.6 for p in co.pipes), name=f'vxQp_{0}')
    for tstep in range(numSteps):
        m.addConstrs((vQp_zm[tstep][p] == zm(var_node_p[tstep][p[0]],var_node_p[tstep][p[1]]) for p in co.pipes), name=f'vxQp_zm_{tstep}')
    for tstep in range(1, numSteps):
        m.addConstrs((vQp_vi_rhs[tstep][p] * ( b2p * var_node_p[tstep - 1][p[0]] ) == rho / 3.6 * var_pipe_Qo_in[tstep - 1][p] for p in co.pipes),
                     name=f'vxQp_vi_rhs_{tstep}')
        m.addConstrs((vQp_vo_rhs[tstep][p] * ( b2p * var_node_p[tstep - 1][p[1]] ) == rho / 3.6 * var_pipe_Qo_out[tstep - 1][p] for p in co.pipes),
                     name=f'vxQp_vo_rhs_{tstep}')
        m.addConstrs((vQp_vi[tstep][p] == (Rs * Tm * vQp_zm[tstep - 1][p] / A(co.diameter[p])) * vQp_vi_rhs[tstep][p] for p in co.pipes),
                     name=f'vxQp_vi_{tstep}')
        m.addConstrs((vQp_vo[tstep][p] == (Rs * Tm * vQp_zm[tstep - 1][p] / A(co.diameter[p])) * vQp_vo_rhs[tstep][p] for p in co.pipes),
                     name=f'vxQp_vo_{tstep}')
        m.addConstrs((vQp[tstep][p] == ( vQp_vi[tstep][p] * var_pipe_Qo_in[tstep][p] +
                                         vQp_vo[tstep][p] * var_pipe_Qo_out[tstep][p] )
                      * rho / 3.6 for p in co.pipes), name=f'vxQp_{tstep}')
        # original constraint:
        #vQp[p] == (vi(t, *p) * var_pipe_Qo_in[p] + vo(t, *p) * var_pipe_Qo_out[
        #    p]) * rho / 3.6
        # vi(t,i,o) = rtza(t,i,o) * rho / 3.6 * q_in_old(t,(i,o)) / ( b2p * p_old(t,i) )
        # vo(t,i,o) = rtza(t,i,o) * rho / 3.6 * q_out_old(t,(i,o)) / ( b2p * p_old(t,o) )
        # rtza(t,i,o) = Rs * Tm * zm(p_old(t,i),p_old(t,o)) / A(co.diameter[(i,o)])
        # (the obvious 'divided by two' is carried out in the function xip (in fuctions.py) according to eqn. 18 in the Station_Model_Paper.pdf (docs))
    # ... and resistors
    m.addConstrs((vQr[0][r] == (rho / 3.6 * ((Rs * Tm * zm(states[-2]["p"][r[0]],states[-2]["p"][r[1]]) / A(co.diameter[r]))))
                  * var_non_pipe_Qo[0][r] * rho / 3.6 for r in co.resistors), name=f'vxQr_{0}')
    for tstep in range(1, numSteps):
        m.addConstrs((vQr_zm[tstep][r] == zm(var_node_p[tstep - 1][r[0]],var_node_p[tstep - 1][r[1]]) for r in co.resistors), name=f'vxQr_zm_{tstep}')
        m.addConstrs((vQr[tstep][r] == (rho / 3.6 * ( (Rs * Tm * vQr_zm[tstep][r] / A(co.diameter[r]))))
                      * var_non_pipe_Qo[tstep][r] * rho / 3.6 for r in co.resistors), name=f'vxQr_{tstep}')
        # original constraint:
        # vQr[r] == vm(t,*r) * var_non_pipe_Qo[r] * rho / 3.6
        # vm(t,i,o) =  max(rho / 3.6 * ( rtza(t,i,o) * q_old(t,(i,o)) ) / 2 * 1 / b2p * ( 1 / p_old(t,i) + 1 / p_old(t,o) ), 2)
#
    ## constraints only to track dispatcher agent's decisions
    # m.addConstrs((va_DA[v] == get_agent_decision(agent_decisions["va"]["VA"][joiner(v)],t) for v in co.valves), name='va_mode')
    # m.addConstrs((zeta_DA[r] == get_agent_decision(agent_decisions["zeta"]["RE"][joiner(r)],t) for r in co.resistors), name='re_drag')
    # m.addConstrs((gas_DA[cs] == get_agent_decision(agent_decisions["gas"]["CS"][joiner(cs)],t) for cs in co.compressors), name='cs_fuel')
    # m.addConstrs((compressor_DA[cs] == get_agent_decision(agent_decisions["compressor"]["CS"][joiner(cs)],t) for cs in co.compressors), name='cs_mode')
    #
    ## constraints only to track smoothing of special pipe flows
    m.addConstrs((smoothed_special_pipe_flow_DA[0][s] == ( states[-2]["q_in"][s] + states[-2]["q_out"][s] + var_pipe_Qo_in[0][s] + var_pipe_Qo_out[0][s] ) / 4 for s in co.special), name=f'special_pipe_smoothing_{0}')
    for tstep in range(1, numSteps):
        m.addConstrs((smoothed_special_pipe_flow_DA[tstep][s] == ( var_pipe_Qo_in[tstep -1][s] + var_pipe_Qo_out[tstep - 1][s] + var_pipe_Qo_in[tstep][s] + var_pipe_Qo_out[tstep][s] ) / 4 for s in co.special), name=f'special_pipe_smoothing_{tstep}')
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
    m.addConstrs((b2p * (var_node_p[0][p[0]] + var_node_p[0][p[1]] - states[-2]["p"][p[0]] - states[-2]["p"][p[1]]) + rho / 3.6 * (2 * (Rs * Tm * vQp_zm[0][p] / A(co.diameter[p])) * dt) / co.length[p] * (var_pipe_Qo_out[0][p] - var_pipe_Qo_in[0][p]) <= var_cont_slack[p] for p in co.pipes), name=f'c_e_cons_pipe_continuity_<={0}')
    m.addConstrs((b2p * (var_node_p[0][p[0]] + var_node_p[0][p[1]] - states[-2]["p"][p[0]] - states[-2]["p"][p[1]]) + rho / 3.6 * (2 * (Rs * Tm * vQp_zm[0][p] / A(co.diameter[p])) * dt) / co.length[p] * (var_pipe_Qo_out[0][p] - var_pipe_Qo_in[0][p]) >= -var_cont_slack[p] for p in co.pipes), name=f'c_e_cons_pipe_continuity_>={0}')
    m.addConstrs(( b2p * delta_p[0][p] - xip(p) * vQp[0][p] <= var_mom_slack[p] for p in co.pipes), name=f'c_e_cons_pipe_momentum_<={0}')
    m.addConstrs(( b2p * delta_p[0][p] - xip(p) * vQp[0][p] >= -var_mom_slack[p] for p in co.pipes), name=f'c_e_cons_pipe_momentum_>={0}')
    for tstep in range(1, numSteps):
        m.addConstrs(( b2p * ( var_node_p[tstep][p[0]] + var_node_p[tstep][p[1]] - var_node_p[tstep - 1][p[0]] - var_node_p[tstep - 1][p[1]] ) + rho / 3.6 * ( 2 * (Rs * Tm * vQp_zm[tstep][p] / A(co.diameter[p])) * dt ) / co.length[p] * ( var_pipe_Qo_out[tstep][p] - var_pipe_Qo_in[tstep][p] ) == 0 for p in co.pipes), name=f'c_e_cons_pipe_continuity_{tstep}')
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
    m.addConstrs(( zeta_DA[0][r] == gp.quicksum([2**(d/3) * zeta_aux[0][d][r] for d in range(discretization + 1)]) for r in co.resistors), name=f'resistor_aux_0')
    m.addConstrs(( gp.quicksum([zeta_aux[0][d][r] for d in range(discretization + 1)]) == 1 for r in co.resistors), name=f'resistor_bins_0')
    m.addConstrs((b2p * delta_p[0][r] - xir(r, zeta_DA[0][r]) * vQr[0][r] <= var_resis_slack[r] for r in co.resistors), name=f'resistor_eq_<={0}')
    m.addConstrs((b2p * delta_p[0][r] - xir(r, zeta_DA[0][r]) * vQr[0][r] >= -var_resis_slack[r] for r in co.resistors), name=f'resistor_eq_>={0}')
    for tstep in range(1, numSteps):
        # Suggested by Klaus and inspired by section "2.3 Resistors" of https://opus4.kobv.de/opus4-zib/frontdoor/index/index/docId/7364.
        # We use resistors as control valves by controlling the resistors drag factor from outside
        #
        ## pressure drop equation (eqn. 21 Station_Model_Paper.pdf)
        # we use 2 ** (zeta/3) to map the "original" interval of relevant zeta values to the interaval [0,100]
        if tstep % config["nomination_freq"] == 0:
            m.addConstrs((zeta_DA[tstep][r] == gp.quicksum([2 ** (d / 3) * zeta_aux[tstep][d][r] for d in range(discretization + 1)]) for r in co.resistors),
                         name=f'resistor_aux_{tstep}')
            m.addConstrs((gp.quicksum([zeta_aux[tstep][d][r] for d in range(discretization + 1)]) == 1 for r in co.resistors),
                         name=f'resistor_bins_{tstep}')
        else:
            m.addConstrs((zeta_DA[tstep][r] == zeta_DA[tstep - 1][r] for r in co.resistors), name=f'resistor_continue_{tstep}')
        m.addConstrs(( b2p * delta_p[tstep][r] == xir(r, zeta_DA[tstep][r]) * vQr[tstep][r] for r in co.resistors), name=f'resistor_eq_{tstep}')
        #
        #
    ### CHECK VALVE MODEL ###
    for tstep in range(numSteps):
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
    m.addConstrs((var_non_pipe_Qo[0][cs] == 3.6 * states[-2]["p"][cs[0]] * comp_phi_new[0][cs] for cs in co.compressors), name=f'compressor_eq_two_{0}')
    m.addConstrs(( (compressors[joiner(cs)]["pi_1"] - compressors[joiner(cs)]["pi_2"]) / compressors[joiner(cs)]["phi_max"] * comp_phi_frac[0][cs] +
                   compressors[joiner(cs)]["pi_2"] * states[-2]["p"][cs[0]] - states[-2]["p"][cs[1]] + 1000 >= 1000 * compressor_DA[0][cs]
                   for cs in co.compressors), name=f"compressor_eq_three_0")
    m.addConstrs((comp_phi_frac[0][cs] * comp_phi[0][cs] == states[-2]["p"][cs[0]] for cs in co.compressors), name=f"compressor_eq_four_0")
    m.addConstrs((comp_phi_new[0][cs] == comp_phi[0][cs] * compressor_DA[0][cs] for cs in co.compressors), name=f"compressor_eq_five_0")
    m.addConstrs((comp_phi[0][cs] == gp.min_(comp_phi_aux[0][cs], compressors[joiner(cs)]["phi_max"]) for cs in co.compressors), name=f"compressor_eq_six_0")
    m.addConstrs((comp_phi_aux[0][cs] == gp.max_(comp_i[0][cs], compressors[joiner(cs)]["phi_min"]) for cs in co.compressors), name=f"compressor_eq_seven_0")
    # pressure squared equation for further time steps
    m.addConstrs((comp_gas_aux[0][cs] == gas_DA[0][cs] * states[-2]["p"][cs[0]] for cs in co.compressors), name=f"compressor_eq_nine_0")
    m.addConstrs((comp_i[0][cs] == compressors[joiner(cs)]["phi_MIN"]/((-compressors[joiner(cs)]["p_in_max"] + compressors[joiner(cs)]["p_in_min"]) * compressors[joiner(cs)]["pi_MIN"]) *
                  (compressors[joiner(cs)]["pi_MAX"] * states[-2]["p"][cs[0]]**2 * comp_gas_aux[0][cs] -
                   compressors[joiner(cs)]["pi_MAX"] * compressors[joiner(cs)]["eta"] * states[-2]["p"][cs[0]]**2 * comp_gas_aux[0][cs] -
                   compressors[joiner(cs)]["pi_MAX"] * compressors[joiner(cs)]["p_in_max"] * comp_gas_aux[0][cs] * states[-2]["p"][cs[0]] -
                   compressors[joiner(cs)]["pi_MIN"] * compressors[joiner(cs)]["p_in_max"] * states[-2]["p"][cs[0]]**2 +
                   compressors[joiner(cs)]["pi_MIN"] * compressors[joiner(cs)]["p_in_max"] * comp_gas_aux[0][cs] * states[-2]["p"][cs[0]] +
                   compressors[joiner(cs)]["pi_MAX"] * compressors[joiner(cs)]["p_in_min"] * compressors[joiner(cs)]["eta"] * comp_gas_aux[0][cs] * states[-2]["p"][cs[0]] +
                   compressors[joiner(cs)]["pi_MIN"] * compressors[joiner(cs)]["p_in_min"] * states[-2]["p"][cs[0]]**2 -
                   compressors[joiner(cs)]["pi_MIN"] * compressors[joiner(cs)]["p_in_min"] * comp_gas_aux[0][cs] * states[-2]["p"][cs[0]] +
                   compressors[joiner(cs)]["p_in_max"] * states[-2]["p"][cs[0]] * states[-2]["p"][cs[1]] -
                   compressors[joiner(cs)]["p_in_min"] * states[-2]["p"][cs[0]] * states[-2]["p"][cs[1]])
                  for cs in co.compressors), f"compressor_eq_ten_0")
    # phi_new(compressor_DA[0][cs],compressors[joiner(cs)]["phi_min"],compressors[joiner(cs)]["phi_max"],compressors[joiner(cs)]["pi_1"],compressors[joiner(cs)]["pi_2"],compressors[joiner(cs)]["pi_MIN"],compressors[joiner(cs)]["phi_MIN"],compressors[joiner(cs)]["p_in_min"],compressors[joiner(cs)]["p_in_max"],compressors[joiner(cs)]["pi_MAX"],compressors[joiner(cs)]["eta"], gas_DA[0][cs],states[-2]["p"][cs[0]],states[-2]["p"][cs[1]])
    for tstep in range(1, numSteps):
        m.addConstrs((var_non_pipe_Qo[tstep][cs] >= 0 for cs in co.compressors), name=f'compressor_eq_one_{tstep}')
        #m.addConstrs((var_non_pipe_Qo[tstep][cs] == compressor_DA[tstep][cs] * 3.6 * var_node_p[tstep - 1][cs[0]] * phi_new(compressor_DA[tstep][cs],compressors[joiner(cs)]["phi_min"],compressors[joiner(cs)]["phi_max"],compressors[joiner(cs)]["pi_1"],compressors[joiner(cs)]["pi_2"],compressors[joiner(cs)]["pi_MIN"],compressors[joiner(cs)]["phi_MIN"],compressors[joiner(cs)]["p_in_min"],compressors[joiner(cs)]["p_in_max"],compressors[joiner(cs)]["pi_MAX"],compressors[joiner(cs)]["eta"],gas_DA[tstep][cs],var_node_p[tstep - 1][cs[0]],var_node_p[tstep - 1][cs[1]]) for cs in co.compressors), name=f'compressor_eq_two_{tstep}')
        m.addConstrs((var_non_pipe_Qo[0][cs] == 3.6 * var_node_p[tstep -1][cs[0]] * comp_phi_new[tstep][cs] for cs in co.compressors),
                     name=f'compressor_eq_two_{tstep}')
        m.addConstrs(((compressors[joiner(cs)]["pi_1"] - compressors[joiner(cs)]["pi_2"]) / compressors[joiner(cs)]["phi_max"] * comp_phi_frac[tstep][cs] +
                      compressors[joiner(cs)]["pi_2"] * var_node_p[tstep - 1][cs[0]] - var_node_p[tstep - 1][cs[1]] + 1000 >= 1000 * compressor_DA[tstep][cs]
                      for cs in co.compressors), name=f"compressor_eq_three_{0}")
        m.addConstrs((comp_phi_frac[tstep][cs] * comp_phi[tstep][cs] == var_node_p[tstep - 1][cs[0]] for cs in co.compressors), name=f"compressor_eq_four_{tstep}")
        m.addConstrs((comp_phi_new[tstep][cs] == comp_phi[tstep][cs] * compressor_DA[tstep][cs] for cs in co.compressors), name=f"compressor_eq_five_{tstep}")
        m.addConstrs((comp_phi[tstep][cs] == gp.min_(comp_phi_aux[tstep][cs], compressors[joiner(cs)]["phi_max"]) for cs in co.compressors), name=f"compressor_eq_six_{tstep}")
        m.addConstrs((comp_phi_aux[tstep][cs] == gp.max_(comp_i[tstep][cs], compressors[joiner(cs)]["phi_min"]) for cs in co.compressors), name=f"compressor_eq_seven_{tstep}")
        m.addConstrs((var_node_p_squ[tstep - 1][cs[0]] == var_node_p[tstep - 1][cs[0]]**2 for cs in co.compressors), name=f"compressor_eq_eight_{tstep}")
        m.addConstrs((comp_gas_aux[tstep][cs] == gas_DA[tstep][cs] * var_node_p[tstep - 1][cs[0]] for cs in co.compressors), name=f"compressor_eq_nine_{tstep}")
        m.addConstrs((comp_i[tstep][cs] == compressors[joiner(cs)]["phi_MIN"] / (-compressors[joiner(cs)]["p_in_max"] + compressors[joiner(cs)]["p_in_min"]) * compressors[joiner(cs)]["pi_MIN"] *
                      (compressors[joiner(cs)]["pi_MAX"] * var_node_p_squ[tstep - 1][cs[0]] * comp_gas_aux[tstep][cs] -
                       compressors[joiner(cs)]["pi_MAX"] * compressors[joiner(cs)]["eta"] * var_node_p_squ[tstep - 1][cs[0]] * comp_gas_aux[tstep][cs] -
                       compressors[joiner(cs)]["pi_MAX"] * compressors[joiner(cs)]["p_in_max"] * comp_gas_aux[tstep][cs] * var_node_p[tstep - 1][cs[0]] -
                       compressors[joiner(cs)]["pi_MIN"] * compressors[joiner(cs)]["p_in_max"] * var_node_p_squ[tstep - 1][cs[0]] +
                       compressors[joiner(cs)]["pi_MIN"] * compressors[joiner(cs)]["p_in_max"] * comp_gas_aux[tstep][cs] * var_node_p[tstep - 1][cs[0]] +
                       compressors[joiner(cs)]["pi_MAX"] * compressors[joiner(cs)]["p_in_min"] * compressors[joiner(cs)]["eta"] * comp_gas_aux[tstep][cs] * var_node_p[tstep - 1][cs[0]] +
                       compressors[joiner(cs)]["pi_MIN"] * compressors[joiner(cs)]["p_in_min"] * var_node_p_squ[tstep - 1][cs[0]] -
                       compressors[joiner(cs)]["pi_MIN"] * compressors[joiner(cs)]["p_in_min"] * comp_gas_aux[tstep][cs] * var_node_p[tstep - 1][cs[0]] +
                       compressors[joiner(cs)]["p_in_max"] * var_node_p[tstep - 1][cs[0]] * var_node_p[tstep - 1][cs[1]] -
                       compressors[joiner(cs)]["p_in_min"] * var_node_p[tstep - 1][cs[0]] * var_node_p[tstep - 1][cs[1]])
                      for cs in co.compressors), f"compressor_eq_ten_{tstep}")
    #
    ### ENTRY MODEL ###
    for tstep in range(numSteps):
        # Suggested by Klaus and described in gasnet_control/docs/urmel-entry.pdf
        #
        m.addConstrs((var_node_Qo_in[tstep][e] <= no.entry_flow_bound[e] for e in no.entries), name=f'entry_flow_model_{tstep}')
        m.addConstrs((var_pipe_Qo_out[tstep][s] + nom_entry_slack_DA[tstep][s] == get_agent_decision(agent_decisions["entry_nom"]["S"][joiner(s)],(tstep//config['nomination_freq']-2)*config['nomination_freq']) for s in co.special), name=f'nomination_check_{tstep}')
        if tstep % config["nomination_freq"] == 0:
            m.addConstrs((nom_entry_dev_pos[tstep][s] >= 1/4400 * gp.quicksum([nom_entry_slack_DA[k][s] for k in range(tstep, tstep + config["nomination_freq"])]) for s in co.special), name=f"nom_pos_deviation_{tstep}")
            m.addConstrs((-nom_entry_dev_neg[tstep][s] <= 1/4400 * gp.quicksum([nom_entry_slack_DA[k][s] for k in range(tstep, tstep + config["nomination_freq"])]) for s in co.special), name=f"nom_neg_deviation_{tstep}")

    #m.addConstrs((var_pipe_Qo_out[s] + nom_entry_slack_DA[s] == get_agent_decision(agent_decisions["entry_nom"]["S"][joiner(s)],t//config['nomination_freq']) for s in co.special), name='nomination_check')
    #print("Exit nominations: {}".format([get_agent_decision(agent_decisions["exit_nom"]["X"][x],t) for x in no.exits]))
    #print("Entry nominations decision for this step ({}) was taken in step {}: ".format(t, (t//config['nomination_freq']-2)*config['nomination_freq']), [get_agent_decision(agent_decisions["entry_nom"]["S"][joiner(s)],(t//config['nomination_freq']-2)*config['nomination_freq']) for s in co.special])
    #print(agent_decisions["entry_nom"]["S"])
    #
    #
    ### TRACKING OF RELEVANT VALUES ###
    #
        m.addConstr((sum([get_agent_decision(agent_decisions["entry_nom"]["S"][joiner(s)],(tstep//config['nomination_freq']-2)*config['nomination_freq']) for s in co.special]) + sum([get_agent_decision(agent_decisions["exit_nom"]["X"][x],tstep) for x in no.exits]) == scenario_balance_TA[tstep]), f'track_scenario_balance{tstep}')
        #m.addConstr((sum([get_agent_decision(agent_decisions["entry_nom"]["S"][joiner(s)],t//config['nomination_freq']) for s in co.special]) + sum([get_agent_decision(agent_decisions["exit_nom"]["X"][x],t) for x in no.exits]) == scenario_balance_TA), 'track_scenario_balance')
        m.addConstrs((nom_exit_slack_DA[tstep][x] == var_boundary_node_flow_slack_positive[tstep][x] - var_boundary_node_flow_slack_negative[tstep][x] for x in no.exits), name=f'track_exit_nomination_slack_{tstep}')
        m.addConstrs((var_node_p[tstep][n] - no.pressure_limits_upper[n] == ub_pressure_violation_DA[tstep][n] for n in no.exits), name=f'track_ub_pressure_violation_{tstep}')
        m.addConstrs((no.pressure_limits_lower[n] - var_node_p[tstep][n] == lb_pressure_violation_DA[tstep][n] for n in no.exits), name=f'track_lb_pressure_violation_{tstep}')
        m.addConstrs((exit_viol[config["nomination_freq"] * (tstep//config["nomination_freq"])][n] >= ub_pressure_violation_DA[tstep][n] for n in no.exits), name=f"penalize_ub_pressure_viol_{tstep}")
        m.addConstrs((exit_viol[config["nomination_freq"] * (tstep//config["nomination_freq"])][n] >= lb_pressure_violation_DA[tstep][n] for n in no.exits), name=f"penalize_lb_pressure_viol_{tstep}")
        if tstep % config["nomination_freq"] == 0:
            m.addConstr((double_exit_viol[tstep] <= 1/2 * gp.quicksum([exit_viol[tstep][n] for n in no.exits])), f"double_penalize_pressure_viol_{tstep}")
    #
    #
    ### TESTS ###
    #m.addConstr( var_node_p['START_ND'] == 43.5, 'set_p_ND')

    return m

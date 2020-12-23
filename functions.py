# -*- coding: utf-8 -*-

# this file contains helper functions to model gas physics

import importlib
import sys
import re
wd = sys.argv[1].replace("/",".")
wd = re.sub(r'\.$', '', wd)

sc = importlib.import_module(wd + ".init_scenario")
no = importlib.import_module(wd + ".nodes")
co = importlib.import_module(wd + ".connections")

from constants import *
from math import *

def get_init_scenario():
    states = {}
    states[-2] = {'p': {}, 'q': {}, 'q_in': {}, 'q_out': {}}
    states[-1] = {'p': {}, 'q': {}, 'q_in': {}, 'q_out': {}}
    for node in no.nodes:
        states[-2]["p"][node]     = sc.var_node_p_old_old[node]
        states[-1]["p"][node]     = sc.var_node_p_old[node]
    for non_pipe in co.non_pipes:
        states[-2]["q"][non_pipe] = sc.var_non_pipe_Qo_old_old[non_pipe]
        states[-1]["q"][non_pipe] = sc.var_non_pipe_Qo_old[non_pipe]
    for pipe in co.pipes:
        states[-2]["q_in"][pipe]  = sc.var_pipe_Qo_in_old_old[pipe]
        states[-2]["q_out"][pipe] = sc.var_pipe_Qo_out_old_old[pipe]
        states[-1]["q_in"][pipe]  = sc.var_pipe_Qo_in_old[pipe]
        states[-1]["q_out"][pipe] = sc.var_pipe_Qo_out_old[pipe]
    return states

states = get_init_scenario()
print(states)

# Mean values are used to stabilize simulation
def stabilizer(a, b):
    if a * b > 1:
        if a > 0:
            return sqrt(a * b)
        else:
            return -sqrt(a * b)
    else:
        return ( a + b ) / 2

# pressure stabilizer
def p_old(i,n):
    return stabilizer(states[i-1]["p"][n], states[i-2]["p"][n])

# flow stabilizer (non-pipes)
def q_old(i,non_pipe):
    return stabilizer(abs(states[i-1]["q"][non_pipe]), abs(states[i-2]["q"][non_pipe]))
    
# pipe inflow stabilizer
def q_in_old(i,pipe):
    return stabilizer(abs(states[i-1]["q_in"][pipe]), abs(states[i-2]["q_in"][pipe]))
    
# pipe outflow stabilizer
def q_out_old(i,pipe):
    return stabilizer(abs(states[i-1]["q_out"][pipe]), abs(states[i-2]["q_out"][pipe]))

#for node in no.nodes:
#    print(node)
#    print(p_old(0,node))
#for non_pipe in co.non_pipes:
#    print(non_pipe)
#    print(q_old(0,non_pipe))
#for pipe in co.pipes:
#    print(pipe)
#    print(q_in_old(0,pipe))
#for pipe in co.pipes:
#    print(pipe)
#    print(q_out_old(0,pipe))

# reduced pressure (2.4 Forne)
def pr(p):
    return p / pc
    
# reduced Temperature (2.4 Forne)
def Tr(T):
    return T / Tc
    
# cross sectional area of a pipe in m²
def A(diam):
    return Pi * ( diam / 2 ) ** 2
    
# compressibility factor Papay (2.4 Forne) # Felix: "falsch im Forne-Buch; 0.274, nicht 0.247 muss es sein!"
def z(p,T):
    return 1 - 3.52 * pr(p) * exp(-2.26 * Tr(T)) + 0.274 * pr(p) ** 2 * exp(-1.878 * Tr(T))
    
# compressibility factor Papay (2.4 Forne)
def zm(pi,po):
    return ( z(pi,Tm) + z(po,Tm) ) / 2
    
# Nikuradze (2.19 Forne), diameter diam in m, integral roughness rough in m
def lamb(diam, rough):
    return ( 2 * log(diam/rough,10) + 1.138 ) ** -2
    
# Rs * Tm * zm / A
def rtza(t,i,o):
    #print("rtza(%s,%s) = %f" %(i,o,Rs * Tm * zm(p_old(i),p_old(o)) / A(co.diameter[(i,o)])))
    return Rs * Tm * zm(p_old(t,i),p_old(t,o)) / A(co.diameter[(i,o)])
    
# Inflow velocity
def vi(t,i,o):
    return rtza(t,i,o) * rho / 3.6 * q_in_old(t,(i,o)) / ( b2p * p_old(t,i) )
    
# Outflow velocity
def vo(t,i,o):
    return rtza(t,i,o) * rho / 3.6 * q_out_old(t,(i,o)) / ( b2p * p_old(t,o) )

## Function for resistor model
def vm(t,i,o):
    vm =  rho / 3.6 * ( rtza(t,i,o) * q_old(t,(i,o)) ) / 2 * 1 / b2p * ( 1 / p_old(t,i) + 1 / p_old(t,o) )
    vmm = max(vm, 2)
    #print("i: %s, o: %s, vm: %f, vmm: %f" % (i,o,vm,vmm))
    return vmm

# Functions for compressor model
#
def L_min(L_min_pi,L_min_phi,phi):
    return - L_min_pi / L_min_phi * phi + L_min_pi

# Achsenabschnitt der Maximalleistung als Funktion vom Eingangsdruck. Quasi "Maximale Max-Leistung"
def L_max_axis_intercept(L_max_pi,L_eta,p_i_min,p_i_max,p_in):
    return L_max_pi * ( L_eta - 1 ) / ( p_i_max - p_i_min ) * ( p_in - p_i_min ) + L_max_pi

# Maximalleistung als Funktion vom Fluss unter Verwendung des Achsenabschnitts. Die Steigung erhalten wir aus der Parametrierung der Minimalleistung.
def L_max(L_min_pi,L_min_phi,L_max_pi,L_eta,p_i_min,p_i_max,phi,p_in):
    return - L_min_pi / L_min_phi * phi + L_max_axis_intercept(L_max_pi,L_eta,p_i_min,p_i_max,p_in)

# Leistung als Funktion des "Gaspedals" zwischen 0% und 100%
def L_gas(L_min_pi,L_min_phi,phi,gas,L_max_pi,eta,p_i_min,p_i_max,p_in):
    return ( 1 - gas ) * L_min(L_min_pi,L_min_phi,phi) + gas * L_max(L_min_pi,L_min_phi,L_max_pi,L_eta,p_i_min,p_i_max,phi,p_in)

# pi_2: Geradengleichung mit den Punkten (0,pi_2) und (phi_max,pi_1)
def ulim(phi,phi_max,pi_1,pi_2):
    return ( pi_1 - pi_2 ) / phi_max * phi + pi_2

# Berechnung der phi-Koordinate des Schnittpunkts zwischen der Druckverhältnisgeraden (=p_out/p_in) und L_gas
def intercept(L_min_pi,L_min_phi,p_i_min,p_i_max,L_max_pi,L_eta,gas,p_in,p_out):
    return (L_min_phi * (gas * L_max_pi * p_in ** 2 - gas * L_eta * L_max_pi * p_in ** 2 - 
   gas * L_max_pi * p_in * p_i_max - L_min_pi * p_in * p_i_max + 
   gas * L_min_pi * p_in * p_i_max + gas * L_eta * L_max_pi * p_in * p_i_min + 
   L_min_pi * p_in * p_i_min - gas * L_min_pi * p_in * p_i_min + p_i_max * p_out -
    p_i_min * p_out))/(L_min_pi * p_in * (-p_i_max + p_i_min))

# Berechnung des neuen phi (Prüfung, ob im Kennfeld in phi_new)
def phi_new_tmp(compressor,phi_min,phi_max,pi_1,pi_2,L_min_pi,Lmaxpi,L_min_phi,p_i_min,p_i_max,L_max_pi,eta,gas,p_in,p_out):
    return min(max(intercept(L_min_pi,L_min_phi,p_i_min,p_i_max,L_max_pi,eta,gas,p_in,p_out),phi_min),phi_max)

# Prüfung, ob (phi_new_tmp,p_out/p_in) im Kennfeld; falls ja, so ist es das finale phi
def phi_new(compressor,phi_min,phi_max,pi_1,pi_2,L_min_pi,Lmaxpi,L_min_phi,p_i_min,p_i_max,L_max_pi,eta,gas,p_in,p_out):
    phi = phi_new_tmp(compressor,phi_min,phi_max,pi_1,pi_2,L_min_pi,Lmaxpi,L_min_phi,p_i_min,p_i_max,L_max_pi,eta,gas,p_in,p_out)
    if compressor == 1 and ulim(phi,phi_max,pi_1,pi_2) >= p_out/p_in:
        return phi
    else:
        return 0

# factors for pressure drop for pipes ...
def xip(i):
    #print("length[%s,%s]" % (i,co.length[i]))
    #print("xip[%s,%s]" % (i,lamb(co.diameter[i], co.roughness[i]) * co.length[i] / ( 4 * co.diameter[i] * A(co.diameter[i]) )))
    #print("lamb[%s,%s]" % (i,lamb(co.diameter[i], co.roughness[i])))
    #print("rough[%s,%s])" % (i,co.roughness[i]))
    return lamb(co.diameter[i], co.roughness[i]) * co.length[i] / ( 4 * co.diameter[i] * A(co.diameter[i]) )

# ... and resistors
def xir(i,zeta):
    return zeta / ( 2 * A(co.diameter[i]) );

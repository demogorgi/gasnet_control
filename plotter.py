from urmel import *

def plot(step, _step, agent_decisions, compressors, output):
    for k in compressors:
        cs = compressors[k]
        compressor = get_agent_decision(agent_decisions["compressor"]["CS"][k],step)
        _from, _to = k.split("^")
        gas = get_agent_decision(agent_decisions["gas"]["CS"][k],step) 
        pi_MIN = cs["pi_MIN"]
        phi_MIN = cs["phi_MIN"]
        pi_MAX = cs["pi_MAX"]
        eta = cs["eta"]
        p_in_min = cs["p_in_min"]
        p_in_max = cs["p_in_max"]
        pi_min = cs["pi_min"]
        pi_1 = cs["pi_1"]
        pi_2 = cs["pi_2"]
        phi_max = cs["phi_max"]
        phi_min = cs["phi_min"]

        # max(pi_min,L_min(phi_min))
        p1y = max( pi_min, - ( pi_MIN / phi_MIN ) * phi_min + pi_MIN )
        # ulim(phi_min)
        p2y = ( pi_1 - pi_2 ) / phi_max * phi_min + pi_2
        # ulim(phi_max)
        p3y = ( pi_1 - pi_2 ) / phi_max * phi_max + pi_2
        # interception point ulim and Lmax
        p4x = (phi_MIN * phi_max * (pi_MAX * p_in_max - pi_2 * p_in_max - eta * pi_MAX * p_in_min + pi_2 * p_in_min - pi_MAX * p_old(step,_from) + eta * pi_MAX * p_old(step,_from)))/((pi_MIN * phi_max + phi_MIN * pi_1 - phi_MIN * pi_2) * (p_in_max - p_in_min))
        # min(phi_max,interception point min_pi and Lmax)
        p5x = min( phi_max, ( L_max_axis_intercept(pi_MAX,eta,p_in_min,p_in_max,p_old(step,_from)) - pi_min ) / ( pi_MIN / phi_MIN ) )
        # interception point L_min and p_min
        p6x = max( phi_min, ( pi_MIN - pi_min ) / ( pi_MIN / phi_MIN ) )
        # ulim(p4x)
        p4y =  ( pi_1 - pi_2 ) / phi_max * p4x + pi_2
        # min(p3y,max(pi_MIN, L_max(phi_max)))
        p5y = min( p3y ,max(pi_min, - ( pi_MIN / phi_MIN ) * phi_max +  L_max_axis_intercept(pi_MAX,eta,p_in_min,p_in_max,p_old(step,_from))) )

        cmd = ";".join([
"gnuplot -e \"set term pdfcairo enhanced font 'Calibri Light, 10'",
"set output '%s/CS_%s_%s%s.pdf'" % (output, _from, _to, _step),
# title
"set title '{/:Bold Verdichter %s -> %s (%s)}'" % (_from, _to, _step.replace("_","").replace("_"," \/ ")),
# labels
"set xlabel 'Fluss {/Symbol f}/m^3/s'",
"set ylabel 'Druckverh\344ltnis {/Symbol p}/1'",

# Wheel map polygon
#"set label at %f, %f 'a' point pointtype 7 pointsize 0.4" % (phi_min,p1y),
#"set label at %f, %f 'b' point pointtype 7 pointsize 0.4" % (phi_min,p2y),
#"set label at %f, %f 'c' point pointtype 7 pointsize 0.4" % (p4x,p4y),
#"set label at %f, %f 'd' point pointtype 7 pointsize 0.4" % (p5x,p5y),
#"set label at %f, %f 'e' point pointtype 7 pointsize 0.4" % (p5x,pi_min),
#"set label at %f, %f 'f' point pointtype 7 pointsize 0.4" % (p6x,pi_min),
"set object 1 polygon from %f,%f to %f,%f to %f,%f to %f,%f to %f,%f to %f,%f to %f,%f fillstyle transparent solid 0.3" % (
                          #  a        b        c        d        e       f       g
    # a (starting upper left corner then clockwise)
    phi_min,p1y,
    # b
    phi_min,p2y,
    # c intersection point ulim and Lmax
    p4x,p4y,
    # d
    p5x,p5y,
    # e
    p5x,pi_min,
    # f
    p6x,pi_min,
    # g
    phi_min,p1y
    ),

# LINES
"plot [0:%f] %s" % (cs["phi_max"] + 1, " ".join([

    # l min line
    "[0:%f]" % (cs["pi_MAX"] + 0.5),
    "- %f / %f * x + %f title 'L_{min}' lt 1 lw 2, " % (
        pi_MIN,
        phi_MIN,
        pi_MIN
    ),

    # l max line
    "- %f / %f * x + %f title 'L_{max}' lt 1 lw 2, " % (
        pi_MIN,
        phi_MIN,
        L_max_axis_intercept(
            pi_MAX,
            eta,
            p_in_min,
            p_in_max,
            p_old(step,_from)
        )
    ),

    # l gas line
    "(1 - %f) * ((-%f / %f) * x + %f) + %f * ((-%f / %f) * x + %f) dashtype 4 lt 3 title 'L_{gas}', " % (
        gas,
        pi_MIN,
        phi_MIN,
        pi_MIN,
        gas,
        pi_MIN,
        phi_MIN,
        L_max_axis_intercept(
            pi_MAX,
            eta,
            p_in_min,
            p_in_max,
            p_old(step,_from)
        )
    ),

    # pi 1 line
    "%f dashtype 3 lt 6 title '{/Symbol p}_1', " % (pi_1),

    # L_max_max line
    "(-%f / %f) * x + %f dashtype 3 lt 1 lw 1 title 'L_{MAX}', " % (
        pi_MIN,
        phi_MIN,
        L_max_axis_intercept(
            pi_MAX,
            eta,
            p_in_min,
            p_in_max,
            p_in_min
        )
    ),

    # ulim line
    "(%f - %f) / %f * x + %f lt 6 lw 2 title 'ulim', " % (
        pi_1,
        pi_2,
        phi_max,
        pi_2
    ),

    # (old) pressure_to / pressure_from line
    "(%f / %f) dashtype 5 lt 3 title 'p_{out} / p_{in}'" % (
        p_old(step,_to),
        p_old(step,_from)
    ),
])),
# phi_min line
"set arrow from %f,0 to %f,%f nohead lw 2 lc rgb '#ff00ff' " % (
    phi_min,
    phi_min,
    pi_2
),

# phi_max line
"set arrow from %f,0 to %f,(%f+%f)/2 nohead lw 2 lc rgb '#ff00ff'" % (
    phi_max,
    phi_max,
    pi_1,
    pi_2
),

# TICKS
# add L_max_axis_intercept value as tic
"set ytics add('L_{max\\_axis\\_int}(%s))     ' %f) " % (
    str(round(p_old(step,_from), 1)),
    L_max_axis_intercept(
        pi_MAX,
        eta,
        p_in_min,
        p_in_max,
        p_old(step,_from)
    )
),

# add pi_2 value as a tic
"set ytics add ('{/Symbol p}_{2}     ' %f) " % pi_2,

# add pi_1 value as a tic
"set ytics add ('{/Symbol p}_{1}     ' %f) " % pi_1,

# add pi_MIN value as a tic
"set ytics add ('{/Symbol p}_{min}' %f)" % pi_min,

# add phi_min value as tic
"set xtics add ('\n{/Symbol f}_{min}' %f)" % phi_min,

# add phi_max value as a tic
"set xtics add ('\n{/Symbol f}_{max}' %f)" % phi_max,

# add L_phi_min value as a tic
"set xtics add ('\n\nL_{/Symbol f}_{,min}' %f)" % phi_MIN,

# POINTS
# add interception point
"set label at %f, %f '' point pointtype 7 pointsize 1" % (
  phi_new(
      compressor,
      phi_min,
      phi_max,
      pi_1,
      pi_2,
      pi_MIN,
      pi_MAX,
      phi_MIN,
      p_in_min,
      p_in_max,
      pi_MAX,
      eta,
      gas,
      p_old(step,_from),
      p_old(step,_to)
      ),
  0 if phi_new(compressor,phi_min,phi_max,pi_1,pi_2,pi_MIN,pi_MAX,phi_MIN,p_in_min,p_in_max,pi_MAX,eta,gas,p_old(step,_from),p_old(step,_to)) == 0 else p_old(step,_to) / p_old(step,_from)
  ),
  #if phi_new(phi_min,phi_max,pi_1,pi_2,pi_MIN,pi_MAX,phi_MIN,p_in_min,p_in_max,pi_MAX,eta,gas,p_old(step,l),p_old(step,r)) == 0:
  #    0
  #else:
  #    p_old(step,r) / p_old(step,l)
  #),

#    intercept(
#        pi_MIN,
#        phi_MIN,
#        p_in_min,
#        p_in_max,
#        pi_MAX,
#        eta,
#        gas,
#        p_old(step,_from),
#        p_old(step,_to)
#    ),
#    (p_old(step,_to) / p_old(step,_from))
#),

# FINILIZE
"set output '%s/CS_%s_%s%s.pdf'" % (output, _from, _to, _step),
"replot; \""
        ])

        return cmd

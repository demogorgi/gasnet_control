from params import *
from urmel import get_agent_decision, p_old, L_max_axis_intercept
from main import agent_decisions


import matplotlib.pyplot as plt

for k in compressors:
    cs = compressors[k]

compressor = get_agent_decision(agent_decisions["compressor"]["CS"][k], 0)
_from, _to = k.split("^")
gas = get_agent_decision(agent_decisions["gas"]["CS"][k], 0)
pi_MIN = 2#cs["pi_MIN"]
phi_MIN = cs["phi_MIN"]
pi_MAX = cs["pi_MAX"]
eta = cs["eta"]
p_in_min = cs["p_in_min"]
p_in_max = cs["p_in_max"]
pi_min = cs["pi_min"]
pi_1 = cs["pi_1"]
pi_2 = cs["pi_2"]
phi_max = 6.5#cs["phi_max"]
phi_min = cs["phi_min"]

# max(pi_min,L_min(phi_min))
p1y = max(pi_min, - (pi_MIN / phi_MIN) * phi_min + pi_MIN)
# U(phi_min)
p2y = (pi_1 - pi_2) / phi_max * phi_min + pi_2
# U(phi_max)
p3y = (pi_1 - pi_2) / phi_max * phi_max + pi_2
# interception point U and Lmax
p4x = (phi_MIN * phi_max * (
            pi_MAX * p_in_max - pi_2 * p_in_max - eta * pi_MAX * p_in_min + pi_2 * p_in_min - pi_MAX * p_old(
        0, _from) + eta * pi_MAX * p_old(0, _from))) / (
                  (pi_MIN * phi_max + phi_MIN * pi_1 - phi_MIN * pi_2) * (
                      p_in_max - p_in_min))
# min(phi_max,interception point min_pi and Lmax)
p5x = min(phi_max, (L_max_axis_intercept(pi_MAX, eta, p_in_min, p_in_max,
                                         p_old(0, _from)) - pi_min) / (
                      pi_MIN / phi_MIN))
# interception point L_min and p_min
p6x = max(phi_min, (pi_MIN - pi_min) / (pi_MIN / phi_MIN))
# U(p4x)
p4y = (pi_1 - pi_2) / phi_max * p4x + pi_2
# min(p3y,max(pi_MIN, L_max(phi_max)))
p5y = min(p3y, max(pi_min,
                   - (pi_MIN / phi_MIN) * phi_max + L_max_axis_intercept(
                       pi_MAX, eta, p_in_min, p_in_max, p_old(0, _from))))


# setup first plot
plt.figure(figsize=(8, 5))
plt.xlabel(r"volume flow $\phi$ in $\frac{m^3}{s}$")
plt.xlim(0, 7.5)
plt.ylabel(r"pressure ratio $\frac{\pi}{1}$")
plt.ylim(0, 6.5)
plt.tick_params(labeltop=True, labelright=True, top=True, right=True)

# plot the first bounds
phi_min_values = [[phi_min, phi_min], [0, 6.5]]
phi_max_values = [[phi_max, phi_max], [0, 6.5]]
pi_min_values = [[0, 7.5], [pi_min, pi_min]]

x_tick_labels = ['0', '', r'$\phi_\min$','2', '3', '4', r'$\phi_\max$', '5', '6', '7']
x_ticks = [0, 1, phi_min, 2, 3, 4, phi_max, 5, 6, 7]
y_tick_labels = ['0', r'$\pi_\min$', '2', '3', '4', '5', '6']
y_ticks = [0, pi_min, 2, 3, 4, 5, 6]

plt.plot(phi_min_values[0], phi_min_values[1], 'blue')
plt.plot(phi_max_values[0], phi_max_values[1], 'blue')
plt.plot(pi_min_values[0], pi_min_values[1], 'blue')
#plt.annotate(text=r'$\phi_\min$', xy=(phi_min, 0))
plt.xticks(ticks=x_ticks, labels=x_tick_labels)
plt.yticks(ticks=y_ticks, labels=y_tick_labels)
plt.fill_between([phi_min, phi_max], [pi_min, pi_min], [6.5, 6.5], facecolor='grey')

plt.show()

# setup second plot
plt.figure(figsize=(8, 5))
plt.xlabel(r"volume flow $\phi$ in $\frac{m^3}{s}$")
plt.xlim(0, 7.5)
plt.ylabel(r"pressure ratio $\frac{\pi}{1}$")
plt.ylim(0, 6.5)
plt.tick_params(labeltop=True, labelright=True, top=True, right=True)

# plot the second bounds

MIN_values = [[0, phi_MIN], [pi_MIN, 0]]
slope = (MIN_values[1][1] - MIN_values[1][0])/(MIN_values[0][1] - MIN_values[0][0])
MAX_values = [[0, -pi_MAX / slope], [pi_MAX, 0]]
pi_max = pi_MAX*((eta - 1)*(p_old(0, _from) - p_in_min)/(p_in_max - p_in_min) + 1)
max_values = [[0, -pi_max / slope], [pi_max, 0]]

x_tick_labels = ['0', '1', '2', '', r'$\phi_{\mathrm{MIN}}$', '4', '5', '6', '7']
x_ticks = [0, 1, 2, 3, phi_MIN, 4, 5, 6, 7]
y_tick_labels = ['0', '1', '', r'$\pi_{\mathrm{MIN}}$', '3', '4', '5',
                 r'$\pi_\max$', r'$\pi_{\mathrm{MAX}}$']
y_ticks = [0, 1, 2, pi_MIN, 3, 4, 5, pi_max, pi_MAX]

plt.plot(MIN_values[0], MIN_values[1], 'orange')
plt.plot(MAX_values[0], MAX_values[1], 'orange')
plt.plot(max_values[0], max_values[1], color='orange', linestyle='--')

plt.xticks(ticks=x_ticks, labels=x_tick_labels)
plt.yticks(ticks=y_ticks, labels=y_tick_labels)
plt.fill_between([0, phi_MIN, -pi_max / slope], [pi_MIN, 0, 0], [pi_max, phi_MIN * slope + pi_max, 0], facecolor='grey')

plt.show()

# setup third plot
plt.figure(figsize=(8, 5))
plt.xlabel(r"volume flow $\phi$ in $\frac{m^3}{s}$")
plt.xlim(0, 7.5)
plt.ylabel(r"pressure ratio $\frac{\pi}{1}$")
plt.ylim(0, 6.5)
plt.tick_params(labeltop=True, labelright=True, top=True, right=True)

# plot the third bounds
slope_pi12 = (pi_1 - pi_2)/(phi_max)
pi_values = [[0, 7.5], [pi_2, 7.5*slope_pi12 + pi_2]]

x_tick_labels = ['0', '1','2', '3', '4', r'$\phi_\max$', '5', '6', '7']
x_ticks = [0, 1, 2, 3, 4, phi_max, 5, 6, 7]
y_tick_labels = ['0', '1', r'$\pi_1$', '2', '3', r'$\pi_2$', '4', '5', '6']
y_ticks = [0, 1, pi_1, 2, 3, pi_2, 4, 5, 6]

plt.plot(pi_values[0], pi_values[1], 'green')
plt.plot(pi_values[0], [pi_1, pi_1], 'g--')
plt.plot([phi_max, phi_max], [0, 6.5], 'g--')

plt.xticks(ticks=x_ticks, labels=x_tick_labels)
plt.yticks(ticks=y_ticks, labels=y_tick_labels)
plt.fill_between([0, 7.5], [0, 0], [pi_2, 7.5*slope_pi12 + pi_2], facecolor='grey')

plt.show()

# setup all bounds plot
plt.figure(figsize=(8, 5))
plt.xlabel(r"volume flow $\phi$ in $\frac{m^3}{s}$")
plt.xlim(0, 7.5)
plt.ylabel(r"pressure ratio $\frac{\pi}{1}$")
plt.ylim(0, 6.5)
plt.tick_params(labeltop=True, labelright=True, top=True, right=True)

# plot all bounds
x_tick_labels = ['0', '', r'$\phi_\min$', '2', '', r'$\phi_{\mathrm{MIN}}$',
                 '4', r'$\phi_\max$', '5', '6', '7']
x_ticks = [0, 1, phi_min, 2, 3, phi_MIN, 4, phi_max, 5, 6, 7]
y_tick_labels = ['0', r'$\pi_\min$', r'$\pi_1$', '', r'$\pi_{\mathrm{MIN}}$',
                 '3', r'$\pi_2$', '4', '5', r'$\pi_\max$', r'$\pi_{\mathrm{MAX}}$']
y_ticks = [0, pi_min, pi_1, 2, pi_MIN, 3, pi_2, 4, 5, pi_max, pi_MAX]

plt.plot(phi_min_values[0], phi_min_values[1], 'blue')
plt.plot(phi_max_values[0], phi_max_values[1], 'blue')
plt.plot(pi_min_values[0], pi_min_values[1], 'blue')
plt.plot(MIN_values[0], MIN_values[1], 'orange')
plt.plot(MAX_values[0], MAX_values[1], 'orange')
plt.plot(max_values[0], max_values[1], color='orange', linestyle='--')
plt.plot(pi_values[0], pi_values[1], 'green')

plt.xticks(ticks=x_ticks, labels=x_tick_labels)
plt.yticks(ticks=y_ticks, labels=y_tick_labels)
x_values_grey = [phi_min, (pi_min - pi_MIN)/slope, (pi_max - pi_2)/(slope_pi12 - slope), phi_max]
y_low_grey = [slope*phi_min + pi_MIN, pi_min, pi_min, pi_min]
y_up_grey = [slope_pi12*phi_min + pi_2, slope_pi12*x_values_grey[1] + pi_2,
             slope_pi12*x_values_grey[2] + pi_2, slope*phi_max + pi_max]
plt.fill_between(x_values_grey, y_low_grey, y_up_grey, facecolor='grey')

plt.show()


# setup complete characteristic diagram plot
plt.figure(figsize=(8, 5))
plt.xlabel(r"volume flow $\phi$ in $\frac{m^3}{s}$")
plt.xlim(0, 7.5)
plt.ylabel(r"pressure ratio $\frac{\pi}{1}$")
plt.ylim(0, 6.5)
plt.tick_params(labeltop=True, labelright=True, top=True, right=True)

# plot all bounds
x_tick_labels = ['0', '', r'$\phi_\min$', '2', '', r'$\phi_{\mathrm{MIN}}$',
                 '4', r'$\phi_\max$', '5', '6', '7']
x_ticks = [0, 1, phi_min, 2, 3, phi_MIN, 4, phi_max, 5, 6, 7]
y_tick_labels = ['0', r'$\pi_\min$', r'$\pi_1$', '', r'$\pi_{\mathrm{MIN}}$',
                 '3', r'$\pi_2$', '4', '5', r'$\pi_\max$', r'$\pi_{\mathrm{MAX}}$']
y_ticks = [0, pi_min, pi_1, 2, pi_MIN, 3, pi_2, 4, 5, pi_max, pi_MAX]

plt.plot(phi_min_values[0], phi_min_values[1], 'blue')
plt.plot(phi_max_values[0], phi_max_values[1], 'blue')
plt.plot(pi_min_values[0], pi_min_values[1], 'blue')
plt.plot(MIN_values[0], MIN_values[1], 'orange')
plt.plot(MAX_values[0], MAX_values[1], 'orange')
plt.plot(max_values[0], max_values[1], color='orange', linestyle='--')
plt.plot(pi_values[0], pi_values[1], 'green')
plt.plot(0, 0, color='red', marker='o', mew=5.0)

plt.xticks(ticks=x_ticks, labels=x_tick_labels)
plt.yticks(ticks=y_ticks, labels=y_tick_labels)
x_values_grey = [phi_min, (pi_min - pi_MIN)/slope, (pi_max - pi_2)/(slope_pi12 - slope), phi_max]
y_low_grey = [slope*phi_min + pi_MIN, pi_min, pi_min, pi_min]
y_up_grey = [slope_pi12*phi_min + pi_2, slope_pi12*x_values_grey[1] + pi_2,
             slope_pi12*x_values_grey[2] + pi_2, slope*phi_max + pi_max]
plt.fill_between(x_values_grey, y_low_grey, y_up_grey, facecolor='grey')

plt.show()


# setup example operating point
plt.figure(figsize=(8, 5))
plt.xlabel(r"volume flow $\phi$ in $\frac{m^3}{s}$")
plt.xlim(0, 7.5)
plt.ylabel(r"pressure ratio $\frac{\pi}{1}$")
plt.ylim(0, 6.5)
plt.tick_params(labeltop=True, labelright=True, top=True, right=True)

# plot all bounds and the example power and pressure line
power_values = [[0, -3.25/slope], [3.25, 0]]
pressure_values = [[0, 7.5], [1.3, 1.3]]
x_tick_labels = ['0', '', r'$\phi_\min$', '2', '', r'$\phi_{\mathrm{MIN}}$',
                 '4', r'$\phi_\max$', '5', '6', '7']
x_ticks = [0, 1, phi_min, 2, 3, phi_MIN, 4, phi_max, 5, 6, 7]
y_tick_labels = ['0', r'$\pi_\min$',r'$\pi$', r'$\pi_1$', '', r'$\pi_{\mathrm{MIN}}$',
                 '3', r'$\pi_2$', '4', '5', r'$\pi_\max$', r'$\pi_{\mathrm{MAX}}$']
y_ticks = [0, pi_min, 1.3, pi_1, 2, pi_MIN, 3, pi_2, 4, 5, pi_max, pi_MAX]

plt.plot(phi_min_values[0], phi_min_values[1], 'blue')
plt.plot(phi_max_values[0], phi_max_values[1], 'blue')
plt.plot(pi_min_values[0], pi_min_values[1], 'blue')
plt.plot(MIN_values[0], MIN_values[1], 'orange')
plt.plot(MAX_values[0], MAX_values[1], 'orange')
plt.plot(max_values[0], max_values[1], color='orange', linestyle='--')
plt.plot(pi_values[0], pi_values[1], 'green')
plt.plot(0, 0, color='red', marker='o', mew=5.0)
plt.plot(power_values[0], power_values[1], color='m', linestyle='--', label=r'$L$', lw=1.0)
plt.plot(pressure_values[0], pressure_values[1], color='c', linestyle='--', lw=1.0)
plt.plot((1.3 - 3.25)/slope, 1.3, color='black', marker='x', ms=8.0, mew=2.0, label='operating point', lw=0.0)


plt.xticks(ticks=x_ticks, labels=x_tick_labels)
plt.yticks(ticks=y_ticks, labels=y_tick_labels)
x_values_grey = [phi_min, (pi_min - pi_MIN)/slope, (pi_max - pi_2)/(slope_pi12 - slope), phi_max]
y_low_grey = [slope*phi_min + pi_MIN, pi_min, pi_min, pi_min]
y_up_grey = [slope_pi12*phi_min + pi_2, slope_pi12*x_values_grey[1] + pi_2,
             slope_pi12*x_values_grey[2] + pi_2, slope*phi_max + pi_max]
plt.fill_between(x_values_grey, y_low_grey, y_up_grey, facecolor='grey')

plt.legend()
plt.show()
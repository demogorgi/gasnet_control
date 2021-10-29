set terminal qt size 1200, 400 font 'Helvetica, 6'
set key autotitle columnhead noenhanced outside center bottom
set datafile separator ';'
set xtics 0, 8
set grid
plot 'pltit.csv' u 1 w steps lt 1 lw 1, 'pltit.csv' u 2 w steps lt 2 lw 1,  'pltit.csv' u 3 w steps lt 1 dt 2,  'pltit.csv' u 4 w steps lt 2 dt 2
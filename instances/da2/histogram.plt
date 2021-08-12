reset
n=110 #number of intervals
max=110. #max value
min=-0 #min value
width=(max-min)/n #interval width
#function used to map a value to the intervals
hist(x,width)=width*(floor((x-min)/width)+0.5) + min
set term png #output terminal and file
set output "histogram.png"
set xrange [min:max]
set yrange [0:3500]
#to put an empty boundary around the
#data inside an autoscaled graph.
set offset graph 0.05,0.05,0.05,0.0
#set xtics min,(max-min)/5,max
set boxwidth width*0.9
set style fill solid 0.5 #fillstyle
set tics out nomirror
#set xtics 0,0.1,1.1
#set ytics 0,100
set grid
set xtics 0,10,100
set xlabel "Zeta"
set ylabel "Frequency"
#count and plot
plot "zeta_idle_true.csv" u (hist($2,width)):(1.0) smooth freq w boxes lc rgb "green" notitle

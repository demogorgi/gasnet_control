#!/usr/bin/env ruby
puts "usage: ./pltit.rb test_case variable_list_from_solution_file"
puts "example: ./pltit.rb da2 \"var_pipe_Qo_in\\[EN_aux0,EN\" \"entry_nom_TA\\[EN_aux0,EN\" \"var_pipe_Qo_in\\[EH_aux0,EH\" \"entry_nom_TA\\[EH_aux0,EH\"\n\n"
require 'csv'

def hash2csv(h,file)
  csv = h.keys.join(';') + "\n"
  csv += h.values.transpose.map{ |x| x.join(';') }.join("\n")
  File.write(file, csv)
end

data = {}
Dir.glob('instances/' + ARGV[0] + '/output/*.sol').sort.each do |sol_filename|
  File.open(sol_filename).each do |line|
    x = ARGV[1..-1].map{ |a| line.match(a + '.*') }.compact
    if x[0] != nil
      tup = x[0][0].split(" ")
      data[tup[0]] ||= []
      data[tup[0]] << tup[1].to_f
    end
  end
end
hash2csv(data, 'pltit.csv')

plt = "set terminal qt size 1200, 400 font 'Helvetica, 6'\n"
plt += "set key autotitle columnhead noenhanced outside center bottom\n"
plt += "set datafile separator ';'\n"
plt += "set xtics 0, 8\n"
plt += "set grid\n"
#plt += "set size ratio 0.2\n"
#
# general setup
#plt += "plot for [i=1:10] 'pltit.csv' u i w steps"
#
# special setup for nom vs. flow
plt += "plot 'pltit.csv' u 1 w steps lt 1 lw 1, 'pltit.csv' u 2 w steps lt 2 lw 1,  'pltit.csv' u 3 w steps lt 1 dt 2,  'pltit.csv' u 4 w steps lt 2 dt 2"
#
File.write('pltit.plt', plt)
system('gnuplot -p pltit.plt')

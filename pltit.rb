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
#Dir.glob('instances/' + ARGV[0] + '/output/*.sol') do |sol_filename|
  pp sol_filename
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

plt = "set key autotitle columnhead noenhanced\nset datafile separator ';'\n"
plt += "plot for [i=1:10] 'pltit.csv' u i w l"
File.write('pltit.plt', plt)
system('gnuplot -p pltit.plt')

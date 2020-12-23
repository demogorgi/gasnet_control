#
# this script generates state-files from solution files. state files can be viewed in contour.
#
# Invocation:
# ruby sol2state.rb scenarioPath
#
# Gurobi has to write .sol files if you want to use this script
#
require 'rexml/document'
require 'pp'
require 'date'
require 'fileutils'

include REXML

def getBoundaryNodeFlowIn(solutionfile)
	flow_in = {}
	File.readlines(solutionfile).each do |line|
	        data = line.match(/var_node_Qo_in\[([^\s]*)\]\s(.*)/)
		if !data.nil?
			flow_in[data[1]] = data[2].to_f.round(3).to_s
		end
	end
	flow_in
end

def setBoundaryNodeFlowIn(stateXml,inflow)
	stateXml.elements.each("//boundaryNodes/*") do |e|
		al = e.attributes['alias']
		e.elements["inflow"].attributes["value"] = inflow[al]
	end
end

# get node pressures in barg
def getNodePressures(solutionfile)
	ps = {}
	File.readlines(solutionfile).each do |line|
		data = line.match(/var_node_p\[([^\s]*)\]\s(.*)/)
		if !data.nil?
		    ps[data[1]] = (data[2].to_f - Barg2Bar).round(3).to_s
		end
	end
	ps
end

def setNodePressures(stateXml, pressures)
	stateXml.elements.each("//nodes/*/*") do |e|
		al = e.attributes["alias"]
		e.elements["pressure"].attributes["value"] = pressures[al]
	end
end

def mean(a,b)
	return (a+b)/2.0
end

def getPipeFlow(solutionfile)
	flow = {}
	File.readlines(solutionfile).each do |line|
	        data = line.match(/var_pipe_Qo_(out|in)\[([^\s]*)\]\s*(.*)/)
		if !data.nil?
			flow[data[2].sub(",","^")] = data[3].to_f.round(3).to_s
		end
	end
	flow
end

def setPipeFlow(stateXml, flow)
	stateXml.elements.each("//connections/pipe") do |e|
		al= e.attributes['alias']
		e.elements["flow"].attributes["value"] = flow[al]
	end
end

def getNonPipeFlow(solutionfile)
	flow = {}
	File.readlines(solutionfile).each do |line|
		data = line.match(/var_non_pipe_Qo\[([^\s]*)\]\s*(.*)/)
		if !data.nil?
		        flow[data[1].sub(",","^")] = data[2].to_f.round(3).to_s
		end
	end
	flow
end

def setNonPipeFlow(stateXml, flow)
	stateXml.elements.each("//connections/*") do |e|
		if e.name != "pipe"
			al = e.attributes['alias']
			e.elements["flow"].attributes["value"] = flow[al]
		end
	end
end

def getDragFactor(solutionFile)
	df = {}
	File.readlines(solutionFile).each do |line|
		data = line.match(/zeta_DA\[([^\s]*)\]\s*(.*)/)
		if !data.nil?
			df[data[1].sub(",","^")] = data[2].to_f.round(3).to_s
		end
	end
	df
end

def setDragFactor(stateXml, dragFactor)
	stateXml.elements.each("//connections/resistor") do |e|
		al = e.attributes['alias']
		e.elements["dragFactor"].attributes["value"] = dragFactor[al]
	end
end

def openClosed(val)
	if val == "1"
		"open"
	else
		"closed"
	end
end

def getValveMode(solutionFile)
	va = {}
	File.readlines(solutionFile).each do |line|
		data = line.match(/va_DA\[([^\s]*)\]\s*(.*)/)
		if !data.nil?
			va[data[1].sub(",","^")] = openClosed(data[2])
		end
	end
	va
end

def setValveMode(stateXml, modes)
	stateXml.elements.each("//connections/valve") do |e|
		al = e.attributes['alias']
		e.elements["mode"].attributes["value"] = modes[al]
	end
end

def closedFree(val)
	if val == "1"
		"free"
	else
		"closed"
	end
end

def getCompressorConfig(solutionFile)
	cs = {}
	File.readlines(solutionFile).each do |line|
		data = line.match(/compressor_DA\[([^\s]*)\]\s*(.*)/)
		if !data.nil?
			cs[data[1].sub(",","^")] = closedFree(data[2])
		end
	end
	cs
end

def setCompressorConfig(stateXml, configs)
	stateXml.elements.each("//connections/compressorStation") do |e|
		al = e.attributes['alias']
		e.attributes["configuration"] = configs[al]
	end
end

Barg2Bar = 1.01325

def sol2state(scenarioPath,contourInput,solFile,stateTemplate,timestamp)

	i = File.basename(solFile).scan(/(\d+)/).join("_")
	stateFile = File.join(contourInput, "state_#{i}.xml")
	FileUtils.cp(stateTemplate, stateFile)

	doc = Document.new(File.open(stateFile))

	el = doc.elements['//timeOfState/']
	el.attributes['timestamp'] = timestamp

	setBoundaryNodeFlowIn(doc, getBoundaryNodeFlowIn(solFile))
	setNodePressures(doc, getNodePressures(solFile))
	setPipeFlow(doc, getPipeFlow(solFile))
	setNonPipeFlow(doc, getNonPipeFlow(solFile))
	setDragFactor(doc, getDragFactor(solFile))
	setValveMode(doc, getValveMode(solFile))
	setCompressorConfig(doc, getCompressorConfig(solFile))

	File.write(stateFile, doc)
	formatter = Formatters::Pretty.new(5)
	formatter.compact = true
	formatter.write(doc, stateFile)

end

scenarioPath = ARGV[0]
stateTemplate = File.open(File.join(scenarioPath, "state_sim.xml"), "r")
solFiles = Dir.glob(File.join(scenarioPath, "output/*.sol")).sort()
contourInput = File.join(scenarioPath, "/output/contour")
Dir.mkdir(contourInput) unless File.exists?(contourInput)
FileUtils.cp(File.join(scenarioPath,"net_sim.xml"), contourInput)
#FileUtils.cp(File.join(scenarioPath,"state_sim.xml"), contourInput)
timestamp = Time.now.utc
solFiles.each_with_index{ |f,i|
    puts("process #{f}")
    sol2state(scenarioPath,contourInput,f,stateTemplate,(timestamp + i * 900).utc.strftime("%Y-%m-%dT%H:%M:%SZ"))
}
stateTemplate.close()

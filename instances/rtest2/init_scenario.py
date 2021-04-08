import gurobipy as gp
from gurobipy import GRB

var_node_p_old_old = {'START_ND': 81.01325, 'START_HD': 80.9978297372234, 'END': 68.78488020817191, 'N407': 80.3257295639811, 'N366': 78.0711343441741, 'N409': 79.20789934513083, 'N408': 79.76910041389574, 'N433': 74.53376936204182, 'N376': 72.05362500540015, 'N432': 75.1377081876678, 'N431': 75.73563373234985, 'N430': 76.32774728547794, 'N436': 72.68376204759265, 'N435': 73.30701411980178, 'N434': 73.92360947894471, 'N418': 69.45441271502251, 'B': 72.05362532417954, 'N428': 77.49531194286514, 'N429': 76.91424378588161, 'N410': 78.64196604539336, 'N322': 80.87794131911261, 'N415': 71.41514880697154, 'N416': 70.76928754650065, 'A': 78.07113431393165, 'N417': 70.11579393925979, 'START': 80.89169103225669, 'START_aux0': 80.99011826211758, 'START_aux1': 80.9978297372234, 'START_NDin1': 81.00554029611732, 'START_NDin2': 81.00554029611732, 'START_aux2': 80.99782973722343, 'START_aux3': 80.99782973722343, 'START_HDin1': 80.99782973722343, 'START_HDin2': 80.99782973722343, 'B_aux': 72.05362532417954}
var_node_p_old = {'START_ND': 81.01325, 'START_HD': 80.99782666062444, 'END': 68.78315577176214, 'N407': 80.32559204200153, 'N366': 78.07054898406895, 'N409': 79.2075380143119, 'N408': 79.76885127941566, 'N433': 74.53257378566387, 'N376': 72.05210538233997, 'N432': 75.1366063916376, 'N431': 75.73462965791241, 'N430': 76.32684429103494, 'N436': 72.68231461595957, 'N435': 73.30564514361389, 'N434': 73.9223246479508, 'N418': 69.45271226862101, 'B': 72.05210569715962, 'N428': 77.49461887422002, 'N429': 76.91344465094896, 'N410': 78.6414924591323, 'N322': 80.87791431852956, 'N415': 71.41357220252311, 'N416': 70.76766148223705, 'A': 78.07054895371725, 'N417': 70.11412643503078, 'START': 80.89166677747194, 'START_aux0': 80.99011364765889, 'START_aux1': 80.99782666062444, 'START_NDin1': 81.00553875788103, 'START_NDin2': 81.00553875788103, 'START_aux2': 80.99782666062447, 'START_aux3': 80.99782666062447, 'START_HDin1': 80.99782666062447, 'START_HDin2': 80.99782666062447, 'B_aux': 72.05210569715962}
var_non_pipe_Qo_old_old = {('A', 'B_aux'): 0.25174232136384334, ('START_NDin2', 'START_NDin1'): 792.4766746929794, ('START_HDin1', 'START_aux3'): 0.0, ('B_aux', 'B'): 0.25174232136384334}
var_non_pipe_Qo_old = {('A', 'B_aux'): 0.2517596152795359, ('START_NDin2', 'START_NDin1'): 792.555087154818, ('START_HDin1', 'START_aux3'): 0.0, ('B_aux', 'B'): 0.2517596152795359}
var_pipe_Qo_in_old_old = {('N432', 'N433'): 794.353244437853, ('N431', 'N432'): 794.0028562754134, ('N430', 'N431'): 793.6852079575067, ('N429', 'N430'): 793.4013975845849, ('N436', 'N376'): 796.0535368955839, ('N435', 'N436'): 795.5869955128003, ('N322', 'N407'): 792.4831033952868, ('N434', 'N435'): 795.1468905199703, ('N433', 'N434'): 794.7350828813012, ('N366', 'A'): 0.05603834518842632, ('N428', 'N429'): 793.1523326403596, ('N415', 'N416'): 797.8104960147251, ('START', 'N322'): 792.4828881471561, ('N416', 'N417'): 798.3391584730284, ('N417', 'N418'): 798.8821657060867, ('N418', 'END'): 799.4367501820949, ('N366', 'N428'): 792.938731586458, ('B', 'N376'): 0.25174232136384334, ('N407', 'N408'): 792.5107266493599, ('N408', 'N409'): 792.5755843675971, ('N409', 'N410'): 792.6778850940465, ('N376', 'N415'): 797.2987932103612, ('N410', 'N366'): 792.8176624059338, ('START_aux0', 'START'): 792.4828820847233, ('START_aux1', 'START_aux0'): 792.4802977013981, ('START_NDin1', 'START_aux1'): 792.4766746929794, ('START_ND', 'START_NDin2'): 792.4761577944209, ('START_aux2', 'START_aux1'): 4.652186505431156e-06, ('START_aux3', 'START_aux2'): 0.0, ('START_HDin2', 'START_HDin1'): -4.652186502013627e-06, ('START_HD', 'START_HDin2'): -0.0020722906297262287}
var_pipe_Qo_in_old = {('N432', 'N433'): 794.4094584419271, ('N431', 'N432'): 794.0627218363821, ('N430', 'N431'): 793.7483838618763, ('N429', 'N430'): 793.4675312017183, ('N436', 'N376'): 796.0920304824559, ('N435', 'N436'): 795.6303515699843, ('N322', 'N407'): 792.5614488246193, ('N434', 'N435'): 795.194833381282, ('N433', 'N434'): 794.7873175198841, ('N366', 'A'): 0.05809522615891183, ('N428', 'N429'): 793.2210618985172, ('N415', 'N416'): 797.8333181958043, ('START', 'N322'): 792.5612358208485, ('N416', 'N417'): 798.3564702079818, ('N417', 'N418'): 798.8938173758887, ('N418', 'END'): 799.4426211114851, ('N366', 'N428'): 793.0096869299997, ('B', 'N376'): 0.2517596152795359, ('N407', 'N408'): 792.5887840863793, ('N408', 'N409'): 792.6529656752934, ('N409', 'N410'): 792.7542000386612, ('N376', 'N415'): 797.32694889969, ('N410', 'N366'): 792.8925204657539, ('START_aux0', 'START'): 792.5612298216281, ('START_aux1', 'START_aux0'): 792.558672385852, ('START_NDin1', 'START_aux1'): 792.555087154818, ('START_ND', 'START_NDin2'): 792.5545756460047, ('START_aux2', 'START_aux1'): 4.603677818813015e-06, ('START_aux3', 'START_aux2'): 0.0, ('START_HDin2', 'START_HDin1'): -4.6036778219638316e-06, ('START_HD', 'START_HDin2'): -0.00205068270748246}
var_pipe_Qo_out_old_old = {('N432', 'N433'): 794.7350828813012, ('N431', 'N432'): 794.353244437853, ('N430', 'N431'): 794.0028562754134, ('N429', 'N430'): 793.6852079575067, ('N436', 'N376'): 796.544467117002, ('N435', 'N436'): 796.0535368955839, ('N322', 'N407'): 792.5107266493599, ('N434', 'N435'): 795.5869955128003, ('N433', 'N434'): 795.1468905199703, ('N366', 'A'): 0.25174232136384334, ('N428', 'N429'): 793.4013975845849, ('N415', 'N416'): 798.3391584730284, ('START', 'N322'): 792.4831033952868, ('N416', 'N417'): 798.8821657060867, ('N417', 'N418'): 799.4367501820949, ('N418', 'END'): 800.0, ('N366', 'N428'): 793.1523326403596, ('B', 'N376'): 0.7543260933622883, ('N407', 'N408'): 792.5755843675971, ('N408', 'N409'): 792.6778850940465, ('N409', 'N410'): 792.8176624059338, ('N376', 'N415'): 797.8104960147251, ('N410', 'N366'): 792.9947699316438, ('START_aux0', 'START'): 792.4828881471561, ('START_aux1', 'START_aux0'): 792.4828820847233, ('START_NDin1', 'START_aux1'): 792.4782254107631, ('START_ND', 'START_NDin2'): 792.4766746929794, ('START_aux2', 'START_aux1'): 0.002072290630798471, ('START_aux3', 'START_aux2'): 4.652186505431156e-06, ('START_HDin2', 'START_HDin1'): 0.0, ('START_HD', 'START_HDin2'): -4.652186502013627e-06}
var_pipe_Qo_out_old = {('N432', 'N433'): 794.7873175198841, ('N431', 'N432'): 794.4094584419271, ('N430', 'N431'): 794.0627218363821, ('N429', 'N430'): 793.7483838618763, ('N436', 'N376'): 796.5778438824179, ('N435', 'N436'): 796.0920304824559, ('N322', 'N407'): 792.5887840863793, ('N434', 'N435'): 795.6303515699843, ('N433', 'N434'): 795.194833381282, ('N366', 'A'): 0.2517596152795359, ('N428', 'N429'): 793.4675312017183, ('N415', 'N416'): 798.3564702079818, ('START', 'N322'): 792.5614488246193, ('N416', 'N417'): 798.8938173758887, ('N417', 'N418'): 799.4426211114851, ('N418', 'END'): 800.0, ('N366', 'N428'): 793.2210618985172, ('B', 'N376'): 0.7491050172704846, ('N407', 'N408'): 792.6529656752934, ('N408', 'N409'): 792.7542000386612, ('N409', 'N410'): 792.8925204657539, ('N376', 'N415'): 797.8333181958043, ('N410', 'N366'): 793.0677821561602, ('START_aux0', 'START'): 792.5612358208485, ('START_aux1', 'START_aux0'): 792.5612298216281, ('START_NDin1', 'START_aux1'): 792.5566217031446, ('START_ND', 'START_NDin2'): 792.555087154818, ('START_aux2', 'START_aux1'): 0.002050682706226558, ('START_aux3', 'START_aux2'): 4.603677818813015e-06, ('START_HDin2', 'START_HDin1'): 0.0, ('START_HD', 'START_HDin2'): -4.6036778219638316e-06}
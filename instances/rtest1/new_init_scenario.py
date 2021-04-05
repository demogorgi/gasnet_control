import gurobipy as gp
from gurobipy import GRB

var_node_p_old_old = {'START_ND': 108.67950810584539, 'START_HD': 26.5980107878264, 'START_1_ND': 51.01325, 'START_1_HD': 51.01325, 'END_1': 50.85221691450455, 'END': 108.66855585061104, 'N363_1': 51.011622305755466, 'N366_1': 51.01151288131502, 'N365_1': 51.0115493565737, 'N364_1': 51.011585831384075, 'N376_1': 50.852243589674764, 'N380_1': 50.85223439649015, 'N379_1': 50.85223717462243, 'N378_1': 50.852239662380306, 'N377_1': 50.852241821153726, 'N381_1': 50.8522313552679, 'B_1': 50.85224484193249, 'N382_1': 50.852228071576825, 'N383_1': 50.85222456171496, 'N384_1': 50.852220838983655, 'N322_1': 51.011658779697136, 'A_1': 51.011476405598735, 'N384': 108.66855573959609, 'N383': 108.66855565066145, 'N382': 108.66855558135377, 'N381': 108.66855552921969, 'N380': 108.6685554918058, 'N379': 108.66855546665876, 'N378': 108.6685554513252, 'N377': 108.66855544335175, 'N376': 108.66855544028505, 'N366': 108.66855544028505, 'N365': 108.66855544335175, 'N364': 108.6685554513252, 'N363': 108.66855546665876, 'N322': 108.6685554918058, 'B': 108.6685554396717, 'A': 108.6685554396717, 'START': 108.66855587514472, 'START_aux0': 108.66856038043885, 'START_aux1': 108.66885048206443, 'START_NDin1': 108.67251059609562, 'START_NDin2': 108.67251059609562, 'START_aux2': 108.66857362463936, 'START_aux3': 108.66857112705821, 'START_HDin1': 26.598010766652884, 'START_HDin2': 26.598010766842187, 'START_1': 51.01184114337883, 'START_1_aux0': 51.013147576408976, 'START_1_aux1': 51.01325, 'START_1_NDin1': 51.01325, 'START_1_NDin2': 51.01325, 'START_1_aux2': 51.01325, 'START_1_aux3': 51.01325, 'START_1_HDin1': 51.01325, 'START_1_HDin2': 51.01325, 'B_aux': 108.6685554396717}
var_node_p_old = {'START_ND': 109.75824475097632, 'START_HD': 26.603078050146785, 'START_1_ND': 51.01325, 'START_1_HD': 51.01325, 'END_1': 50.85175368464963, 'END': 109.74741627399196, 'N363_1': 51.01161751451075, 'N366_1': 51.01150777023661, 'N365_1': 51.011544351999284, 'N364_1': 51.01158093342166, 'N376_1': 50.85178037408403, 'N380_1': 50.85177117227519, 'N379_1': 50.851773952508026, 'N378_1': 50.85177644249155, 'N377_1': 50.85177860349446, 'N381_1': 50.85176812917523, 'B_1': 50.851781627970986, 'N382_1': 50.8517648439122, 'N383_1': 50.85176133285684, 'N384_1': 50.85175760937564, 'N322_1': 51.011654095273535, 'A_1': 51.01147118812667, 'N384': 109.74741616423188, 'N383': 109.74741607630253, 'N382': 109.74741600777827, 'N381': 109.74741595623348, 'N380': 109.74741591924253, 'N379': 109.74741589437976, 'N378': 109.74741587921955, 'N377': 109.74741587133622, 'N376': 109.74741586830417, 'N366': 109.74741586830417, 'N365': 109.7474158713362, 'N364': 109.74741587921952, 'N363': 109.74741589437977, 'N322': 109.74741591924254, 'B': 109.74741586769777, 'A': 109.74741586769777, 'START': 109.74741629824833, 'START_aux0': 109.74742075261614, 'START_aux1': 109.74770757506647, 'START_NDin1': 109.75132631965782, 'START_NDin2': 109.75132631965782, 'START_aux2': 109.747433847109, 'START_aux3': 109.74743137775964, 'START_HDin1': 26.60307802897757, 'START_HDin2': 26.603078029166834, 'START_1': 51.01183699450735, 'START_1_aux0': 51.01314727305452, 'START_1_aux1': 51.01325, 'START_1_NDin1': 51.01325, 'START_1_NDin2': 51.01325, 'START_1_aux2': 51.01325, 'START_1_aux3': 51.01325, 'START_1_HDin1': 51.01325, 'START_1_HDin2': 51.01325, 'B_aux': 109.74741586769777}
var_non_pipe_Qo_old_old = {('A', 'B_aux'): 0.0, ('START_NDin2', 'START_NDin1'): 751.7419965162735, ('START_HDin1', 'START_aux3'): 0.0, ('START_1_NDin2', 'START_1_NDin1'): 0.0, ('START_1_HDin1', 'START_1_aux3'): 99.70455080700016, ('B_aux', 'B'): 0.0}
var_non_pipe_Qo_old = {('A', 'B_aux'): 0.0, ('START_NDin2', 'START_NDin1'): 751.741742383131, ('START_HDin1', 'START_aux3'): 0.0, ('START_1_NDin2', 'START_1_NDin1'): 99.70205438214067, ('START_1_HDin1', 'START_1_aux3'): 0.0, ('B_aux', 'B'): 0.0}
var_pipe_Qo_in_old_old = {('N363_1', 'N364_1'): 99.70970858829281, ('N365_1', 'N366_1'): 99.71028500231593, ('N364_1', 'N365_1'): 99.70999363672617, ('START_1', 'N322_1'): 99.70813094978563, ('N376_1', 'N377_1'): 99.74911036883152, ('B_1', 'N376_1'): 99.72123594053556, ('N377_1', 'N378_1'): 99.77698506538329, ('N378_1', 'N379_1'): 99.8048601125836, ('N379_1', 'N380_1'): 99.8327355765419, ('N380_1', 'N381_1'): 99.86061151468589, ('N322_1', 'N363_1'): 99.70942985688606, ('N381_1', 'N382_1'): 99.88848797871668, ('N382_1', 'N383_1'): 99.91636501617278, ('N383_1', 'N384_1'): 99.9442426713854, ('N384_1', 'END_1'): 99.97212098611469, ('N366_1', 'A_1'): 99.71058268519471, ('N384', 'END'): -5.586203122568922, ('N383', 'N384'): -4.965513887790755, ('N382', 'N383'): -4.344824652543054, ('N381', 'N382'): -3.724135416935007, ('N380', 'N381'): -3.103446181061251, ('N379', 'N380'): -2.4827569450018707, ('N378', 'N379'): -1.8620677088223785, ('N377', 'N378'): -1.2413784725737316, ('N376', 'N377'): -0.6206892362923281, ('N366', 'A'): 0.6206892362923281, ('N365', 'N366'): 1.241378472573726, ('N364', 'N365'): 1.8620677088223787, ('N363', 'N364'): 2.482756945001873, ('N322', 'N363'): 3.1034461810612526, ('B', 'N376'): 0.0, ('START', 'N322'): 6.206892355297886, ('A_1', 'B_1'): 99.7108866854979, ('START_aux0', 'START'): 6.362064660383313, ('START_aux1', 'START_aux0'): 254.63740313690386, ('START_NDin1', 'START_aux1'): 751.7419965162735, ('START_ND', 'START_NDin2'): 1000.0, ('START_aux2', 'START_aux1'): -0.5586202222091643, ('START_aux3', 'START_aux2'): 0.0, ('START_HDin2', 'START_HDin1'): 0.0022449488659459504, ('START_HD', 'START_HDin2'): 1.0, ('START_1_aux0', 'START_1'): 99.70809823388494, ('START_1_aux1', 'START_1_aux0'): 99.70455080700016, ('START_1_NDin1', 'START_1_aux1'): 0.0, ('START_1_ND', 'START_1_NDin2'): 0.0, ('START_1_aux2', 'START_1_aux1'): 99.70455080700016, ('START_1_aux3', 'START_1_aux2'): 99.70455080700016, ('START_1_HDin2', 'START_1_HDin1'): 99.70455080700016, ('START_1_HD', 'START_1_HDin2'): 99.70455080700016}
var_pipe_Qo_in_old = {('N363_1', 'N364_1'): 99.70725790519946, ('N365_1', 'N366_1'): 99.7078394047016, ('N364_1', 'N365_1'): 99.707545468714, ('START_1', 'N322_1'): 99.70566634144625, ('N376_1', 'N377_1'): 99.74699276907593, ('B_1', 'N376_1'): 99.71888304925929, ('N377_1', 'N378_1'): 99.77510275707584, ('N378_1', 'N379_1'): 99.80321309551756, ('N379_1', 'N380_1'): 99.83132385035302, ('N380_1', 'N381_1'): 99.85943507882997, ('N322_1', 'N363_1'): 99.70697671408504, ('N381_1', 'N382_1'): 99.88754683244906, ('N382_1', 'N383_1'): 99.9156591585296, ('N383_1', 'N384_1'): 99.94377210116613, ('N384_1', 'END_1'): 99.97188570186594, ('N366_1', 'A_1'): 99.70813971323675, ('N384', 'END'): -5.586200315442361, ('N383', 'N384'): -4.965511392544385, ('N382', 'N383'): -4.344822469186935, ('N381', 'N382'): -3.724133545476861, ('N380', 'N381'): -3.10344462150677, ('N379', 'N380'): -2.4827556973550187, ('N378', 'N379'): -1.8620667730857225, ('N377', 'N378'): -1.241377848748752, ('N376', 'N377'): -0.6206889243797207, ('N366', 'A'): 0.6206889243797207, ('N365', 'N366'): 1.241377848748764, ('N364', 'N365'): 1.862066773085733, ('N363', 'N364'): 2.4827556973550284, ('N322', 'N363'): 3.103444621506779, ('B', 'N376'): 0.0, ('START', 'N322'): 6.206889236335067, ('A_1', 'B_1'): 99.70844639439561, ('START_aux0', 'START'): 6.362061463527766, ('START_aux1', 'START_aux0'): 254.63728280221105, ('START_NDin1', 'START_aux1'): 751.741742383131, ('START_ND', 'START_NDin2'): 1000.0, ('START_aux2', 'START_aux1'): -0.5586199434255406, ('START_aux3', 'START_aux2'): 0.0, ('START_HDin2', 'START_HDin1'): 0.0022449488659450623, ('START_HD', 'START_HDin2'): 1.0, ('START_1_aux0', 'START_1'): 99.70563333656467, ('START_1_aux1', 'START_1_aux0'): 99.70205438214067, ('START_1_NDin1', 'START_1_aux1'): 99.70205438214067, ('START_1_ND', 'START_1_NDin2'): 99.70205438214067, ('START_1_aux2', 'START_1_aux1'): 0.0, ('START_1_aux3', 'START_1_aux2'): 0.0, ('START_1_HDin2', 'START_1_HDin1'): 0.0, ('START_1_HD', 'START_1_HDin2'): 0.0}
var_pipe_Qo_out_old_old = {('N363_1', 'N364_1'): 99.70999363672617, ('N365_1', 'N366_1'): 99.71058268519471, ('N364_1', 'N365_1'): 99.71028500231593, ('START_1', 'N322_1'): 99.70942985688606, ('N376_1', 'N377_1'): 99.77698506538329, ('B_1', 'N376_1'): 99.74911036883152, ('N377_1', 'N378_1'): 99.8048601125836, ('N378_1', 'N379_1'): 99.8327355765419, ('N379_1', 'N380_1'): 99.86061151468589, ('N380_1', 'N381_1'): 99.88848797871668, ('N322_1', 'N363_1'): 99.70970858829281, ('N381_1', 'N382_1'): 99.91636501617278, ('N382_1', 'N383_1'): 99.9442426713854, ('N383_1', 'N384_1'): 99.97212098611469, ('N384_1', 'END_1'): 100.0, ('N366_1', 'A_1'): 99.7108866854979, ('N384', 'END'): -6.2068923567537695, ('N383', 'N384'): -5.586203122568922, ('N382', 'N383'): -4.965513887790755, ('N381', 'N382'): -4.344824652543054, ('N380', 'N381'): -3.724135416935007, ('N379', 'N380'): -3.103446181061251, ('N378', 'N379'): -2.4827569450018707, ('N377', 'N378'): -1.8620677088223785, ('N376', 'N377'): -1.2413784725737316, ('N366', 'A'): 0.0, ('N365', 'N366'): 0.6206892362923281, ('N364', 'N365'): 1.241378472573726, ('N363', 'N364'): 1.8620677088223787, ('N322', 'N363'): 2.482756945001873, ('B', 'N376'): -0.6206892362923281, ('START', 'N322'): 3.1034461810612526, ('A_1', 'B_1'): 99.72123594053556, ('START_aux0', 'START'): 6.206892355297886, ('START_aux1', 'START_aux0'): 6.362064660383313, ('START_NDin1', 'START_aux1'): 503.4713461163464, ('START_ND', 'START_NDin2'): 751.7419965162735, ('START_aux2', 'START_aux1'): -248.8339429794405, ('START_aux3', 'START_aux2'): -0.5586202222091643, ('START_HDin2', 'START_HDin1'): 0.0, ('START_HD', 'START_HDin2'): 0.0022449488659459504, ('START_1_aux0', 'START_1'): 99.70813094978563, ('START_1_aux1', 'START_1_aux0'): 99.70809823388494, ('START_1_NDin1', 'START_1_aux1'): 0.0, ('START_1_ND', 'START_1_NDin2'): 0.0, ('START_1_aux2', 'START_1_aux1'): 99.70455080700016, ('START_1_aux3', 'START_1_aux2'): 99.70455080700016, ('START_1_HDin2', 'START_1_HDin1'): 99.70455080700016, ('START_1_HD', 'START_1_HDin2'): 99.70455080700016}
var_pipe_Qo_out_old = {('N363_1', 'N364_1'): 99.707545468714, ('N365_1', 'N366_1'): 99.70813971323675, ('N364_1', 'N365_1'): 99.7078394047016, ('START_1', 'N322_1'): 99.70697671408504, ('N376_1', 'N377_1'): 99.77510275707584, ('B_1', 'N376_1'): 99.74699276907593, ('N377_1', 'N378_1'): 99.80321309551756, ('N378_1', 'N379_1'): 99.83132385035302, ('N379_1', 'N380_1'): 99.85943507882997, ('N380_1', 'N381_1'): 99.88754683244906, ('N322_1', 'N363_1'): 99.70725790519946, ('N381_1', 'N382_1'): 99.9156591585296, ('N382_1', 'N383_1'): 99.94377210116613, ('N383_1', 'N384_1'): 99.97188570186594, ('N384_1', 'END_1'): 100.0, ('N366_1', 'A_1'): 99.70844639439561, ('N384', 'END'): -6.206889237759924, ('N383', 'N384'): -5.586200315442361, ('N382', 'N383'): -4.965511392544385, ('N381', 'N382'): -4.344822469186935, ('N380', 'N381'): -3.724133545476861, ('N379', 'N380'): -3.10344462150677, ('N378', 'N379'): -2.4827556973550187, ('N377', 'N378'): -1.8620667730857225, ('N376', 'N377'): -1.241377848748752, ('N366', 'A'): 0.0, ('N365', 'N366'): 0.6206889243797207, ('N364', 'N365'): 1.241377848748764, ('N363', 'N364'): 1.862066773085733, ('N322', 'N363'): 2.4827556973550284, ('B', 'N376'): -0.6206889243797207, ('START', 'N322'): 3.103444621506779, ('A_1', 'B_1'): 99.71888304925929, ('START_aux0', 'START'): 6.206889236335067, ('START_aux1', 'START_aux0'): 6.362061463527766, ('START_NDin1', 'START_aux1'): 503.4711087017772, ('START_ND', 'START_NDin2'): 751.741742383131, ('START_aux2', 'START_aux1'): -248.83382589955866, ('START_aux3', 'START_aux2'): -0.5586199434255406, ('START_HDin2', 'START_HDin1'): 0.0, ('START_HD', 'START_HDin2'): 0.0022449488659450623, ('START_1_aux0', 'START_1'): 99.70566634144625, ('START_1_aux1', 'START_1_aux0'): 99.70563333656467, ('START_1_NDin1', 'START_1_aux1'): 99.70205438214067, ('START_1_ND', 'START_1_NDin2'): 99.70205438214067, ('START_1_aux2', 'START_1_aux1'): 0.0, ('START_1_aux3', 'START_1_aux2'): 0.0, ('START_1_HDin2', 'START_1_HDin1'): 0.0, ('START_1_HD', 'START_1_HDin2'): 0.0}
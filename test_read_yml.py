import yaml
import os
import time

start = time.time()

with open(os.path.join('./instances/da2', 'init_decisions_backupAllCombi.yml')) as file:
    agent_decisions = yaml.load(file, Loader=yaml.FullLoader)

end = time.time()

print("Time elapsed: {}".format(end - start))

start = end

with open(os.path.join('./instances/da2', 'init_decisions.yml')) as file:
    agent_decisions = yaml.load(file, Loader=yaml.FullLoader)

end = time.time()

print("Time elapsed: {}".format(end - start))

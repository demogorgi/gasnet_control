import yaml
import os
import time

nreps = 10000

start = time.time()

for _ in range(nreps):
    with open(os.path.join('./instances/da2', 'init_decisions_backupAllCombi.yml')) as file:
        agent_decisions = yaml.load(file, Loader=yaml.FullLoader)

end = time.time()

time_elapsed_long = end - start
print(f"Time elapsed: {time_elapsed_long}")
print("Time elapsed per loading: {}".format(time_elapsed_long/nreps))

start = time.time()

for _ in range(nreps):
    with open(os.path.join('./instances/da2', 'init_decisions.yml')) as file:
        agent_decisions = yaml.load(file, Loader=yaml.FullLoader)

end = time.time()

time_elapsed = end - start
print(f"Time elapsed: {time_elapsed}")
print("Time elapsed per loading: {}".format(time_elapsed/nreps))
print(f"Factor of first to second run {time_elapsed_long/time_elapsed}")

import os

for seed in [1, 40, 500, 100, 2019]:
    for task in ['case1', 'task1', 'task2']:
        cmd = 'python main.py --task {:s} --seed {:d}'.format(task, seed)
        os.system(cmd)

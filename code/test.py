import os
import sys
from tools import parse, py_op
args = parse.args


for seed in [1, 40, 500, 100, 2019]:
    for task in ['case1', 'task1', 'task2']:
        cmd = 'python main.py --phase test --final 0 --batch-size 8 --task {:s} --seed {:d} --resume ../data/models/{:s}-snm-{:d}-snr-{:d}-value-{:d}-trend-{:d}-cat-{:d}-lt-{:d}-size-{:d}-seed-{:d}-{:s}'.format(task, seed, task, 
        # cmd = 'python main.py --phase valid --batch-size 8 --task {:s} --seed {:d} --resume ../data/models/{:s}-snm-{:d}-snr-{:d}-value-{:d}-trend-{:d}-cat-{:d}-lt-{:d}-size-{:d}-seed-{:d}-{:s}'.format(task, seed, task, 
            args.split_num, args.split_nor, args.use_value, args.use_trend, 
            args.use_cat, args.last_time, args.embed_size, args.seed, 'best.ckpt')
        print cmd
        os.system(cmd)
        print
        print
        print
    break


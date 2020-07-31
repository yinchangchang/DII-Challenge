# coding=utf8

import os
import sys
import json
sys.path.append('../code')

import tools
from tools import parse, py_op
import numpy as np

args = parse.args

def check_result():
    def get_patient(task):
        if task == 'case1':
            fo = '/home/yin/comparison/onset_case1/onset_case1_training_seed{:d}/'.format(args.seed)
        elif task == 'task1':
            fo = '/home/yin/comparison/onset_case2/onset_case2_training_seed{:d}/'.format(args.seed)
        elif task == 'task2':
            fo = '/home/yin/comparison/mortality/mortality_training_seed{:d}/'.format(args.seed)
        result_csv = os.path.join(fo, 'final_result.csv')
        pids = set()
        for i,line in enumerate(open(result_csv)):
            assert i == len(pids)
            pid = line.split(',')[0]
            pids.add(pid)
        return pids
    test_patient_dict = json.load(open('../file/test_patient_dict.json'))
    for task in ['case1', 'task1', 'task2']:
        test = set(test_patient_dict[task])
        pids = get_patient(task)
        print len(pids), len(test), len(test & pids)
        # if len(test) > len(pids):
        #     print test - pids

check_result()


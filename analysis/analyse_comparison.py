
# coding=utf8

import os
import sys
import json
sys.path.append('../code')

import tools
from tools import parse, py_op
import numpy as np

args = parse.args

# def myprint():
#     pass

def ana_patient():
    fo = '/home/yin/comparison'
    for task in os.listdir(fo):
        print '\n', fo
        task_dir = os.path.join(fo, task)
        task_dir = os.path.join(task_dir, os.listdir(task_dir)[-1])
        for fi in os.listdir(task_dir):
            patients = py_op.myreadjson(os.path.join(task_dir, fi))
            print fi, len(patients)

        
def read_result(task='task2'):
    if task == 'case1':
        fo = '/home/yin/comparison/onset_case1/onset_case1_training_seed{:d}/'.format(args.seed)
    elif task == 'task1':
        fo = '/home/yin/comparison/onset_case2/onset_case2_training_seed{:d}/'.format(args.seed)
    elif task == 'task2':
        fo = '/home/yin/comparison/mortality/mortality_training_seed{:d}/'.format(args.seed)
    test_dict = json.load(open(os.path.join(fo, 'test.json')))
    pset = set()
    for line in open(os.path.join(fo, 'result.csv')):
        p = line.split(',')[0]
        pset.add(p)
    print set(test_dict) - pset

         

    



def main():
    ana_patient()
    read_result()


if __name__ == '__main__':
    os.system('clear')
    main()

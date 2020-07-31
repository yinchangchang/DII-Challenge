# coding=utf8

import os
import sys
import json
import numpy as np
sys.path.append('../code')

import tools
from tools import parse
args = parse.args

# def myprint():
#     pass

def analyze_time(task='task1'):
    '''
    Analyze features of vital data
    '''
    # wf = open(os.path.join(args.result_dir, 'task'))
    wf = open(task + '.csv', 'w')
    def myprint(s):
        wf.write(s + '\n')

    task_dir = os.path.join(args.data_dir, 'sepsis2_{:s}_training'.format(task))


    # vital information
    vital_file = os.path.join(task_dir, 'sepsis2_{:s}_vital_training.csv'.format(task))
    patient_time_dict = dict()
    for i_line,line in enumerate(open(vital_file)):
        if i_line % 10000 == 0:
            print i_line
        if i_line:
            patient, time = line.strip().split(',')[:2]
            patient_time_dict[patient] = max(patient_time_dict.get(patient, -10000000), int(float(time)))
    time_list = patient_time_dict.values()
    print max(time_list), min(time_list)
    x = range(max(time_list) + 1)
    y = [0 for _ in x]
    for t in time_list:
        y[t] += 1
    with open('../file/time_dist.json', 'w') as f:
        f.write(json.dumps([x,y], indent=4))

def draw_time_dist():
    import matplotlib.pyplot as plt
    time_dist = json.load(open('../file/time_dist.json'))
    x,y = time_dist
    for xi,yi in zip(x, y):
        print xi, yi
    plt.plot(x, y)
    plt.savefig('../result/fig/time.png')




def analyze_features(task):
    '''
    Analyze features of vital data
    '''
    # wf = open(os.path.join(args.result_dir, 'task'))
    wf = open(task + '.csv', 'w')
    def myprint(s):
        wf.write(s + '\n')

    task_dir = os.path.join(args.data_dir, 'sepsis2_{:s}_training'.format(task))


    # vital information
    vital_file = os.path.join(task_dir, 'sepsis2_{:s}_vital_training.csv'.format(task))
    vital_dict = { } # key-valuelist-dict
    for i_line,line in enumerate(open(vital_file)):
        if i_line == 0:
            new_line = ''
            vis = 0
            for c in line:
                if c == '"':
                    vis = (vis + 1) % 2
                if vis == 1 and c == ',':
                    c = ';'
                new_line += c
            line = new_line
            col_list = line.strip().split(',')[1:]
            for col in col_list:
                vital_dict[col] = []
        else:
            ctt_list = line.strip().split(',')[1:]
            assert len(ctt_list) == len(col_list)
            for col,ctt in zip(col_list, ctt_list):
                if len(ctt):
                    vital_dict[col].append(float(ctt))
        # if i_line > 10000:
        #    break

    ms_list = []
    myprint('{:s}:\t vital info: \n'.format(task))
    myprint('Feature, Missing Rate, Min, 25%, 75%, Max')
    for col in col_list:
        value_list = sorted(vital_dict[col])
        if len(value_list) == 0:
            continue
        fn = len(value_list) / 4
        myprint('{:s}, {:d}%, {:3.2f}, {:3.2f},{:3.2f},{:3.2f}'.format(col.replace(';', ','), (i_line - len(value_list))*100/i_line, value_list[0], value_list[fn], value_list[fn*3], value_list[-1]))

        ms_list.append((i_line - len(value_list))*100.0/i_line)
    
    ms_list = sorted(ms_list)
    myprint('\nMissing Rate')
    myprint('\nMissing Rate Min: {:d}%'.format(int(ms_list[0])))
    myprint('\nMissing Rate Max: {:d}%'.format(int(ms_list[-1])))
    myprint('\nMissing Rate Mean: {:d}%'.format(int(sum(ms_list)/len(ms_list))))
    

    
        
    



def main():
    # analyze_time()
    draw_time_dist()
    # analyze_features('task1')
    # analyze_features('task2')


if __name__ == '__main__':
    os.system('clear')
    main()

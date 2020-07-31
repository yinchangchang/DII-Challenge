
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

def ana_time(task='task2'):
    if task == 'task2':
        vital_file = '/home/yin/contestdata2/DII_sepsis2_task2_evaluation/sepsis2_task2_evaluation_vital.csv'
    elif task == 'case1':
        vital_file = '/home/yin/contestdata2/DII_sepsis2_task1_evaluation/sepsis2_task1_evaluation_case1_vital.csv'
    else:
        vital_file = '/home/yin/contestdata2/DII_sepsis2_task1_evaluation/sepsis2_task1_evaluation_case2_vital.csv'
    patient_time_dict = dict()
    for line in open(vital_file):
        data = line.split(',')
        patient, time = data[:2]
        if time != 'event_time':
            patient_time_dict[patient] = patient_time_dict.get(patient, []) + [float(time)]
    mx, mn = -100, 100
    for p,ts in patient_time_dict.items():
        if min(ts) > 5:
            print p
        mx = max(mx, min(ts))
        mn = min(mn, max(ts))
    print mx, mn

def ana_patient():
    def get_patients(task):
        if task == 'task2':
            master_file = '/home/yin/contestdata2/DII_sepsis2_task2_evaluation/sepsis2_task2_evaluation_master.csv'
        elif task == 'case1':
            master_file = '/home/yin/contestdata2/DII_sepsis2_task1_evaluation/sepsis2_task1_evaluation_case1_master.csv'
        else:
            master_file = '/home/yin/contestdata2/DII_sepsis2_task1_evaluation/sepsis2_task1_evaluation_case2_master.csv'
        pids = set()
        for i,line in enumerate(open(master_file)):
            if i == 0:
                # print line
                continue
            pid = line.split(',')[0]
            pids.add(pid)
        return pids
    pids_case1 = get_patients('case1')
    pids_case2 = get_patients('case2')
    pids_task2 = get_patients('task2')
    print 'case1', len(pids_case1), len(pids_case1 & pids_case2)
    print 'case2', len(pids_case2)
    print 'task2', len(pids_task2), len(pids_task2 & pids_case2)
    print pids_task2 & pids_case2
    test_patient_dict = {
            'case1': sorted(pids_case1),
            'task1': sorted(pids_case2),
            'task2': sorted(pids_task2)
            }
    py_op.mywritejson(os.path.join(args.file_dir, 'test_patient_dict.json'), test_patient_dict)

def get_patient_line_dict():
    def get_data(task):
        if task == 'task2':
            vital_file = '/home/yin/contestdata2/DII_sepsis2_task2_evaluation/sepsis2_task2_evaluation_vital.csv'
        elif task == 'case1':
            vital_file = '/home/yin/contestdata2/DII_sepsis2_task1_evaluation/sepsis2_task1_evaluation_case1_vital.csv'
        else:
            vital_file = '/home/yin/contestdata2/DII_sepsis2_task1_evaluation/sepsis2_task1_evaluation_case2_vital.csv'
        patient_data = dict()
        for line in open(vital_file):
            if 'event' not in line:
                data = line.strip().split(',')
                patient = data[0]
                line = ','.join(data[2:])
                patient_data[patient] = patient_data.get(patient, []) + [line]
        for p, d in patient_data.items():
            if len(d) < 4:
                print task, p, len(d)
        return patient_data
    task_patient_data = dict()
    for k in ['case1', 'case2', 'task2']:
        print k
        task_patient_data[k] = get_data(k)
    print 'write'
    with open('../result/task_patient_data.json', 'w') as f:
        f.write(json.dumps(task_patient_data))

        
def ana_data_similar():
    def get_master(task):
        if task == 'task2':
            master_file = '/home/yin/contestdata2/DII_sepsis2_task2_evaluation/sepsis2_task2_evaluation_master.csv'
        elif task == 'case1':
            master_file = '/home/yin/contestdata2/DII_sepsis2_task1_evaluation/sepsis2_task1_evaluation_case1_master.csv'
        else:
            master_file = '/home/yin/contestdata2/DII_sepsis2_task1_evaluation/sepsis2_task1_evaluation_case2_master.csv'
        master_pid_dict = dict()
        for i,line in enumerate(open(master_file)):
            if i == 0:
                continue
            pid = line.split(',')[0]
            # master = line.replace(pid+',', '')
            master = line[len(pid) + 1:]
            master = ''.join(master.split())
            master_pid_dict[master] = master_pid_dict.get(master, []) + [pid]
        return master_pid_dict
    task_master_pid_dict = dict()
    task_patient_data = py_op.myreadjson('../result/task_patient_data.json')
    for k in ['case1', 'case2', 'task2']:
        task_master_pid_dict[k] = get_master(k)

    kf = 'case1'
    ks = 'task2'
    ks = 'case2'
    master_set = set(task_master_pid_dict[kf]) & set(task_master_pid_dict[ks])
    cset = set()
    n = 0
    for master in master_set:
        pc = task_master_pid_dict[kf][master]
        pt = task_master_pid_dict[ks][master]
        if len(pc) + len(pt) >= 2:
            for ppc in pc:
                n += 1
                for ppt in pt:
                    ppc_data = set(task_patient_data[kf][ppc])
                    ppt_data = set(task_patient_data[ks][ppt])
                    same = 0
                    for cline in ppc_data:
                        for tline in ppt_data:
                            if cline == tline:
                                # print ppc, ppt
                                # cset.add(ppc)
                                # print cline
                                # print tline
                                same += 1
                    if same > 5:
                        print same, len(ppc_data), len(ppt_data)
                        cset.add(ppc)
    print len(cset), n

def main():
    # ana_time('case1')
    ana_patient()
    # get_patient_line_dict()
    # ana_data_similar()


if __name__ == '__main__':
    os.system('clear')
    main()

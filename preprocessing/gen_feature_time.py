# coding=utf8

import os
import sys
sys.path.append('../code')

import tools
from tools import parse, py_op
args = parse.args



def gen_patient_time_dict():
    vital_file = args.vital_file
    patient_time_dict = dict()
    for i_line,line in enumerate(open(vital_file)):
        if 'event_time' not in line:
            patient, time = line.strip().split(',')[:2]
            patient_time_dict[patient] = max(patient_time_dict.get(patient, 0), float(time))
    py_op.mywritejson(os.path.join(args.result_dir, 'patient_time_dict.json'), patient_time_dict)



def main():
    gen_patient_time_dict()





if __name__ == '__main__':
    main()

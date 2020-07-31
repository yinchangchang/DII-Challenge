# coding=utf8

import os
import sys
import json
sys.path.append('../code')

import tools
from tools import parse, py_op
args = parse.args




def gen_patient_label_dict():
    patient_label_dict = dict()
    label_file = args.label_file
    for i_line,line in enumerate(open(label_file)):
        if i_line != 0:
            data = line.strip().split(',')
            patient = data[0]
            label  = data[-1]
            patient_label_dict[patient] = int(label)
    py_op.mywritejson(os.path.join(args.result_dir, 'patient_label_dict.json'), patient_label_dict)






def main():
    gen_patient_label_dict()





if __name__ == '__main__':
    main()

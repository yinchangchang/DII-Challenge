# coding=utf8

import os
import sys
import json
sys.path.append('../code')

import tools
from tools import parse, py_op
args = parse.args



def analyse_variation_trend(task='task1'):
    '''
    generate new vital file
    '''
    feature_variation_trend_dict = dict()

    feature_value_order_dict = py_op.myreadjson(os.path.join(args.file_dir, 'feature_value_order_dict.{:s}.json'.format(task)))

    patient_time_dict = py_op.myreadjson(os.path.join(args.file_dir, 'patient_time_dict.json'))

    task_dir = os.path.join(args.data_dir, 'sepsis2_{:s}_training'.format(task))
    vital_file = os.path.join(task_dir, 'sepsis2_{:s}_vital_training.csv'.format(task))
    vital_dict = { } # key-valuelist-dict

    last_patient = ''
    feature_time_value_dict = dict()
    for i_line,line in enumerate(open(vital_file)):
        if i_line % 10000 == 0:
            print i_line
        if i_line:
            ctt_list = line.strip().split(',')[2:]
            new_ctt = line.strip().split(',')[:2]
            if task == 'task1':
                patient, time = new_ctt
                new_time = float(time) - patient_time_dict[patient] - 4.0
                new_ctt = [patient, str(new_time)]

            patient, time = new_ctt
            time = int(float(time))

            if patient != last_patient:
                for feature, tv in feature_time_value_dict.items():
                    if len(tv) > 4:
                        ts = sorted(tv.keys())
                        vs = [tv[t] for t in ts]
                        feature_variation_trend_dict[feature] = feature_variation_trend_dict.get(feature, []) + [[ts, vs]]
                if i_line >= 500000:
                    break

                feature_time_value_dict = dict()
                last_patient = patient

            for idx, value in enumerate(ctt_list):
                if len(value.strip()):
                    value = float(value.strip())
                    if idx not in feature_time_value_dict:
                        feature_time_value_dict[idx] = { }
                    feature_time_value_dict[idx][time] = value



    # py_op.mywritejson(os.path.join(args.file_dir, 'feature_variation_trend_dict.json'), feature_variation_trend_dict)
    with open (os.path.join(args.file_dir, 'feature_variation_trend_dict.json'), 'w') as f:
        f.write(json.dumps(feature_variation_trend_dict))





def draw_pic():
    import numpy as np
    import matplotlib.pyplot as plt
    flc = np.load('../file/feature_label_count.npy')
    fvt = py_op.myreadjson(os.path.join(args.file_dir, 'feature_variation_trend_dict.json'))

    for f in range(143):
        vt = fvt[str(f)]
        print vt
        for i, (t, v) in enumerate(vt):
            plt.plot(t,v)
            if i > 10:
                break
        plt.savefig('../result/variation_trend/{:d}.png'.format(f))
        plt.clf()
                
            


def main():
    # analyse_variation_trend()
    draw_pic()
    pass





if __name__ == '__main__':
    os.system('clear')
    main()

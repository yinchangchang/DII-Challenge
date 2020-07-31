# coding=utf8

import os
import sys
sys.path.append('../code')

import tools
from tools import parse, py_op
import numpy as np


args = parse.args

# def myprint():
#     pass

def ana_feat_dist(task):
    n_split = 100
    feature_label_count = np.zeros((143, 2, n_split))
    patient_time_record_dict = py_op.myreadjson(os.path.join(args.result_dir, 'json_data', '{:s}.json'.format(args.task)))
    patient_label_dict = py_op.myreadjson(os.path.join(args.file_dir, 'patient_label_dict.{:s}.json'.format(args.task)))
    [ [ [0. for _ in range(n_split)], [0. for _ in range(n_split)] ] for i in range(143) ]
    for ip, (p, t_dict) in enumerate(patient_time_record_dict.items()):
        if ip % 10000 == 0:
            print ip, len(patient_time_record_dict)

        label = patient_label_dict[p]
        for t, vs in t_dict.items():
            for v in vs:
                feature, value = v
                idx = int(value * n_split)
                feature_label_count[feature, label, idx] += 1
    for f in range(143):
        for l in range(2):
            feature_label_count[feature, label] /= feature_label_count[feature, label].sum()
    np.save('../file/feature_label_count.npy', feature_label_count)
         


def draw_pic():
    def avg(ys, n = 50):
        nys = []
        for i,y in enumerate(ys):
            st = max(0, i - n)
            en = min(len(ys), i + n + 1) 
            nys.append(np.mean(ys[st:en]))

        return nys

    import matplotlib.pyplot as plt
    flc = np.load('../file/feature_label_count.npy')
    for f in range(143):
        lc = flc[f]
        x = range(len(lc[0]))
        plt.plot(x,avg(lc[0]),'b')
        plt.plot(x,avg(lc[1]),'r')
        plt.savefig('../result/fig/{:d}.png'.format(f))
        plt.clf()
        if f > 10:
            break


    



def main():
    # analyze_features('task1')
    # ana_feat_dist('task1')
    draw_pic()


if __name__ == '__main__':
    os.system('clear')
    main()

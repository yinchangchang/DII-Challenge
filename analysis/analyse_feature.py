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
    feature_count_dict = py_op.myreadjson(os.path.join(args.file_dir, 'feature_count_dict.{:s}.json'.format(args.task)))
    normal_range_order_dict = py_op.myreadjson(os.path.join(args.file_dir, 'normal_range_order_dict.{:s}.json'.format(args.task)))
    feature_index_dict = py_op.myreadjson(os.path.join(args.file_dir, 'feature_index_dict.json'.format(args.task)))
    cnt_list = []
    print normal_range_order_dict.keys()
    for k,c in feature_count_dict.items():
        if c > 1 and 'event_time' != k:
            idx = feature_index_dict[k]
            if str(idx) not in normal_range_order_dict:
                continue
            mn, mx = normal_range_order_dict[str(idx)]
            print mn, mx
            a = int(mn * c)
            b = int((mx - mn) * c)
            c = int((1 - mx) * c)
            cnt_list += [a, b, c]
    print sorted(cnt_list)
    print len(cnt_list)

        
         

    



def main():
    ana_feat_dist('task2')


if __name__ == '__main__':
    os.system('clear')
    main()

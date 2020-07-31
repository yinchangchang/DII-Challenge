# coding=utf8

import os
import sys
sys.path.append('../code')

import tools
from tools import parse, py_op
args = parse.args


def gen_feature_order_dict():
    '''
    generate the order of value for each feature
    '''

    feature_value_order_dict = dict()

    # vital information
    vital_file = args.vital_file
    vital_dict = { } # key-valuelist-dict
    for i_line,line in enumerate(open(vital_file)):
        if i_line % 10000 == 0:
            print i_line
        # if i_line > 10000:
        #     break
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
        # if i_line % 10000 == 0:
        #     print i_line


    # add group info
    groups = py_op.myreadjson(os.path.join(args.file_dir, 'similar.json'))
    feature_index_dict = py_op.myreadjson(os.path.join(args.file_dir, 'feature_index_dict.json'))
    index_feature_list = py_op.myreadjson(os.path.join(args.file_dir, 'index_feature_list.json'))
    for g in groups:
        for k in g:
            mg = min(g)
            if k != mg:
                kf = index_feature_list[k]
                mf = index_feature_list[mg]
                vital_dict[mf] = vital_dict[mf] + vital_dict[kf]
                vital_dict.pop(kf)
    print 'features', len(vital_dict)

    # feature_count_dict = { k: len(v) for k,v in vital_dict.items() }
    # py_op.mywritejson(os.path.join(args.file_dir, 'feature_count_dict.json'), feature_count_dict)



    ms_list = []
    for col in col_list:
        if col not in vital_dict:
            continue
        value_list = sorted(vital_dict[col])
        value_order_dict = dict()
        value_minorder_dict = dict()
        value_maxorder_dict = dict()
        for i_value, value in enumerate(value_list):
            if value not in value_minorder_dict:
                value_minorder_dict[value] = i_value
            if value == value_list[-1]:
                value_maxorder_dict[value] = len(value_list) - 1
                break
            if value != value_list[i_value+1]:
                value_maxorder_dict[value] = i_value
        for value in value_maxorder_dict:
            value_order_dict[value] = (value_maxorder_dict[value] + value_minorder_dict[value]) / 2.0 / len(value_list)
        feature_value_order_dict[col] = value_order_dict
    py_op.mywritejson(os.path.join(args.file_dir, 'feature_value_order_dict.json'), feature_value_order_dict)


def main():
    gen_feature_order_dict()





if __name__ == '__main__':
    os.system('clear')
    main()

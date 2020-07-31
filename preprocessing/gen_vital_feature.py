# coding=utf8

import os
import sys
import json
sys.path.append('../code')

import tools
from tools import parse, py_op
args = parse.args


def gen_json_data():
    vital_file = args.vital_file
    patient_time_record_dict = dict()
    feature_index_dict = py_op.myreadjson(os.path.join(args.file_dir, 'feature_index_dict.json'))
    feature_value_order_dict = py_op.myreadjson(os.path.join(args.file_dir, 'feature_value_order_dict.json'))
    feature_value_order_dict = { str(feature_index_dict[k]):v for k,v in feature_value_order_dict.items()  if 'event' not in k}
    index_group_dict = py_op.myreadjson(os.path.join(args.file_dir, 'index_group_dict.json'))
    patient_time_dict = py_op.myreadjson(os.path.join(args.result_dir, 'patient_time_dict.json'))
    mx_time = - 100
    for i_line, line in enumerate(open(vital_file)):
        if i_line % 10000 == 0:
            print 'line', i_line
        if 'event_time' not in line:
            data = line.strip().split(',')
            patient, time = data[:2]
            time = int(float(time))
            mx_time = max(mx_time, time)
            if patient not in patient_time_record_dict:
                patient_time_record_dict[patient] = dict()
            if time not in patient_time_record_dict[patient]:
                patient_time_record_dict[patient][time] = dict()

            data = data[2:]
            vs = dict()
            for idx, val in enumerate(data):
                if len(val) == 0:
                    continue
                if str(idx) in index_group_dict:
                    idx = index_group_dict[str(idx)]
                value_order = feature_value_order_dict[str(idx)]
                vs[idx] = value_order[val]
            patient_time_record_dict[patient][time].update(vs)

    new_d = dict()
    for p, tr in patient_time_record_dict.items():
        new_d[p] = dict()
        for t, vs in tr.items():
            if mx_time > 0:
                t = int(t - patient_time_dict[p] - 4)
            if t < - 102:
                continue
            nvs = []
            for k in sorted(vs.keys()):
                nvs.append([k, vs[k]])
            new_d[p][t] = nvs
    with open(os.path.join(args.result_dir, 'patient_time_record_dict.json'), 'w') as f:
        # f.write(json.dumps(new_d, indent=4))
        f.write(json.dumps(new_d))



def main():
    gen_json_data()





if __name__ == '__main__':
    main()

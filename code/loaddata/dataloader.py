# encoding: utf-8

"""
Read images and corresponding labels.
"""

import numpy as np
import os
import sys
import json
import torch
from torch.utils.data import Dataset

sys.path.append('loaddata')
import data_function


class DataSet(Dataset):
    def __init__(self, 
            patient_list, 
            patient_time_record_dict, 
            patient_label_dict,
            patient_master_dict,
            phase='train',          # phase
            split_num=5,            # split feature value into different parts
            args=None               # 全局参数
            ):

        self.patient_list = patient_list
        self.patient_time_record_dict = patient_time_record_dict
        self.patient_label_dict = patient_label_dict
        self.patient_master_dict = patient_master_dict
        self.phase = phase
        self.split_num = split_num
        self.split_nor = args.split_nor
        self.split_nn = args.split_nn
        self.args = args
        if args.task == 'task2':
            self.length = 49
        else:
            self.length = 98


    def get_visit_info(self, time_record_dict):
        # times = sorted([float(t) for t in time_record_dict.keys()])
        times = sorted(time_record_dict.keys(), key=lambda s:float(s))
        # for t in time_record_dict:
        #     time_record_dict[str(float(t))] = time_record_dict[t]
        visit_list = []
        value_list = []
        mask_list = []
        time_list = []

        n_code = 72
        import traceback

        # trend
        trend_list = []
        previous_value = [[[],[]] for _ in range(143)]
        change_th = 0.02
        start_time = - self.args.avg_time * 2
        end_time = -1

        for time in times :
            if float(time) <= -4 - self.length:
                continue
            if self.args.task == 'task2':
                if float(time) > self.args.last_time:
                    continue
            time = str(time)
            records = time_record_dict[time]
            feature_index = [r[0] for r in records]
            feature_value = [float(r[1]) for r in records]

            # embed feature value
            feature_index = np.array(feature_index)
            feature_value = np.array(feature_value)
            feature = feature_index * self.split_nn + feature_value * self.split_num

            # trend
            trend = np.zeros(n_code, dtype=np.int64)
            i_v = 0
            for idx, val in zip(feature_index, feature_value):
                # delete val with time less than start_time
                ptimes = previous_value[idx][0]
                lip = 0
                for ip, pt in enumerate(ptimes):
                    if pt >= float(time) + start_time:
                        lip = ip
                        break

                avg_val = None
                if len(previous_value[idx][0]) == 1:
                    avg_val = previous_value[idx][1][-1]

                previous_value[idx] = [
                        previous_value[idx][0][lip:],
                        previous_value[idx][1][lip:]]

                # trend value
                if len(previous_value[idx][0]):
                    avg_val = np.mean(previous_value[idx][1])
                if avg_val is not None:
                    if val < avg_val - change_th:
                        delta = 0
                    elif val > avg_val + change_th:
                        delta = 1
                    else:
                        delta = 2
                    trend[i_v] = idx * 3 + delta + 1

                # add new val
                previous_value[idx][0].append(float(time))
                previous_value[idx][1].append(float(val))

                i_v += 1





            visit = np.zeros(n_code, dtype=np.int64)
            mask = np.zeros(n_code, dtype=np.int64)
            i_v = 0
            for feat, idx, val in zip(feature, feature_index,  feature_value):

                # order
                mask[i_v] = 1
                visit[i_v] = int(feat + 1)
                i_v += 1


                    

            value = np.zeros((2, n_code ), dtype=np.int64)
            value[0][: len(feature_index)] = feature_index + 1
            value[1][: len(feature_index)] = (feature_value * 100).astype(np.int64)
            value_list.append(value)

            visit_list.append(visit)
            mask_list.append(mask)
            time_list.append(float(time))
            trend_list.append(trend)

        if self.args.task == 'task2':
            num_len = self.length + self.args.last_time
            # print 'task2', num_len, self.args.last_time
        else:
            num_len = self.length 
            # print 'task1'
        # print 'num_len', num_len
        # print len(visit_list)
        assert len(visit_list) <= num_len
        visit = np.zeros(n_code, dtype=np.int64)
        trend = np.zeros(n_code, dtype=np.int64)
        value = np.zeros((2, n_code), dtype=np.int64)
        while len(visit_list) < num_len:
            visit_list.append(visit)
            value_list.append(value)
            mask_list.append(visit)
            time_list.append(0)
            trend_list.append(trend)

        return np.array(visit_list), np.array(value_list), np.array(mask_list, dtype=np.float32), np.array(time_list, dtype=np.float32), np.array(trend_list)




    def __getitem__(self, index):
        patient = self.patient_list[index]
        if self.args.use_visit:
            visit_list, value_list, mask_list, time_list, trend_list= self.get_visit_info(self.patient_time_record_dict[patient])
            master = self.patient_master_dict[patient]
            master = [int(m) for m in master]
            master = np.float32(master)
            if self.args.final == 1:
                label = np.float32(0)
            else:
                label = np.float32(self.patient_label_dict[patient])
            if self.phase == 'test':
                return visit_list, value_list, mask_list, master, label, time_list, trend_list, patient
            else:
                return visit_list, value_list, mask_list, master, label, time_list, trend_list




    def __len__(self):
        return len(self.patient_list) 

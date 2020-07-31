#########################################################################
# File Name: run.sh
# Author: ccyin
# mail: ccyin04@gmail.com
# Created Time: Thu 12 Sep 2019 06:46:05 AM UTC
#########################################################################
#!/bin/bash

# 	task1
# python run.py --label-file /home/yin/data/sepsis2_task1_training/sepsis2_task1_label_training.csv   --vital-file /home/yin/data/sepsis2_task1_training/sepsis2_task1_vital_training.csv	--master-file  /home/yin/data/sepsis2_task1_training/sepsis2_task1_master_training.csv --task task1

# 	task2
python run.py --label-file /home/yin/data/sepsis2_task2_training/sepsis2_task2_label_training.csv   --vital-file /home/yin/data/sepsis2_task2_training/sepsis2_task2_vital_training.csv	--master-file  /home/yin/data/sepsis2_task2_training/sepsis2_task2_master_training.csv --task task2

# cd code
# python main.py --task task1



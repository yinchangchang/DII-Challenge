
import os
import sys
sys.path.append('code/tools')
import parse
args = parse.args



os.system('mkdir result')
os.system('mkdir data')
os.system('mkdir data/models')

cmd = '''
		cd preprocessing
                python gen_master_feature.py --master-file {:s}
'''.format(args.master_file)
# python gen_master_feature.py --master-file ../file/master.csv
print cmd
os.system(cmd)

cmd = '''
		cd preprocessing
                python gen_feature_time.py --vital-file {:s}
'''.format(args.vital_file)
# python gen_feature_time.py --vital-file ../file/vital.csv
print cmd
os.system(cmd)

cmd = '''
		cd preprocessing
		python gen_feature_order.py  --vital-file {:s}
'''.format(args.vital_file)
# python gen_vital_feature.py --vital-file ../file/vital.csv
print cmd
os.system(cmd)



cmd = '''
		cd preprocessing
		python gen_vital_feature.py  --vital-file {:s}
'''.format(args.vital_file)
# python gen_vital_feature.py --vital-file ../file/vital.csv
print cmd
os.system(cmd)


cmd = '''
		cd preprocessing
		python gen_label.py --label-file {:s}
'''.format(args.label_file)
# python gen_label.py --label-file ../file/label.csv
print cmd
os.system(cmd)

cmd = '''
		cd code
		python main.py --task {:s}
'''.format(args.task)
print cmd
os.system(cmd)
		


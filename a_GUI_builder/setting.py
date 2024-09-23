# base = 'G:/hover_net-master/hover_net-master/datasets/data_test_1/'
base = 'temp'

path = base
path_overlay = base + '/infer/overlay'
input = base
output = base + 'infer'
import sys
# PYTHON = 'D:\pytorch\Anaconda3\envs\hovernet\python.exe'
# infer = 'G:/hover_net-master/hover_net-master/run_infer.py'
# type_info_path = 'G:/hover_net-master/hover_net-master/type_info.json'
# model_path = 'G:/hover_net-master/hover_net-master/pretrained/hovernet_original_consep_notype_tf2pytorch.tar'

PYTHON = sys.executable
infer = '../run_infer.py'
type_info_path = '../type_info.json'
model_path = '../pretrained/hovernet_original_consep_notype_tf2pytorch.tar'

# path = 'G:/hover_net-master/hover_net-master/datasets/data_test_1'
# path_overlay = 'G:/hover_net-master/hover_net-master/datasets/data_test_1/infer/overlay'
# input = 'G:/hover_net-master/hover_net-master/datasets/data_test_1'
# output = 'G:/hover_net-master/hover_net-master/datasets/data_test_1/infer'
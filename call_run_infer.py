import subprocess
import run_infer
from a_GUI_builder.setting import input, output, PYTHON, infer, type_info_path, model_path

def only_classification():
    # python = 'D:\pytorch\Anaconda3\envs\hovernet\python.exe'
    python = PYTHON
    cmd = [python,
           # 'G:/hover_net-master/hover_net-master/run_infer.py',
           infer,
           '--nr_types=0',
           # '--type_info_path=G:/hover_net-master/hover_net-master/type_info.json',
           f'--type_info_path={type_info_path}',
           '--batch_size=4',
           '--model_mode=original',
           # '--model_path=G:/hover_net-master/hover_net-master/pretrained/hovernet_original_consep_notype_tf2pytorch.tar',
           f'--model_path={model_path}',
           '--nr_inference_workers=4',
           '--nr_post_proc_workers=4',
           'tile',
           # '--input_dir=G:/hover_net-master/hover_net-master/datasets/data_test_1',  # 程序运行过程中处理图像的位置
           # '--output_dir=G:/hover_net-master/hover_net-master/datasets/data_test_1/infer',
           f'--input_dir={input}',
           f'--output_dir={output}',
           '--mem_usage=0.2',
           '--draw_dot',
           '--save_qupath',
           '--save_raw_map'
           ]
    try:
        # 使用 subprocess.run 来调用 run.py 并传递参数
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print("run.py was executed successfully.")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Failed to execute run_infer.py.")
        print(f'e.stderr: {e.stderr}')
        print(f'e.stdout: {e.stdout}')


def classification_and_segmentation():
    python = 'D:\pytorch\Anaconda3\envs\hovernet\python.exe'
    cmd = [python,
           'G:/hover_net-master/hover_net-master/run_infer.py',
           '--nr_types=5',
           '--type_info_path=G:/hover_net-master/hover_net-master/type_info.json',
           '--batch_size=4',
           '--model_mode=original',
           '--model_path=G:/hover_net-master/hover_net-master/pretrained/hovernet_original_consep_type_tf2pytorch.tar',
           '--nr_inference_workers=4',
           '--nr_post_proc_workers=4',
           'tile',
           # '--input_dir=G:/hover_net-master/hover_net-master/datasets/data_test_1',
           # '--output_dir=G:/hover_net-master/hover_net-master/datasets/data_test_1/infer',
           f'--input_dir={input}',
           f'--output_dir={output}',
           '--mem_usage=0.2',
           '--draw_dot',
           '--save_qupath',
           '--save_raw_map'
           ]
    try:
        # 使用 subprocess.run 来调用 run.py 并传递参数
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print("run.py was executed successfully.")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Failed to execute run_infer.py.")
        print(f'e.stderr: {e.stderr}')
        print(f'e.stdout: {e.stdout}')
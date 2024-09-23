import tkinter as tk
import os
import shutil
from tkinter import filedialog, Toplevel
from call_run_infer import classification_and_segmentation, only_classification



def clear_folder(path):
    print("CLEARING FOLDER")
    # =============================================================================================创建一个暂存图片的文件夹
    # path = 'G:/hover_net-master/hover_net-master/datasets/data_test_1'  # +++++++++++++++++++++++++++++++++++++++++++++
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    print("CLEARING FOLDER OVER")


def upload_move_image(src_img, path, t):
    """ 将 src_folder 中的所有图像复制到 dst_folder """
    # path = 'G:/hover_net-master/hover_net-master/datasets/data_test_1'  # ==============================================
    if not os.path.exists(path):
        os.makedirs(path)

    directory, filename = os.path.split(src_img)
    name, ext = os.path.splitext(filename)
    new_name = name + '-' + t + ext
    destination_file = os.path.join(path, new_name)


    # destination_file = os.path.join(path, os.path.basename(src_img))
    shutil.copy(src_img, destination_file)
    print(f'move successfully {destination_file}')

def download_move_image(des_img, path_overlay):
    # path = 'G:/hover_net-master/hover_net-master/datasets/data_test_1/infer/overlay'  # 暂存overlay图像的文件夹
    print("DOWNLOADING IMAGE")
    if os.path.exists(path_overlay):
        print("path_overlay exists")
        os.makedirs(des_img, exist_ok=True)
        for filename in os.listdir(path_overlay):
            # 检查文件是否为图像
            if filename.lower().endswith((".png")):
                source_path = os.path.join(path_overlay, filename)
                destination_path = os.path.join(des_img, filename)
                # 移动文件
                shutil.copy2(source_path, destination_path)
                print(f"Moved: {source_path} to {destination_path}")
    else:
        print(f"No such file or directory: {path_overlay}")


def open_img(files, path, t):  # 选择自己电脑中需要处理的图像
    file_type = [("Image Files", "*.png")]
    file_path = filedialog.askopenfilename(title="Open file", filetypes=file_type)
    if file_path:

        print("selected file: ", file_path)
        files.append(file_path)
        upload_move_image(file_path, path, t)
        print("files in function open_img: ", files)

    else:
        print("No file selected")


def open_new_toplevel(root):  # 打开一个window的标准模板
    new_window = Toplevel(root)
    new_window.title('CoNSeP')
    new_window.geometry('500x150+100+330')
    new_window.lift()  # 将窗口提升到主窗口前面
    # new_window.attributes('-topmost', False)  # 确保它不会总是置顶
    # new_window.attributes('-topmost', 0)
    return new_window


def save_img(window, root, path_overlay):  # 选择将图像存储在电脑中的位置
    print("save image")
    folder_selected = filedialog.askdirectory(title="请选择图片保存在哪个文件夹下面")
    if folder_selected:
        print(f"Selected folder: {folder_selected}")
        download_move_image(folder_selected, path_overlay)
        root.destroy()

        # 这里可以存储或者进一步使用选中的文件夹路径
def show_error(window):  # 没有选择图片 或者 暂存图片的文件夹不存在
    window1 = open_new_toplevel(window)
    label = tk.Label(window1, text='未选择图片，请选择图片')
    label.pack(padx=10, pady=10)



def confirm_1(files, window, root, path, path_overlay):  # classification only 的确认键
    # fold = 'G:/hover_net-master/hover_net-master/datasets/data_test_1'
    # path_overlay = 'G:/hover_net-master/hover_net-master/datasets/data_test_1/infer/overlay'
    print("len(os.listdir(fold)) = ", len(os.listdir(path)))
    print("os.path.isdir(fold) = ", os.path.isdir(path))
    if os.path.isdir(path) | len(os.listdir(path)) == 0:
        show_error(window)
        return
    only_classification()
    window.destroy()
    window_save = open_new_toplevel(root)
    label = tk.Label(window_save, text=f'请选择结果保存位置')
    label.pack()
    button_save = tk.Button(window_save, text='save file', command=lambda: save_img(window_save, root, path_overlay))
    button_save.pack()


def confirm_2(files, window, root, path, path_overlay):  # classification and segmentation 的确认键
    # fold = 'G:/hover_net-master/hover_net-master/datasets/data_test_1'
    # path_overlay = 'G:/hover_net-master/hover_net-master/datasets/data_test_1/infer/overlay'
    if os.path.isdir(path) | len(os.listdir(path)) == 0:
        show_error(window)
        return
    classification_and_segmentation()
    window.destroy()
    window_save = open_new_toplevel(root)
    label = tk.Label(window_save, text=f'请选择结果保存位置')
    label.pack()
    button_save = tk.Button(window_save, text='save file', command=lambda: save_img(window_save, root, path_overlay))
    button_save.pack()


def consep_window_1(root, path, path_overlay):  # classification only  ConSeP点击进入该函数
    t = "classification_only"
    # path = 'G:/hover_net-master/hover_net-master/datasets/data_test_1'
    # path_overlay = 'G:/hover_net-master/hover_net-master/datasets/data_test_1/infer/overlay'
    clear_folder(path)   # 删除存放图像的文件夹
    window = open_new_toplevel(root)  # 按照模板新建一个window框框
    files = []
    button_img = tk.Button(window, text="select image", command=lambda: open_img(files, path, t))
    button_img.pack(side=tk.TOP, pady=20, padx=70)
    print("files in function consep_window_1: ", files)

    # button_save = tk.Button(window, text='save_dir')
    # button_save.pack(side=tk.RIGHT, pady=10, padx=(70, 80))

    button_sure = tk.Button(window, text='confirm', command=lambda: confirm_1(files, window, root, path, path_overlay))
    button_sure.pack(side=tk.BOTTOM, pady=20)


def consep_window_2(root, path, path_overlay):  # classification and segmentation  ConSeP点击进入该函数
    t = "classification_and_segmentation"
    # path = 'G:/hover_net-master/hover_net-master/datasets/data_test_1'
    # path_overlay = 'G:/hover_net-master/hover_net-master/datasets/data_test_1/infer/overlay'
    clear_folder(path)
    window = open_new_toplevel(root)
    files = []
    button_img = tk.Button(window, text="select image", command=lambda: open_img(files, path, t))
    button_img.pack(side=tk.TOP, pady=10, padx=70)

    # button_save = tk.Button(window, text='save_dir')
    # button_save.pack(side=tk.RIGHT, pady=10, padx=(70, 80))

    button_sure = tk.Button(window, text='confirm', command=lambda: confirm_2(files, window, root, path, path_overlay))
    button_sure.pack(side=tk.BOTTOM, pady=20)




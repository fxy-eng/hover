import tkinter as tk
from PIL import Image, ImageTk
from input import consep_window_1, consep_window_2
from setting import path, path_overlay


# path = 'G:/hover_net-master/hover_net-master/datasets/data_test_1'
# path_overlay = 'G:/hover_net-master/hover_net-master/datasets/data_test_1/infer/overlay'

root = tk.Tk()
root.title('cell classification and segmentation')
root.geometry('500x550+100+100')

# ---------------------------------------------------------------------------------------------------------------------
# 上边的大框框
frame1 = tk.LabelFrame(root, height=200, width=480, text='classification only')
frame1.pack_propagate(False)  # 阻止LabelFrame根据内容自动调整大小
frame1.pack(pady=(80, 15))
label1 = tk.Label(frame1, text='请选择要分割的类型')
label1.pack(pady=10, fill=tk.X)
# ---------------------------------------------------------------------------------------------------------------------

# Image1_1  上边的图像
image1_1_path = 'classification_only.png'  # ++++++++++++++++++++++++++++++++++++++++++++++++++++++
image1_1 = Image.open(image1_1_path)
image1_1 = image1_1.resize((70, 70), Image.ANTIALIAS)
photo1_1 = ImageTk.PhotoImage(image1_1)
image_label1_1 = tk.Label(frame1, image=photo1_1)
image_label1_1.pack(side=tk.TOP, fill='both', expand=True, padx=30, pady=10)

# button1_1  classification 中的 ConSeP 按键
button1_1 = tk.Button(frame1, text='ConSeP', command=lambda: consep_window_1(root, path, path_overlay))
button1_1.pack(side=tk.TOP, padx=30, pady=10)


# ---------------------------------------------------------------------------------------------------------------------
# 下边的大框框
frame2 = tk.LabelFrame(root, height=200, width=480, text='classification and segmentation')
frame2.pack_propagate(False)  # 阻止LabelFrame根据内容自动调整大小
frame2.pack(pady=15)
label2 = tk.Label(frame2, text='请选择要分割和分类的类型')
label2.pack(pady=10, fill=tk.X)
# ---------------------------------------------------------------------------------------------------------------------

# Image2_1  下边的图像
image2_1_path = 'classification_and_segmentation.png'  # ++++++++++++++++++++++++++++++++++++++++++++++++++++++
image2_1 = Image.open(image2_1_path)
image2_1 = image2_1.resize((70, 70), Image.ANTIALIAS)
photo2_1 = ImageTk.PhotoImage(image2_1)
image_label2_1 = tk.Label(frame2, image=photo2_1)
image_label2_1.pack(side=tk.TOP, fill='both', expand=True, padx=30, pady=10)

# button2_1  classification and segmentation 中的ConSeP 按键
button2_1 = tk.Button(frame2, text='ConSeP', command=lambda: consep_window_2(root, path, path_overlay))
button2_1.pack(side=tk.TOP, padx=30, pady=10)



root.mainloop()
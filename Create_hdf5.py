'''
生成HDF5文件

'''
import h5py
import cv2
import os
import random
import numpy as np
# -------------------------------------------------createVallist-----------------------------------------------
"""
data:2018.8.6
function name：createVallist()
function: genrate validation list
paramater：
          1.txt_dir: txt file path
          2val_txt_dir:
          2.valRate: validation rate        
return: none
import random
"""
def createVallist(txt_dir, val_txt_dir,valRate):
    with open(txt_dir, 'r') as txtfile:
        lines = txtfile.readlines()
    line_number = len(lines)
    copy_number = int(line_number * valRate)
    val = open(val_txt_dir, 'w')
    for i in range(copy_number):
        txt = random.choice(lines)
        val.write(txt)
    val.close()
    print("validation list create successfully")
# -------------------------------------------------createHdf5Data-----------------------------------------------
'''
__auhor__="syy"
data:2018.8.6
function name：createHdf5Data()
function: genrate hdf5 for caffe
paramater：
          1.img_dir: image file path
          2.txt_dir: txt list path
          3.hdf5_dir: hdf5 output path 
          4.img_img_size: input img_size of image
return: none
'''
def createHdf5Data(img_dir, txt_dir,hdf5_dir,img_size):
    print("开始数转换")
    with open(txt_dir, 'r') as txt_file:
        lines = txt_file.readlines()
    txt_list_number = len(lines)
    print("txt_list_number", txt_list_number)

    datas = np.zeros((txt_list_number, 1, img_size, img_size))
    labels = np.zeros((txt_list_number, 14))
    key_point = np.zeros(14, dtype=float)

    for i, line in enumerate(lines):
        new_line = line.split()
        pic_name = new_line[0]
        x1 = int(new_line[1])
        y1 = int(new_line[2])
        x2 = int(new_line[3])
        y2 = int(new_line[4])
        key_point[0] = float(new_line[5])
        key_point[1] = float(new_line[6])
        key_point[2] = float(new_line[7])
        key_point[3] = float(new_line[8])
        key_point[4] = float(new_line[9])
        key_point[5] = float(new_line[10])
        key_point[6] = float(new_line[11])
        key_point[7] = float(new_line[12])
        key_point[8] = float(new_line[13])
        key_point[9] = float(new_line[14])
        key_point[10] = float(new_line[15])
        key_point[11] = float(new_line[16])
        key_point[12] = float(new_line[17])
        key_point[13] = float(new_line[18])
        img = cv2.imread(os.path.join(img_dir, pic_name))# 读取灰度图
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_crop = img_gray[y1:y2, x1:x2]
        img_resize = cv2.resize(img_crop, (img_size, img_size))

        W_rate = img_size/(x2 - x1)
        H_rate = img_size/(y2 - y1)

        key_point[0] = (key_point[0] - x1)*W_rate  #坐标变换
        key_point[1] = (key_point[1] - x1)*W_rate
        key_point[2] = (key_point[2] - x1)*W_rate
        key_point[3] = (key_point[3] - x1)*W_rate
        key_point[4] = (key_point[4] - x1)*W_rate
        key_point[5] = (key_point[5] - x1)*W_rate
        key_point[6] = (key_point[6] - x1)*W_rate

        key_point[7] = (key_point[7] - y1)*H_rate
        key_point[8] = (key_point[8] - y1)*H_rate
        key_point[9] = (key_point[9] - y1)*H_rate
        key_point[10] = (key_point[10] - y1)*H_rate
        key_point[11] = (key_point[11] - y1)*H_rate
        key_point[12] = (key_point[12] - y1)*H_rate
        key_point[13] = (key_point[13] - y1)*H_rate

        datas[i, :, :, :] = np.array(img_resize).astype(np.float32)/255
        labels[i, :] = np.array(key_point).astype(np.float32) # 抽取label shape (13682,14) can not into shape (14)
        if i % 5 == 0:
            print("图片数据已经转换：", i)
    with h5py.File(hdf5_dir, 'w') as HDF5:
        HDF5.create_dataset('data', data=datas)
        HDF5.create_dataset('label', data=labels)
            # 写入列表文件，可以有多个hdf5文件
        # if i % 5 == 0:
            # print("datas[i, :, :, :]", datas[i, :, :, :])
            # print("key_point", key_pointHDF Explorer v1.4.0绿色版来)
            # print("图片数据已经转换：", i)
    print("已经转换图片总数：", i+1)
    print("******************数据转换写入完毕******************")
# ----------------------------------------------------main()--------------------------------------
if __name__ == "__main__":

    txt_dir = "E:/dataset/FacialPoints/DataAll_7Rt1.5.txt"
    img_dir = "/dataset/FacialPoints/Data_image"
    train_hdf5_dir = "E:/dataset/FacialPoints/train.h5"
    val_hdf5_dir = "E:/dataset/FacialPoints/val.h5"

    val_txt_dir = "E:/dataset/FacialPoints/val_list.txt"
    valRate = 0.1
    createVallist(txt_dir, val_txt_dir, valRate)
    createHdf5Data(img_dir, val_txt_dir, val_hdf5_dir, 40)
    createHdf5Data(img_dir, txt_dir, train_hdf5_dir, 40)

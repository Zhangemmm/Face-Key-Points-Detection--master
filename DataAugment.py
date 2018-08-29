'''
在生成hdf5.文件的基础上，增加数据增强模块实现。训练的样本数量 。
1.高斯模糊
把卷积和换成高斯核（自我认为是高斯核一个加权矩阵，主要用于乘以原来图像中的像素的值然后求和）
实现的函数是cv2.GaussianBlur()，需要指定高斯核的宽和高，都是奇数，
以及高斯函数沿着x轴和y轴的标准差，也可以使用cv2.getGaussianKernel()自己构建一个高斯核
'''
import h5py
import cv2
import os
import random
import numpy as np

# -------------------------------------------------ImagePreprocess ----------------------------------------------
'''
图片预处理，减去均值除以方差
'''
# def ImagePreprocess(impuImage)：
#-------------------------------------------------createTrainValList-----------------------------------------------
"""
data:2018.8.6
function name：createVallist()
function: genrate validation list
paramater：
          1.txt_dir: txt file path
          2val_txt_dir:
          2.valRate: validation rate        
return: none
Version：V2：
            1.以前抽取验证集的方式不对，这次直接原始数据中开头区10%（原始数据已经打乱了所以可以直接抽取）
             剩余的部分即可作为训练集
             
"""
def createTrainValList(txt_dir, train_txt_dir,val_txt_dir,valRate):
    with open(txt_dir, 'r') as txtfile:
        lines = txtfile.readlines()
    line_number = len(lines)
    val_number = int(line_number * valRate)
    val = open(val_txt_dir, 'w')
    for i in range(val_number):
        txt = lines[i]
        val.write(txt)
    val.close()

    train_number = line_number - val_number
    train = open(train_txt_dir, 'w')
    for j in range(train_number):
        txt = lines[val_number + j]
        train.write(txt)
    train.close()
    print("------>>>>生成训练数据数量：", train_number)
    print("------>>>>生成验证数据数量：", val_number)
    print("------>>>>train list & validation list create successfully")
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
Version：V2.0
        1.对原来的程序精简：
          原来存放图形和标签的数组是定义好大小的，改成根据读书数据大小确定，节省空间，便于向hdf5中添加新的文件。
        2.可以插入数据增强模块：x轴镜像，y轴镜像，高斯噪声，椒盐噪声，，旋转
'''
def createHdf5Data(img_dir, txt_dir, hdf5_dir, img_size, argument=False):
    print("开始数转换")
    with open(txt_dir, 'r') as txt_file:
        lines = txt_file.readlines()
    txt_list_number = len(lines)
    print(" txt_list_number", txt_list_number)
    if argument:
        total_number = txt_list_number*5
    else:
        total_number = txt_list_number
    print("total data number  ", total_number)
    datas = np.zeros((total_number, 1, img_size, img_size))
    labels = np.zeros((total_number, 14))
    key_point = np.zeros(14, dtype=float)
    key_point_flip1 = np.zeros(14, dtype=float)
    key_point_flip0 = np.zeros(14, dtype=float)
    key_point_Gauss = np.zeros(14, dtype=float)
    key_point_saltPe = np.zeros(14, dtype=float)
    imgSize = img_size*img_size
    countNumber = 0
    for i, line in enumerate(lines):
        new_line = line.split()
        pic_name = new_line[0]
        x1 = int(new_line[1])
        y1 = int(new_line[2])
        x2 = int(new_line[3])
        y2 = int(new_line[4])
        W_rate = 1 / (x2 - x1)
        H_rate = 1 / (y2 - y1)
        for j in range(7):
            key_point[j] = (float(new_line[j+5]) - x1)*W_rate
            key_point[j + 7] = (float(new_line[j+12]) - y1)*H_rate
        img = cv2.imread(os.path.join(img_dir, pic_name))  # 读取灰度图
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_crop = img_gray[y1:y2, x1:x2]
        img_resize = cv2.resize(img_crop, (img_size, img_size))
        datas[countNumber, :, :, :] = np.array(img_resize).astype(np.float32)/255
        labels[countNumber, :] = np.array(key_point).astype(np.float32)
        countNumber += 1
        if argument:#是否数据增强
            #关于Y方向图像翻转，左右眼睛，左右嘴角也会要对换
            img_flip1 = cv2.flip(img_resize, 1)  # 1水平翻转,x发生变化，y不变
            key_point_flip1[0] = 1 - key_point[3]
            key_point_flip1[1] = 1 - key_point[2]
            key_point_flip1[2] = 1 - key_point[1]
            key_point_flip1[3] = 1 - key_point[0]
            key_point_flip1[4] = 1 - key_point[4]
            key_point_flip1[5] = 1 - key_point[6]
            key_point_flip1[6] = 1 - key_point[5]

            key_point_flip1[7] = key_point[10]  # 3+7
            key_point_flip1[8] = key_point[9]  # 2+7
            key_point_flip1[9] = key_point[8]
            key_point_flip1[10] = key_point[7]
            key_point_flip1[11] = key_point[11]
            key_point_flip1[12] = key_point[13]
            key_point_flip1[13] = key_point[12]
            datas[countNumber, :, :, :] = np.array(img_flip1).astype(np.float32) / 255
            labels[countNumber, :] = np.array(key_point_flip1).astype(np.float32)
            countNumber += 1
            # 关于X轴
            img_flip0 = cv2.flip(img_resize, 0)  # 0，x不变 y变
            for flip0 in range(7):
                key_point_flip0[flip0] = key_point[flip0]
                key_point_flip0[flip0+7] = 1 - key_point[flip0+7]
            datas[countNumber, :, :, :] = np.array(img_flip0).astype(np.float32) / 255
            labels[countNumber, :] = np.array(key_point_flip0).astype(np.float32)
            countNumber += 1
            """
             高斯模糊,针对尺寸为40*40的图片，测试发现高斯核对图形模糊影响较大，然后是x轴的标准差
             高斯核选择3,5,7(9的时候无法辨识)，标准差选择5-10；高斯噪声对标注文件关键点没影响
            """
            kernal_size = random.randrange(3, 9, 2)  # 3,5, 7中间随机选一个,9不包含在内
            xStandDev = random.uniform(5, 10) #随机生成5-10之间
            img_Gauss = cv2.GaussianBlur(img_resize, (kernal_size, kernal_size), xStandDev)
            key_point_Gauss = key_point
            datas[countNumber, :, :, :] = np.array(img_Gauss).astype(np.float32) / 255
            labels[countNumber, :] = np.array(key_point_Gauss).astype(np.float32)
            countNumber += 1
            """
            椒盐噪声，椒盐噪声是数字图像中的常见噪声，一般是图像传感器、传输信道及解码处理等产生的黑白相间的亮暗点噪声，
            椒盐噪声常由图像切割产生。椒盐噪声是指两种噪声：盐噪声和椒噪声。盐噪声一般是白色噪声，椒噪声一般为黑色噪声。
            前者属于高灰度噪声，或者属于低灰度噪声;噪声不能太大，最大占图片的30%
            """
            saltPepperNumber = random.randint(10, 400)
            img_ssaltPe = img_resize

            for k in range(saltPepperNumber):
                randX = random.randint(0, 39)
                randY = random.randint(0, 39)
                if random.randint(0, 1) == 0:
                    img_ssaltPe[randX, randY] = 0
                else:
                    img_ssaltPe[randX, randY] = 255
            key_point_saltPe = key_point
            datas[countNumber, :, :, :] = np.array(img_ssaltPe).astype(np.float32) / 255
            labels[countNumber, :] = np.array(key_point_saltPe).astype(np.float32)
            countNumber += 1
            """
            图片旋转
            """
            """
            图片缩放
            """
        if i % 50 == 0:
            print("当前进度---->>：%.2f%%" % (i / txt_list_number * 100))
        if countNumber % 100 == 0:
            print("图片数据已经转换：", countNumber)
    with h5py.File(hdf5_dir, 'w') as HDF5:
        HDF5.create_dataset('data', data=datas)
        HDF5.create_dataset('label', data=labels)
    print("当前进度---->>: 100%")
    print("----->>>>已经转换图片总数：",  countNumber)
    print("******************数据转换写入完毕******************")
# ----------------------------------------------------main()--------------------------------------
if __name__ == "__main__":
    txt_dir = "E:/dataset/FacialPoints/DataAll_7Rt1.5.txt"
    img_dir = "/dataset/FacialPoints/Data_image"
    train_hdf5_dir = "E:/dataset/FacialPoints/EnhancementTest/DataAu/train.h5"
    val_hdf5_dir = "E:/dataset/FacialPoints/EnhancementTest/DataAu/val.h5"
    val_txt_dir = "E:/dataset/FacialPoints/EnhancementTest/DataAu/val_list.txt"
    train_txt_dir = "E:/dataset/FacialPoints/EnhancementTest/DataAu/train_list.txt"
    valRate = 0.1

    createTrainValList(txt_dir, train_txt_dir, val_txt_dir, valRate)
    createHdf5Data(img_dir, val_txt_dir, val_hdf5_dir, 40, argument=False)
    createHdf5Data(img_dir, txt_dir, train_hdf5_dir, 40, argument=True)
'''
在生成hdf5.文件的基础上，增加数据增强模块实现。训练的样本数量 。
1.高斯模糊
把卷积和换成高斯核（自我认为是高斯核一个加权矩阵，主要用于乘以原来图像中的像素的值然后求和）
实现的函数是cv2.GaussianBlur()，需要指定高斯核的宽和高，都是奇数，
以及高斯函数沿着x轴和y轴的标准差，也可以使用cv2.getGaussianKernel()自己构建一个高斯核

Version：3.3
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
# -------------------------------------------------显示标注点----------------------------------------------------
def circleOnImage(inputImage,inputArray):
    for PoiNum in range(7):
        img_new = cv2.circle(inputImage, (int(inputArray[PoiNum]), int(inputArray[PoiNum + 7])), 5, (0, 0, 255), -1)
        cv2.putText(img_new, str(PoiNum), (int(inputArray[PoiNum]), int(inputArray[PoiNum + 7])),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)  # 照片/添加的文字/左上角坐标/字体/字体大小/颜色/字体粗细
    cv2.imshow('inputImage ', img_new)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# -------------------------------------------------旋转----------------------------------------------------
"""
此处程序是处理图像旋转的，根据随产生的随机角度，选择图像旋转中心生成旋转矩阵，通过cv2函数旋转图片
然后关键点与旋转矩阵相乘生成新旋转后的关键点，图框裁剪图片后更新关键点。需要注意保证图片然关键点中心旋转后
不能超出人脸框的范围
"""
#   ----------------------------------------------------------------------------------
def GetRotateCenter(inputArray):
    rotateCenterX = 0
    rotateCenterY = 0
    for i in range(7):
        rotateCenterX += float(inputArray[i+4])
        rotateCenterY += float(inputArray[i+11])
    return rotateCenterX/7, rotateCenterY/7
#   ----------------------------------------------------------------------------------
def IfPointOutBounding(boundingPoint, Array):
    minPointX, maxPointX = np.min(Array[:6]), np.max(Array[:6])
    minPointY, maxPointY = np.min(Array[7:]), np.max(Array[7:])
    if minPointX < boundingPoint[0] or maxPointX > boundingPoint[2] or minPointY < boundingPoint[1] or maxPointY > boundingPoint[3] :
        return 1
    else:
        return 0
# ----------------------------------------------------------------------------------
def RotateImageAndArray(inputImage,keyPoint, Angle):
    showProcess = False
    outputArray = np.zeros(14)
    getRandomAngle = random.random()*2*Angle - Angle
    rotateCenterX, rotateCenterY = GetRotateCenter(keyPoint)
    rotateMat = cv2.getRotationMatrix2D((rotateCenterX, rotateCenterY), getRandomAngle, 1) #  设置旋转中心旋转中心，旋转角度，缩放比例
    imageAfterRotate = cv2.warpAffine(inputImage, rotateMat, (inputImage.shape[1], inputImage.shape[0]))
    #计算经过旋转后的关键点位置
    for i in range(7):
        outputArray[i] = rotateMat[0, 0] * keyPoint[i+4] + rotateMat[0, 1] * keyPoint[i + 11] + rotateMat[0, 2]
        outputArray[i + 7] = rotateMat[1, 0] * keyPoint[i+4] + rotateMat[1, 1] * keyPoint[i + 11] + rotateMat[1, 2]
    imgCrop = imageAfterRotate[int(keyPoint[1]):int(keyPoint[3]), int(keyPoint[0]):int(keyPoint[2])]
    if showProcess: #调试时候显示图片处理过程方便检查
        print("原图尺寸，高，宽", inputImage.shape[0], inputImage.shape[1])
        print("旋转后尺寸，高，宽", imageAfterRotate.shape[0], imageAfterRotate.shape[1])
        print("getRandomAngle", getRandomAngle)
        print("rotateCenterX, rotateCenterY", rotateCenterX, rotateCenterY)
        print("rotateMat shape", np.shape(rotateMat))
        print("rotateMat", rotateMat)
        #print("rotateMat[0,1]", rotateMat[0, 1])
        cv2.imshow("imageAfterRotate", imageAfterRotate)
        cv2.waitKey()
        cv2.destroyAllWindows()
    return imgCrop, outputArray
# -------------------------------------------------ImageZoom ----------------------------------------------
"""
人脸框随机缩放,加入限制当图框缩小时候，关键点不能到图框外 
Version;根据标注点的坐标计算出可以缩小最小极，根据最小极限生成随机数
形参是上极限，放大的比例
return:返回缩放后的图框坐标 
"""
def  ImageZoom(inputArray, scalePercent,imageShape):
    showProcess = False
    outputArray = np.zeros(4, dtype=int)
    point = [float(i) for i in inputArray]
    width = point[2] - point[0]
    heigh = point[3] - point[1]
    minXval ,maxXval= min(point[4:10]), max(point[4:10])
    minYval, maxYval = min(point[11:]), max(point[11:])
    minXpercent = min((minXval-point[0])/width, (point[2] - maxXval)/width)
    minYpercent = min((minYval-point[1])/heigh, (point[3]-maxXval)/heigh)
    minPercent = min(minXpercent, minYpercent)
    inputMinPercent = -min(minPercent, scalePercent)
    scalePercent1 = random.uniform(inputMinPercent, scalePercent)
    boundingOffsetX = int(scalePercent1 * width / 2)
    boundingOffsetY = int(scalePercent1 * heigh / 2)
    outputArray[0] = int(inputArray[0]) - boundingOffsetX
    outputArray[2] = int(inputArray[2]) + boundingOffsetX
    outputArray[1] = int(inputArray[1]) - boundingOffsetY
    outputArray[3] = int(inputArray[3]) + boundingOffsetY
    #如果边框超过图片尺寸了，把图片的边作为边框的边
    if outputArray[0] < 0:
        outputArray[0] = 0
    if outputArray[1] < 0:
        outputArray[1] = 0
    if outputArray[2] > imageShape[1]:  #imageshape[1]是图片的宽度
        outputArray[2] = imageShape[1]
    if outputArray[3] > imageShape[0]:  # imageshape[1]是图片的宽度
        outputArray[3] = imageShape[0]
    if showProcess:
        print("minXval ,maxXval", minXval, maxXval)
        print(" minYval, maxYval", minYval, maxYval)
        print("XY缩放图框距离：", boundingOffsetX, boundingOffsetY)
        print("缩放后的图框输出图框", outputArray)
    minMaxXY = [minXval, maxXval, minYval, maxYval]
    return outputArray, point, minMaxXY
# -------------------------------------------------BoundingCrop-----------------------------------------------
"""
data:2018.8.6
function name：BoundingChang()
function: 图框随机移动 
paramater：
          1.标注数据数组（包括边框）
          2.扰动范围: -30%-30%       
return: none
Version：V3.0
        1.训练出来的模型在回归预测时候输出值一直一样，查看后发现模型绝大多数参数是0,
        可能是输入图像中人脸相对于关键点位置基本差不多导致，模型只学习到这一个比较名明显的特征并没有根据像素点来来学习特征
        就可以给图框加入扰动，消除因为图框标准相似带来的影响。(关键点坐标不变)
        2.增加图框扰动程序(以右上角交叉两条边为基础)2.1随机选择扰动的边数量。2.2随机选择边扰动的方向和扰动的范围直接使用正负
Version：V3.1
        1.先随机随访边框然后随机移动关键点（关键点移动人图片对象也要移动），先随机移动然后随机缩放        
"""
def BoundingCrop(labelArray, changePercent,imageShape):
    showProcess = False
    key_point = np.zeros(14, dtype=float)
    outputArray = np.zeros(4, dtype=int)
    outputBounding, point, minMaxXY = ImageZoom(labelArray, 0.3, imageShape)
    NewWidth = int(outputBounding[2] - outputBounding[0])
    NewHeigh = int(outputBounding[3] - outputBounding[1])
    #求边可以向内移动的最大百分比，保证关键点不出图框
    minXpercent = min((minMaxXY[0] - outputBounding[0])/NewWidth, (outputBounding[2] - minMaxXY[1])/NewWidth)
    minYpercent = min((minMaxXY[2] - outputBounding[1]) / NewHeigh, (outputBounding[3] - minMaxXY[3])/NewHeigh)
    percent1 = random.uniform(-min(minXpercent, changePercent), min(minXpercent, changePercent))
    percent2 = random.uniform(-min(minYpercent, changePercent), min(minYpercent, changePercent))
    offset1 = int(percent1 * NewWidth)
    offset2 = int(percent2 * NewHeigh)
    outputArray[0] = int(outputBounding[0]) + offset1
    outputArray[2] = int(outputBounding[2]) + offset1
    outputArray[1] = int(outputBounding[1]) + offset2
    outputArray[3] = int(outputBounding[3]) + offset2
    if outputArray[0] < 0:
        outputArray[0] = 0
    if outputArray[1] < 0:
        outputArray[1] = 0
    if outputArray[2] > imageShape[1]:  #imageshape[1]是图片的宽度
        outputArray[2] = imageShape[1]
    if outputArray[3] > imageShape[0]:  # imageshape[1]是图片的宽度
        outputArray[3] = imageShape[0]
    for k in range(7):
        key_point[k] = (point[k+4] - outputArray[0])/NewWidth
        key_point[k + 7] = (point[k + 11]-outputArray[1])/NewHeigh
    if showProcess:
        print("(minMaxXY[0] - outputBounding[0])/NewWidth", (minMaxXY[0] - outputBounding[0]) / NewWidth)
        print("(outputBounding[2] - minMaxXY[1])/NewWidth", (outputBounding[2] - minMaxXY[1]) / NewWidth)
        print("minXpercent", minXpercent)
        print("minYpercent", minYpercent)
        print("percent1", percent1)
        print("percent2", percent2)
        print("offset1", offset1)
        print("offset1", offset2)
        print(" key_point[k]", key_point)
    return outputArray, key_point
# -------------------------------------------------createTrainValList-----------------------------------------------
"""
data:2018.8.16
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
def createTrainValList(txt_dir, train_txt_dir, val_txt_dir, valRate):
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
        2.可以插入数据增强模块：x轴镜像，y轴镜像，高斯噪声，椒盐噪声，旋转等
'''
def createHdf5Data(img_dir, txt_dir, hdf5_dir, img_size,ZoomCropNumber,RotateNumber):
    argument_2 = False
    showImage = False
    changePercent = 0.3 #  随机缩放百分比-30%~30%
    Angle = 45          #   随机旋转角度-45~45
    countNumber = 0
    imgSize = img_size * img_size
    key_point = np.zeros(14, dtype=float)
    key_point_flip1 = np.zeros(14, dtype=float)
    key_point_flip0 = np.zeros(14, dtype=float)
    key_point_Gauss = np.zeros(14, dtype=float)
    key_point_saltPe = np.zeros(14, dtype=float)
    print("开始数转换")
    with open(txt_dir, 'r') as txt_file:
        lines = txt_file.readlines()
    txt_list_number = len(lines)
    print(" txt_list_number", txt_list_number)
    if argument_2:
        total_number = txt_list_number * 5#待定
    else:
        total_number = txt_list_number*(ZoomCropNumber + RotateNumber)# +1是关于y轴镜像的
    datas = np.zeros((total_number, 1, img_size, img_size))
    labels = np.zeros((total_number, 14))
    print("total data number  ", total_number)
    for i, line in enumerate(lines):
        new_line = line.split()
        pic_name = new_line[0]
        keyPoint = [float(j) for j in new_line[1:]]
        img_read = cv2.imread(os.path.join(img_dir, pic_name))
        img_shape = img_read.shape
        # --------------------------------------
        if showImage:  #过程显示
            print("￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥")
            print("pic_name", pic_name)
            print("img_shape",img_shape)
            print("原来图框尺寸", new_line[1:5])
            print("输入标注文件", new_line[5:])
            img_crop1 = img_read[int(keyPoint[1]):int(keyPoint[3]), int(keyPoint[0]):int(keyPoint[2])]
            cv2.imshow("img_read1", img_crop1)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        #--------------------------------------缩放和裁剪-------------------------------------------
        for n_number in range(ZoomCropNumber):
            outputArray, key_point = BoundingCrop(keyPoint, changePercent, img_shape)
            img_crop = img_read[outputArray[1]:outputArray[3], outputArray[0]:outputArray[2]]
            img_gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
            img_resize = cv2.resize(img_gray, (img_size, img_size))
            datas[countNumber, :, :, :] = np.array(img_resize).astype(np.float32) / 255
            labels[countNumber, :] = np.array(key_point).astype(np.float32)
            countNumber += 1
            if showImage:  # 过程显示
                print("改变后图框", outputArray)
                cv2.imshow("img_read", img_crop)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        # --------------------------------------旋转变换-------------------------------------------
        for n1_number in range(RotateNumber):
            imgCrop, rotateOutputArray = RotateImageAndArray(img_read, keyPoint, Angle)
            if IfPointOutBounding(keyPoint[:4], rotateOutputArray[4:]) == 1: # 判断旋转后的点有没有出人脸框
                continue
            # 更新相对于图框的关键点位置
            for i in range(7):
                rotateOutputArray[i] = (rotateOutputArray[i] - keyPoint[0])/(keyPoint[2]-keyPoint[0])
                rotateOutputArray[i + 7] = (rotateOutputArray[i + 7] - keyPoint[1])/(keyPoint[3]-keyPoint[1])
            img_gray = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2GRAY)
            imgResizeRotate = cv2.resize(img_gray, (img_size, img_size))
            datas[countNumber, :, :, :] = np.array(imgResizeRotate).astype(np.float32) / 255
            labels[countNumber, :] = np.array(rotateOutputArray).astype(np.float32)
            countNumber += 1
            if showImage:
                for i in range(7):
                    rotateOutputArray[i] = rotateOutputArray[i]*(keyPoint[2]-keyPoint[0])
                    rotateOutputArray[i + 7] = rotateOutputArray[i + 7]*(keyPoint[3]-keyPoint[1])
                circleOnImage(imgCrop, rotateOutputArray)
        # --------------------------------------旋转变换-------------------------------------------
        if argument_2:  #
            # 关于Y方向图像翻转，左右眼睛，左右嘴角也会要对换
            img_flip1 = cv2.flip(img_read, 1)  # 1水平翻转,x发生变化，y不变
            img_flip1_crop =img_flip1[int(keyPoint[1]):int(keyPoint[3]), int(keyPoint[0]):int(keyPoint[2])]
            key_point = keyPoint[4:]
            boundingWidth = keyPoint[2]-keyPoint[0]
            boundingHeigh = keyPoint[3] - keyPoint[1]
            key_point_flip1[0] = keyPoint[2] - key_point[3]
            key_point_flip1[1] = keyPoint[2] - key_point[2]
            key_point_flip1[2] = keyPoint[2] - key_point[1]
            key_point_flip1[3] = keyPoint[2] - key_point[0]
            key_point_flip1[4] = keyPoint[2] - key_point[4]
            key_point_flip1[5] = keyPoint[2] - key_point[6]
            key_point_flip1[6] = keyPoint[2] - key_point[5]

            key_point_flip1[7] = key_point[10] - keyPoint[1] # 3+7
            key_point_flip1[8] = key_point[9] - keyPoint[1]  # 2+7
            key_point_flip1[9] = key_point[8] - keyPoint[1]
            key_point_flip1[10] = key_point[7] - keyPoint[1]
            key_point_flip1[11] = key_point[11] - keyPoint[1]
            key_point_flip1[12] = key_point[13] - keyPoint[1]
            key_point_flip1[13] = key_point[12] - keyPoint[1]

        # datas[countNumber, :, :, :] = np.array(img_flip1).astype(np.float32) / 255
        # labels[countNumber, :] = np.array(key_point_flip1).astype(np.float32)
        # countNumber += 1
        if showImage:
            circleOnImage(img_flip1_crop , key_point_flip1)
        if argument_2:#
            # 关于 x轴镜像和添加噪声先不用
            img_flip0 = cv2.flip(img_resize, 0)  # 0，x不变 y变
            for flip0 in range(7):
                key_point_flip0[flip0] = key_point[flip0]
                key_point_flip0[flip0 + 7] = 1 - key_point[flip0 + 7]
            datas[countNumber, :, :, :] = np.array(img_flip0).astype(np.float32) / 255
            labels[countNumber, :] = np.array(key_point_flip0).astype(np.float32)
            countNumber += 1
            """
             高斯模糊,针对尺寸为40*40的图片，测试发现高斯核对图形模糊影响较大，然后是x轴的标准差
             高斯核选择3,5,7(9的时候无法辨识)，标准差选择5-10；高斯噪声对标注文件关键点没影响
            """
            kernal_size = random.randrange(3, 9, 2)  # 3,5, 7中间随机选一个,9不包含在内
            xStandDev = random.uniform(5, 10)  # 随机生成5-10之间
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

        if i % 10 == 0:
            print("当前进度---->>：%.2f%%" % (i / txt_list_number * 100))
        if countNumber % 100 == 0:
            print("图片数据已经转换：", countNumber)
    with h5py.File(hdf5_dir, 'w') as HDF5:
        HDF5.create_dataset('data', data=datas)
        HDF5.create_dataset('label', data=labels)
    print("当前进度---->>: 100%")
    print("----->>>>已经转换图片总数：", countNumber)
    print("******************数据转换写入完毕******************")
# ----------------------------------------------------main()--------------------------------------
if __name__ == "__main__":
    txt_dir = "E:/dataset/FacialPoints/DataAll_7Rt1.5.txt"
    img_dir = "E:/dataset/FacialPoints/Data_image"
    train_hdf5_dir = "E:/dataset/FacialPoints/EnhancementTest/DataAu/train.h5"
    val_hdf5_dir = "E:/dataset/FacialPoints/EnhancementTest/DataAu/val.h5"
    val_txt_dir = "E:/dataset/FacialPoints/EnhancementTest/DataAu/val_list.txt"
    train_txt_dir = "E:/dataset/FacialPoints/EnhancementTest/DataAu/train_list.txt"
    valRate = 0.1
    createTrainValList(txt_dir, train_txt_dir, val_txt_dir, valRate)
    createHdf5Data(img_dir, val_txt_dir, val_hdf5_dir, 40, 1, 0)
    createHdf5Data(img_dir, txt_dir, train_hdf5_dir, 40, 5, 5)
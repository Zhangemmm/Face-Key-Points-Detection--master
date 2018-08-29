"""
图像旋转：以关键点点中心为旋转中心，根据旋转中心和和旋转角度生成旋转矩阵
然后关键点乘上旋转矩阵，涉及关键点位置判断，关键点不能出图框,同时标注出旋转后的点
"""
import numpy as np
import cv2
import random
import os
#   ----------------------------------------------------------------------------------
def GetRotateCenter(inputArray):
    rotateCenterX = 0
    rotateCenterY = 0
    for i in range(7):
        rotateCenterX += float(inputArray[i])
        rotateCenterY += float(inputArray[i+7])
    return rotateCenterX/7, rotateCenterY/7
#   ----------------------------------------------------------------------------------
def IfPointOutBounding(outputArray, Array):
    minPointX, maxPointY = min(Array[:6]),max(Array[:6])
    minPointY, maxPointY = min(Array[7:]), max(Array[7:])
#   ----------------------------------------------------------------------------------
def RotateImageAndArray(inputImage, boundingPoint,inputArray, Angle):
    showProcess = True
    outputArray = np.zeros(14)
    Array = [float(i) for i in inputArray] #把string转float
    getRandomAngle = random.random()*2*Angle - Angle
    rotateCenterX, rotateCenterY = GetRotateCenter(inputArray)
    rotateMat = cv2.getRotationMatrix2D((rotateCenterX, rotateCenterY), getRandomAngle, 1) #  设置旋转中心旋转中心，旋转角度，缩放比例
    imageAfterRotate = cv2.warpAffine(inputImage, rotateMat, (inputImage.shape[1], inputImage.shape[0]))

    for i in range(7):
        outputArray[i] = rotateMat[0, 0] * float(Array[i]) + rotateMat[0, 1] * float(Array[i + 7]) + rotateMat[0, 2] - boundingPoint[0]
        outputArray[i + 7] = rotateMat[1, 0] * float(Array[i]) + rotateMat[1, 1] * float(Array[i + 7]) + rotateMat[1, 2] - boundingPoint[1]
    if showProcess:
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
        imgCrop = imageAfterRotate[boundingPoint[1]:boundingPoint[3], boundingPoint[0]:boundingPoint[2]]
        for PoiNum in range(7):
            img_new = cv2.circle(imgCrop, (int(outputArray[PoiNum]), int(outputArray[PoiNum + 7])), 5, (0, 0, 255), -1)
            cv2.putText(img_new, str(PoiNum), (int(outputArray[PoiNum]), int(outputArray[PoiNum + 7])),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)  # 照片/添加的文字/左上角坐标/字体/字体大小/颜色/字体粗细
        cv2.imshow('imageAfterRotate ', imgCrop)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

#   ----------------------------------------------------------------------------------
if __name__ == "__main__":
    showProcess1 = True
    Angle = 90
    pic_dir = "E:/dataset/FacialPoints/Data_image"
    pic_list = "E:/dataset/FacialPoints/EnhancementTest/DataAu/val_list.txt"
    pic_save = "E:/dataset/FacialPoints/pyproject/save"
    with open(pic_list, 'r') as f:
        lines = f.readlines()
    keypoint = np.zeros(14, dtype=int)
    for line in lines:
        new_line = line.split()  # str.split(str="", num=string.count(str)).str -- 分隔符，默认为所有的空字符，
        pic_name = new_line[0]
        boundingPoint = [int(i)for i in new_line[1:5]]
        keypoint = new_line[5:]
        print(len(keypoint))
        # 关键点的数组
        imgRead = cv2.imread(os.path.join(pic_dir, pic_name))
        #imgCrop = imgRead[boundingPoint[1]:boundingPoint[3], boundingPoint[0]:boundingPoint[2]]
        if showProcess1:
            print(pic_name)
            print(os.path.join(pic_dir, pic_name))
            cv2.imshow("imgCrop", imgRead )
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        RotateImageAndArray(imgRead,boundingPoint, keypoint, Angle)
    print("picture saved ok!!!!!!!!!!", pic_name)









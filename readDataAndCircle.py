import cv2
import os
import numpy as np

'''
data：2018.8.7
__author__ == "syy"
function：读取图片画出对应框和关键点
'''
def show_box_point(pic_path,test_list_path,pic_save_path):
    with open(test_list_path, 'r') as f:
        lines = f.readlines()
    keypoint = np.zeros(14, dtype=int)
    print("lines", len(lines))
    Wk = 1415
    Hk =1415
    for line in lines:
        new_line = line.split()  # str.split(str="", num=string.count(str)).str -- 分隔符，默认为所有的空字符，
        print("new_line", len(new_line))
        keypoint[0] = float(new_line[0])*Wk
        keypoint[1] = float(new_line[1])*Wk
        keypoint[2] = float(new_line[2])*Wk
        keypoint[3] = float(new_line[3])*Wk
        keypoint[4] = float(new_line[4])*Wk
        keypoint[5] = float(new_line[5])*Wk
        keypoint[6] = float(new_line[6])*Wk

        keypoint[7] = float(new_line[7])*Hk
        keypoint[8] = float(new_line[8])*Hk
        keypoint[9] = float(new_line[9])*Hk
        keypoint[10] = float(new_line[10])*Hk
        keypoint[11] = float(new_line[11])*Hk
        keypoint[12] = float(new_line[12])*Hk
        keypoint[13] = float(new_line[13])*Hk
        # 关键点的数组
        # print(os.path.join(test_pic_path, pic_name))
        # img = cv2.imread(os.path.join(test_pic_path, pic_name))
        img = cv2.imread(pic_path)
        cv2.imshow("picture", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # img_new = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        for PoiNum in range(7):
            img_new = cv2.circle(img, (int(keypoint[PoiNum]), int(keypoint[PoiNum + 7])), 10, (0, 0, 255), 5)
            cv2.putText(img_new, str(PoiNum), (int(keypoint[PoiNum]), int(keypoint[PoiNum + 7])), cv2.FONT_HERSHEY_COMPLEX, 4, (0, 0, 255), 1)#照片/添加的文字/左上角坐标/字体/字体大小/颜色/字体粗细
        cv2.imshow('pic_name', img_new)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite(pic_save_path,  img_new)
        # cv2.imwrite(os.path.join(pic_save_path, 'mark' + pic_name), img_new)
    # print("picture saved ok!!!!!!!!!!", pic_name)
    print("picture saved ok!!!!!!!!!!")
    print(keypoint)

if __name__ == "__main__":
    pic_dir = "E:/dataset/FacialPoints/4.jpg"
    pic_list = "E:/dataset/FacialPoints/list_1111.txt"
    pic_save = "E:/dataset/FacialPoints/rt_4.jpg"
    show_box_point(pic_dir, pic_list, pic_save)
#['49142.jpg\t238 396 1256 1414\t512.643 676.214 851.929 1015.5 778.357 586.214 899.786 738.636 762.922 750.779 735.779 936.493 1122.21 1127.92']
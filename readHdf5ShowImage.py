"""
读取HDF5文件,并还原出HDF5文件中的图片
"""
import numpy as np
import h5py
from PIL import Image
HDF5_PATH ="E:/dataset/FacialPoints/EnhancementTest/boundingchange/model1/train.h5"
readHdf5 = h5py.File(HDF5_PATH, 'r')
data = readHdf5['data']
label= readHdf5['label']
dataLen = len(data)
labelLen = len(label)
print("dataLen，labelLen",dataLen,labelLen)
for i,data_i in enumerate(data):
    print("data number:------->>>", i)
    print("data_i shape", np.shape(data_i))
    print("data:", data_i)
    imgArray = np.array(data_i, dtype=float)
    imgArray = imgArray.reshape([40, 40])
    img = Image.fromarray(imgArray*255)
    img.show()

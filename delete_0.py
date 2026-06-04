#删除标签中像素全文0的图片
import os
import cv2

folder_path = r'E:\Change-Model\HRSCD\HRSCD\label1\14-2012-0420-6905-LA93-0M50-E080'

for filename in os.listdir(folder_path):
    if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.tif'):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img.sum() == 0:
            os.remove(img_path)
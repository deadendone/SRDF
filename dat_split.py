import os
import random
import shutil


def moveFile(input1, input2, input3,input4,input5, save1, save2,save3,save4, save5):
    pathDir = os.listdir(input1)
    random.seed(1)
    filenumber = len(pathDir)
    rate = 0.1     ######The proportion of extraction
    picknumber = int(filenumber * rate)
    sample = random.sample(pathDir, picknumber)
    print(sample)
    list_len = len(sample)
    print(list_len)
    list = []
    for i in range(len(sample)):
        list.append(sample[i].split('.')[0])
    print(list)
    for flie_name in list:
        path_img = os.path.join(input1, flie_name + '.png')
        shutil.move(path_img, save1)
        path_lab = os.path.join(input2, flie_name + '.png')
        shutil.move(path_lab, save2)
        path_lab = os.path.join(input3, flie_name + '.png')
        shutil.move(path_lab, save3)
        path_lab = os.path.join(input4, flie_name + '.png')
        shutil.move(path_lab, save4)
        path_lab = os.path.join(input5, flie_name + '.png')
        shutil.move(path_lab, save5)




if __name__ == '__main__':
    input_path1 = r'E:\Change-Model\HRSCD\HRSCD_D35\D35\D35\test\im1'
    input_path2 = r'E:\Change-Model\HRSCD\HRSCD_D35\D35\D35\test\im2'
    input_path3 = r'E:\Change-Model\HRSCD\HRSCD_D35\D35\D35\test\label1_rgb'
    input_path4 = r'E:\Change-Model\HRSCD\HRSCD_D35\D35\D35\test\label2_rgb'
    input_path5 = r'E:\Change-Model\HRSCD\HRSCD_D35\D35\D35\test\boundary'
    # input_path5 = 'E:/semisupervisedCD/experiment/SECOND_duoleidiejia/sensetime_change_detection_train/train/labelchange'
    save_img1 = r'E:\SCD-DataSets\HRSCD_10p\test\im1'
    save_img2 = r'E:\SCD-DataSets\HRSCD_10p\test\im2'
    save_lab1 = r'E:\SCD-DataSets\HRSCD_10p\test\label1_rgb'
    save_lab2 = r'E:\SCD-DataSets\HRSCD_10p\test\label2_rgb'
    save_lab3 = r'E:\SCD-DataSets\HRSCD_10p\test\boundary'
    # save_lab3 = 'E:/semisupervisedCD/experiment/SECOND_duoleidiejia/Data4/Val/labelchange'
    if not os.path.exists(save_lab1):
        os.makedirs(save_lab1)
    if not os.path.exists(save_img1):
        os.makedirs(save_img1)
    if not os.path.exists(save_img2):
        os.makedirs(save_img2)
    if not os.path.exists(save_lab2):
        os.makedirs(save_lab2)
    # if not os.path.exists(save_lab3):
    #     os.makedirs(save_lab3)
    moveFile(input_path1, input_path2,input_path3, input_path4, input_path5, save_img1, save_img2,save_lab1, save_lab2, save_lab3)
# python dat_split.py

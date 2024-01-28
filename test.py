# -*- coding: utf-8 -*-

import cv2
from sklearn.metrics import roc_auc_score
import numpy as np
from PIL import Image
from utils.test_utils_res import SegFormer_Segmentation
from utils.utils import decompose, merge, rm_and_make_dir
from tqdm import tqdm
import os
import shutil
import torch
import warnings
warnings.filterwarnings("ignore")
torch.cuda.empty_cache()


def detect_image_stride(test_path, dir_save_path):
    print('Stride Test!!')
    test_size = '512'
    _, path_out = decompose(test_path, test_size)
    print('Decomposition complete.')
    dir_pre_path = r'test_out/temp/input_decompose_' + test_size + '_pred/'
    rm_and_make_dir(dir_pre_path)
    img_names = os.listdir(path_out)
    for img_name in tqdm(img_names):
        if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            image_path  = os.path.join(path_out, img_name)
            image       = Image.open(image_path)

            _, seg_pred= segformer.detect_image_resize(image)
            save_name = img_name[:-4] + '.png'
            if not os.path.exists(dir_pre_path):
                os.makedirs(dir_pre_path)
            seg_pred.save(os.path.join(dir_pre_path, save_name))#滑块预测的概率图
    print('Prediction complete.')
    if os.path.exists('test_out/temp/input_decompose_' + test_size + '/'):
        shutil.rmtree('test_out/temp/input_decompose_' + test_size + '/')
    merge(test_path, dir_pre_path, dir_save_path, test_size)#merge把预测的滑块图合并起来
    print('Merging complete.')
    return


def metric(premask, groundtruth):
    seg_inv, gt_inv = np.logical_not(premask), np.logical_not(groundtruth)#布尔值取反
    true_pos = float(np.logical_and(premask, groundtruth).sum())  # float for division
    true_neg = np.logical_and(seg_inv, gt_inv).sum()
    false_pos = np.logical_and(premask, gt_inv).sum()
    false_neg = np.logical_and(seg_inv, groundtruth).sum()
    f1 = 2 * true_pos / (2 * true_pos + false_pos + false_neg + 1e-6)
    cross = np.logical_and(premask, groundtruth)
    union = np.logical_or(premask, groundtruth)
    iou = np.sum(cross) / (np.sum(union) + 1e-6)
    if np.sum(cross) + np.sum(union) == 0:
        iou = 1
    return f1, iou
def test_mode(dir_origin_path,dir_save_path):
    # img_names = os.listdir(dir_origin_path)
    # for img_name in tqdm(img_names):
    #     if img_name.lower().endswith(
    #             ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
    #         image_path = os.path.join(dir_origin_path, img_name)
    #         image = Image.open(image_path)
    #         # # 无resize
    #         # _, seg_pred= segformer.detect_image_noresize(image)
    #         # # print('No Resize Test')
    #         # save_name = img_name[:-4] + '.png'
    #         # if not os.path.exists(dir_save_path):
    #         #     os.makedirs(dir_save_path)
    #         # seg_pred.save(os.path.join(dir_save_path, save_name))
    #
    #         # 等ratio做resize
    #         _, seg_pred = segformer.detect_image_resize(image)
    #         # print('Resize Test!!')
    #         save_name = img_name[:-4] + '.png'
    #         if not os.path.exists(dir_save_path):
    #             os.makedirs(dir_save_path)
    #         seg_pred.save(os.path.join(dir_save_path, save_name))
    ## 滑动窗口测试
    detect_image_stride(dir_origin_path, dir_save_path)
    print("test_over!")
    return
def evaluate(path_pre,path_gt,dataset_name,record_txt):
    if os.path.exists(path_gt):
        flist = sorted(os.listdir(path_pre))
        auc, f1, iou = [], [], []
        for file in tqdm(flist):
            try:
                pre = cv2.imread(path_pre + file)
                gt = cv2.imread(path_gt + file[:-4] + '_gt.png')
                H, W, C = pre.shape
                Hg, Wg, C = gt.shape
                if H != Hg or W != Wg:
                    gt = cv2.resize(gt, (W, H))
                    gt[gt > 127] = 255
                    gt[gt <= 127] = 0
                if np.max(gt) != np.min(gt):
                    auc.append(roc_auc_score((gt.reshape(H * W * C) / 255).astype('int'), pre.reshape(H * W * C) / 255.))
                pre[pre > 127] = 255
                pre[pre <= 127] = 0
                a, b = metric(pre / 255, gt / 255)
                f1.append(a)
                iou.append(b)
            except Exception as e:
                print(file)

        print(dataset_name)
        print('Evaluation: AUC: %5.4f, F1: %5.4f, IOU: %5.4f' % (np.mean(auc), np.mean(f1), np.mean(iou)))
    with open(record_txt,"a") as f:
        f.writelines(dataset_name+"\n")
        f.writelines('Evaluation: AUC: %5.4f, F1: %5.4f, IOU: %5.4f' % (np.mean(auc), np.mean(f1), np.mean(iou)))
        f.writelines("\n")
    return np.mean(auc), np.mean(f1), np.mean(iou)

if __name__ == "__main__":
    used_weigth=r"./weights/weights_EITL.pth"
    segformer = SegFormer_Segmentation("b2",used_weigth)
    record_txt = r"./test_out/evaluate_result.txt"
    with open(record_txt,"a") as f:
        f.writelines(str(used_weigth))
        f.writelines("\n")
    #test_samples
    # test_path = r'./samples/tamper/'
    # save_path = r"./test_out/samples_predict/"
    # path_gt = r'./samples/gt/'
    test_path = r'F:\Datasets\DSO\tamper/'
    save_path = r'D:\Datasets\EITLNet\DSO\Debug_no_resize/'
    path_gt = r'F:\Datasets\DSO\gt/'
    test_mode(test_path,save_path)
    auc,f1,iou=evaluate(save_path,path_gt,"samples",record_txt)












    # #DSO
    # dir_origin_path = r'H:\Standard_Database\a-ImageForgeriesOSN_Dataset\DSO'+Robust_dataset
    # dir_save_path = r"test_out/DSO/Debug_no_resize/"#保存的预测结果图
    # path_pre = dir_save_path
    # path_gt = r'H:temp/DSO_GT/'
    # test_mode(dir_origin_path,dir_save_path)#要自己手动去选择注释代码来选择测试方式
    # auc,f1,iou=evaluate(path_pre,path_gt,"DSO",record_txt)

    # #Columbia
    # dir_origin_path = r'H:/Standard_Database/a-ImageForgeriesOSN_Dataset/Columbia'+Robust_dataset
    # dir_save_path = r"test_out/Columbia/Debug_no_resize/"
    # path_pre=  dir_save_path
    # path_gt = r'H:\Standard_Database\a-ImageForgeriesOSN_Dataset\Columbia_GT/'
    # test_mode(dir_origin_path, dir_save_path)  # 要自己手动去选择注释代码来选择测试方式
    # auc, f1, iou = evaluate(path_pre, path_gt, "Columbia", record_txt)
    #
    #  #CASIA
    # dir_origin_path = r'H:\Standard_Database\a-ImageForgeriesOSN_Dataset\CASIA'+Robust_dataset#待滑块测试的数据集
    # dir_save_path = r"E:\Experiments\test_out\temp_save\CASIA1\Debug_no_resize/"#滑块后的图片保存在的路径
    # path_pre=dir_save_path
    # path_gt = r"H:\Standard_Database\a-ImageForgeriesOSN_Dataset\CASIA_GT/"#原图大小对应的groundTruth
    # test_mode(dir_origin_path, dir_save_path)  # 要自己手动去选择注释代码来选择测试方式
    # auc, f1, iou = evaluate(path_pre, path_gt, "CASIA", record_txt)
    # #
    #
    # # Coverage
    # dir_origin_path = r'H:\temp\COVERAGE_fixed\Coverage/'
    # dir_save_path = r"test_out/Coverage/Debug_no_resize/"
    # path_pre= dir_save_path
    # path_gt = r"H:\temp\COVERAGE_fixed\Coverage_gt/"
    # test_mode(dir_origin_path, dir_save_path)  # 要自己手动去选择注释代码来选择测试方式
    # auc, f1, iou = evaluate(path_pre, path_gt, "Coverage", record_txt)
    #
    #
    # #
    # # NIST16
    # dir_origin_path = r'H:\Standard_Database\a-ImageForgeriesOSN_Dataset\NIST16'+Robust_dataset
    # dir_save_path = r"test_out/NIST16/Debug_no_resize/"
    # path_pre = dir_save_path
    # path_gt = r"H:\temp\NIST16_GT/"
    # test_mode(dir_origin_path, dir_save_path)  # 要自己手动去选择注释代码来选择测试方式
    # auc, f1, iou = evaluate(path_pre, path_gt, "NIST16", record_txt)
    #
    #
    # # IMD20
    # dir_origin_path = r'H:\temp\IMD20_fixed\tamper_all/'
    # dir_save_path = r"test_out/IMD20/Debug_no_resize/"
    # path_pre = dir_save_path
    # path_gt = r"H:\temp\IMD20_fixed\gt_all/"
    # test_mode(dir_origin_path, dir_save_path)  # 要自己手动去选择注释代码来选择测试方式
    # auc, f1, iou = evaluate(path_pre, path_gt, "IMD20", record_txt)
    #

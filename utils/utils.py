import numpy as np
from PIL import Image
from tqdm import tqdm
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 

def resize_image(image, size):
    iw, ih = image.size
    w, h = size

    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))

    return new_image, nw, nh

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def preprocess_input(image):
    image -= np.array([123.675, 116.28, 103.53], np.float32)
    image /= np.array([58.395, 57.12, 57.375], np.float32)
    return image

def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)

import os
import cv2
import copy
import shutil

def decompose(test_path, test_size):
    flist = sorted(os.listdir(test_path))
    size_list = [int(test_size)]
    for size in size_list:
        path_out = 'test_out/temp/input_decompose_' + str(size) + '/'
        rm_and_make_dir(path_out)
    rtn_list = [[]]
    for file in tqdm(flist):
        img = cv2.imread(test_path + file)
        # img = cv2.rotate(img, cv2.cv2.ROTATE_180)
        H, W, _ = img.shape
        size_idx = 0
        while size_idx < len(size_list) - 1:
            if H < size_list[size_idx+1] or W < size_list[size_idx+1]:
                break
            size_idx += 1
        rtn_list[size_idx].append(file)
        size = size_list[size_idx]
        path_out = 'test_out/temp/input_decompose_' + str(size) + '/'
        X, Y = H // (size // 2) + 1, W // (size // 2) + 1
        idx = 0
        for x in range(X-1):
            if x * size // 2 + size > H:
                break
            for y in range(Y-1):
                if y * size // 2 + size > W:
                    break
                img_tmp = img[x * size // 2: x * size // 2 + size, y * size // 2: y * size // 2 + size, :]
                cv2.imwrite(path_out + file[:-4] + '_%03d.png' % idx, img_tmp)
                idx += 1
            img_tmp = img[x * size // 2: x * size // 2 + size, -size:, :]
            cv2.imwrite(path_out + file[:-4] + '_%03d.png' % idx, img_tmp)
            idx += 1
        for y in range(Y - 1):
            if y * size // 2 + size > W:
                break
            img_tmp = img[-size:, y * size // 2: y * size // 2 + size, :]
            cv2.imwrite(path_out + file[:-4] + '_%03d.png' % idx, img_tmp)
            idx += 1
        img_tmp = img[-size:, -size:, :]
        cv2.imwrite(path_out + file[:-4] + '_%03d.png' % idx, img_tmp)
        idx += 1
    return rtn_list, path_out


def merge(path, path_p, path_s, test_size):
    rm_and_make_dir(path_s)
    size = int(test_size)

    gk = gkern(size)
    gk = 1 - gk

    for file in sorted(os.listdir(path)):
        img = cv2.imread(path + file)
        H, W, _ = img.shape
        X, Y = H // (size // 2) + 1, W // (size // 2) + 1
        idx = 0
        rtn = np.ones((H, W, 3), dtype=np.float32) * -1
        for x in range(X-1):
            if x * size // 2 + size > H:
                break
            for y in range(Y-1):
                if y * size // 2 + size > W:
                    break
                img_tmp = cv2.imread(path_p + file[:-4] + '_%03d.png' % idx)
                weight_cur = copy.deepcopy(rtn[x * size // 2: x * size // 2 + size, y * size // 2: y * size // 2 + size, :])
                h1, w1, _ = weight_cur.shape
                gk_tmp = cv2.resize(gk, (w1, h1))
                weight_cur[weight_cur != -1] = gk_tmp[weight_cur != -1]
                weight_cur[weight_cur == -1] = 0
                weight_tmp = copy.deepcopy(weight_cur)
                weight_tmp = 1 - weight_tmp
                rtn[x * size // 2: x * size // 2 + size, y * size // 2: y * size // 2 + size, :] = weight_cur * rtn[x * size // 2: x * size // 2 + size, y * size // 2: y * size // 2 + size, :] + weight_tmp * img_tmp
                idx += 1
            img_tmp = cv2.imread(path_p + file[:-4] + '_%03d.png' % idx)
            weight_cur = copy.deepcopy(rtn[x * size // 2: x * size // 2 + size, -size:, :])
            h1, w1, _ = weight_cur.shape
            gk_tmp = cv2.resize(gk, (w1, h1))
            weight_cur[weight_cur != -1] = gk_tmp[weight_cur != -1]
            weight_cur[weight_cur == -1] = 0
            weight_tmp = copy.deepcopy(weight_cur)
            weight_tmp = 1 - weight_tmp
            rtn[x * size // 2: x * size // 2 + size, -size:, :] = weight_cur * rtn[x * size // 2: x * size // 2 + size, -size:, :] + weight_tmp * img_tmp
            idx += 1
        for y in range(Y - 1):
            if y * size // 2 + size > W:
                break
            img_tmp = cv2.imread(path_p + file[:-4] + '_%03d.png' % idx)
            weight_cur = copy.deepcopy(rtn[-size:, y * size // 2: y * size // 2 + size, :])
            h1, w1, _ = weight_cur.shape
            gk_tmp = cv2.resize(gk, (w1, h1))
            weight_cur[weight_cur != -1] = gk_tmp[weight_cur != -1]
            weight_cur[weight_cur == -1] = 0
            weight_tmp = copy.deepcopy(weight_cur)
            weight_tmp = 1 - weight_tmp
            rtn[-size:, y * size // 2: y * size // 2 + size, :] = weight_cur * rtn[-size:, y * size // 2: y * size // 2 + size, :] + weight_tmp * img_tmp
            idx += 1
        img_tmp = cv2.imread(path_p + file[:-4] + '_%03d.png' % idx)
        weight_cur = copy.deepcopy(rtn[-size:, -size:, :])
        h1, w1, _ = weight_cur.shape
        gk_tmp = cv2.resize(gk, (w1, h1))
        weight_cur[weight_cur != -1] = gk_tmp[weight_cur != -1]
        weight_cur[weight_cur == -1] = 0
        weight_tmp = copy.deepcopy(weight_cur)
        weight_tmp = 1 - weight_tmp
        rtn[-size:, -size:, :] = weight_cur * rtn[-size:, -size:, :] + weight_tmp * img_tmp
        idx += 1
        # rtn[rtn < 127] = 0
        # rtn[rtn >= 127] = 255
        cv2.imwrite(path_s + file[:-4] + '.png', np.uint8(rtn))
    return path_s


def gkern(kernlen=7, nsig=3):
    """Returns a 2D Gaussian kernel."""
    # x = np.linspace(-nsig, nsig, kernlen+1)
    # kern1d = np.diff(st.norm.cdf(x))
    # kern2d = np.outer(kern1d, kern1d)
    # rtn = kern2d/kern2d.sum()
    # rtn = np.concatenate([rtn[..., None], rtn[..., None], rtn[..., None]], axis=2)
    rtn = [[0, 0, 0],
           [0, 1, 0],
           [0, 0, 0]]
    rtn = np.array(rtn, dtype=np.float32)
    rtn = np.concatenate([rtn[..., None], rtn[..., None], rtn[..., None]], axis=2)
    rtn = cv2.resize(rtn, (kernlen, kernlen))
    return rtn

def rm_and_make_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
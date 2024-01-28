import copy

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from nets.EITLnet import SegFormer
from utils.utils import cvtColor, preprocess_input, resize_image

class SegFormer_Segmentation(object):
    _defaults = {
        "model_path": "",  # 和save_dir一致
        "num_classes": 2,
        "phi": "b2",
        "input_shape": [512, 512],
        "cuda": True,
    }

    # 初始化SegFormer
    def __init__(self,phi,path,**kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        if self.num_classes == 2:
            self.colors = [(0, 0, 0), (255, 255, 255)]
        if path!='':
            self.model_path=path
        if phi!='':
            self.phi=phi
        self.generate()

        # show_config(**self._defaults)

    def generate(self, onnx=False):
        self.net = SegFormer(num_classes=self.num_classes, phi=self.phi, dual=True, pretrained=False)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net.load_state_dict(torch.load(self.model_path, map_location=device,)['state_dict'],strict=False)
        self.net = self.net.eval()
        # print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    def detect_image_resize(self, image):
        image = cvtColor(image)

        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]

        # resize
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            pr = self.net(images)[0]

            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()

            pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh), \
                 int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]

            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)

            seg_pred = pr[:, :, 1]
            seg_pred_new = seg_pred.reshape(orininal_h, orininal_w)
            seg_pred_new = Image.fromarray(np.uint8(seg_pred_new * 255))

            pr = pr.argmax(axis=-1)

        seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
        image = Image.fromarray(np.uint8(seg_img))

        return image, seg_pred_new

    def detect_image_noresize(self, image):
        image = cvtColor(image)
        # 对输入图像进行一个备份，后面用于绘图
        old_img = copy.deepcopy(image)
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]

        # 添加batch_size维度
        image = np.expand_dims(np.transpose(preprocess_input(np.array(image, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image)
            if self.cuda:
                images = images.cuda()

            pr = self.net(images)[0]

            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
            seg_pred = pr[:, :, 1]
            seg_pred_new = seg_pred.reshape(orininal_h, orininal_w)
            seg_pred_new = Image.fromarray(np.uint8(seg_pred_new * 255))

            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
            pr = pr.argmax(axis=-1)

        return image, seg_pred_new
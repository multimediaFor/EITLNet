import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from utils.utils import preprocess_input, cvtColor
# import random
import copy
class SegmentationDataset_train(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path):
        super(SegmentationDataset_train, self).__init__()
        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.train = train
        self.dataset_path = dataset_path

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name = annotation_line.strip()
        ## Processing for jpg\png\tif format in the data set
        path_t = os.path.join(os.path.join(self.dataset_path, "JPEGImages"), name + ".jpg")
        path_t_png = os.path.join(os.path.join(self.dataset_path, "JPEGImages"), name + ".png")
        if os.path.isfile(path_t):
            img = Image.open(os.path.join(os.path.join(self.dataset_path, "JPEGImages"), name + ".jpg"))
        elif os.path.isfile(path_t_png):
            img = Image.open(os.path.join(os.path.join(self.dataset_path, "JPEGImages"), name + ".png"))
        else:
            img = Image.open(os.path.join(os.path.join(self.dataset_path, "JPEGImages"), name + ".tif"))
        
        path_gt=os.path.join(os.path.join(self.dataset_path, "SegmentationClass"), name + ".png")
        if os.path.isfile(path_gt):
            mask = Image.open(os.path.join(os.path.join(self.dataset_path, "SegmentationClass"), name + ".png"))
        else:
            mask = Image.open(os.path.join(os.path.join(self.dataset_path, "SegmentationClass"), name + ".tif"))
        # Data augmentation
        img, mask = self.get_random_data(img, mask, self.input_shape, random_flag = self.train)
        img = np.transpose(preprocess_input(np.array(img, np.float64)), [2,0,1])
        mask = np.array(mask)
        mask[mask >= self.num_classes] = self.num_classes

        seg_labels = np.eye(self.num_classes + 1)[mask.reshape([-1])]
        seg_labels = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))

        return img, mask, seg_labels

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, label, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random_flag=True):
        image = cvtColor(image)
        label = Image.fromarray(np.array(label))

        iw, ih = image.size
        h, w = input_shape

        if not random_flag:
            iw, ih = image.size
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)

            image = image.resize((nw,nh), Image.BICUBIC)
            new_image = Image.new('RGB', [w, h], (128,128,128))
            new_image.paste(image, ((w-nw)//2, (h-nh)//2))

            label = label.resize((nw,nh), Image.NEAREST)
            new_label = Image.new('L', [w, h], (0))
            new_label.paste(label, ((w-nw)//2, (h-nh)//2))
            return new_image, new_label
        # #resize
        new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(0.5, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)
        label = label.resize((nw, nh), Image.NEAREST)

        # RandomCrop
        if (nw < w) or (nh < h):
            dx = np.random.randint(0, max(0, nw - w)+1)
            dy = np.random.randint(0, max(0, nh - h)+1)
            new_w = max(w, nw)
            new_h = max(h, nh)
            new_image = Image.new('RGB', (new_w, new_h), (128, 128, 128))
            new_label = Image.new('L', (new_w, new_h), (0))
            new_image.paste(image, (dx, dy))
            new_label.paste(label, (dx, dy))
            image = copy.deepcopy(new_image)
            label = new_label

        x = np.random.randint(0, max(0, nw - w)+1)
        y = np.random.randint(0, max(0, nh - h)+1)
        image1 = np.array(image)
        label1 = np.array(label)

        # print("hi1", np.array(image).shape)
        image = image1[y:y + 512, x:x + 512]
        label = label1[y:y + 512, x:x + 512]

        # RandomFlip
        if np.random.random() < 0.5:
            image = cv2.flip(image, 1)
            label = cv2.flip(label, 1)
        else:
            image = image
            label = label

        # AddGaussianNoise
        h, w, c = image.shape
        noise = np.random.normal(0, 30, (h, w, c))
        image_data = np.clip(image + noise, 0, 255).astype(np.uint8)

        # GaussianBlur
        blur = self.rand() < 0.25
        if blur:
            image_data = cv2.GaussianBlur(image_data, (5, 5), 0)

        # Pil_jpg
        open_cv_image = cv2.cvtColor(np.array(image_data), cv2.COLOR_RGB2BGR)
        ret, buffer = cv2.imencode('.jpg', open_cv_image)
        decoded_image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        image_data = cv2.resize(decoded_image, (image_data.shape[1], image_data.shape[0]))


        return image_data, label

class SegmentationDataset_val(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path):
        super(SegmentationDataset_val, self).__init__()
        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.train = train
        self.dataset_path = dataset_path

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name = annotation_line.strip()
        ## Processing for jpg\png\tif format in the data set
        path_t = os.path.join(os.path.join(self.dataset_path, "JPEGImages"), name + ".jpg")
        path_t_png = os.path.join(os.path.join(self.dataset_path, "JPEGImages"), name + ".png")
        if os.path.isfile(path_t):
            img = Image.open(os.path.join(os.path.join(self.dataset_path, "JPEGImages"), name + ".jpg"))
        elif os.path.isfile(path_t_png):
            img = Image.open(os.path.join(os.path.join(self.dataset_path, "JPEGImages"), name + ".png"))
        else:
            img = Image.open(os.path.join(os.path.join(self.dataset_path, "JPEGImages"), name + ".tif"))
            
        path_gt=os.path.join(os.path.join(self.dataset_path, "SegmentationClass"), name + ".png")
        if os.path.isfile(path_gt):
            mask = Image.open(os.path.join(os.path.join(self.dataset_path, "SegmentationClass"), name + ".png"))
        else:
            mask = Image.open(os.path.join(os.path.join(self.dataset_path, "SegmentationClass"), name + ".tif"))
        

        img, mask = self.get_random_data(img, mask, self.input_shape, random_flag = self.train)
        img = np.transpose(preprocess_input(np.array(img, np.float64)), [2,0,1])
        mask = np.array(mask)
        mask[mask >= self.num_classes] = self.num_classes

        seg_labels = np.eye(self.num_classes + 1)[mask.reshape([-1])]
        seg_labels = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))

        return img, mask, seg_labels

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, label, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random_flag=True):
        image = cvtColor(image)
        label = Image.fromarray(np.array(label))

        iw, ih = image.size
        h, w = input_shape

        if not random_flag:
            iw, ih = image.size
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)

            image = image.resize((nw,nh), Image.BICUBIC)
            new_image = Image.new('RGB', [w, h], (128,128,128))
            new_image.paste(image, ((w-nw)//2, (h-nh)//2))

            label = label.resize((nw,nh), Image.NEAREST)
            new_label = Image.new('L', [w, h], (0))
            new_label.paste(label, ((w-nw)//2, (h-nh)//2))
            return new_image, new_label
        # #resize
        new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(0.5, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)
        label = label.resize((nw, nh), Image.NEAREST)

        # RandomCrop
        if (nw < w) or (nh < h):
            dx = np.random.randint(0, max(0, nw - w))
            dy = np.random.randint(0, max(0, nh - h))
            new_w = max(w, nw)
            new_h = max(h, nh)
            new_image = Image.new('RGB', (new_w, new_h), (128, 128, 128))
            new_label = Image.new('L', (new_w, new_h), (0))
            new_image.paste(image, (dx, dy))
            new_label.paste(label, (dx, dy))
            image = copy.deepcopy(new_image)
            label = new_label

        x = np.randint(0, max(0, nw - w))
        y = np.random.randint(0, max(0, nh - h))
        image = np.array(image)
        label = np.array(label)

        # print("hi1", np.array(image).shape)
        image_data = image[y:y + 512, x:x + 512]
        label = label[y:y + 512, x:x + 512]


        return image_data, label

def seg_dataset_collate(batch):
    images = []
    masks = []
    seg_labels = []
    for img, mask, labels in batch:
        images.append(img)
        masks.append(mask)
        seg_labels.append(labels)
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    masks = torch.from_numpy(np.array(masks)).long()
    seg_labels = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
    return images, masks, seg_labels

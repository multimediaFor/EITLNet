# -*- coding: utf-8 -*-

import os
import numpy as np
import datetime
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.train_utils import get_lr_scheduler, set_optimizer_lr, weights_init, fit_one_epoch
from utils.callbacks import LossHistory, EvalCallback
from utils.dataloader import SegmentationDataset_train, seg_dataset_collate, SegmentationDataset_val
from nets.EITLnet import SegFormer



def get_net(num_classes=2, phi='b2', pretrained=True, dual=True):
    model = SegFormer(num_classes=num_classes, phi=phi, pretrained=pretrained, dual=dual)
    return model

if __name__ == "__main__":
    Cuda = True
    num_classes = 2
    phi = "b2" #b0
    pretrained = True
    model_path = ''
    input_shape = [512, 512]

    dual = True

    init_epoch = 0
    total_epoch = 100
    batch_size = 8

    Init_lr = 0.0005
    Min_lr = Init_lr * 0.01

    optimizer_type = "adamw"
    momentum = 0.9
    weight_decay = 1e-2

    # Optional parameter='step'、'cos'
    lr_decay_type = 'cos'

    # logs
    save_period = 5
    save_dir = r'./log/b2_network/'

    eval_flag = True
    eval_period = 1

    # dataset
    data_path = r'./train_dataset/'

    # loss
    dice_loss = True
    focal_loss = True
    cls_weights = np.ones([num_classes], np.float32)

    num_workers = 8
    ngpus_per_node = torch.cuda.device_count()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    local_rank = 0

    model = get_net(num_classes=2, phi=phi, pretrained=pretrained, dual=dual)
    if not pretrained:
        weights_init(model)
    if model_path != '':
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))
        checkpoint = torch.load(model_path, map_location=device)
        model_dict = model.state_dict()
        pretrained_dict = checkpoint['state_dict']
        init_epoch = checkpoint['epoch']
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))

    if local_rank == 0:
        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history = None

    model_train = model.train()

    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    with open(os.path.join(data_path, "ImageSets/Segmentation/train.txt"), "r") as f:
        train_lines = f.readlines()
    with open(os.path.join(data_path, "ImageSets/Segmentation/val.txt"), "r") as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)


    nbs = 16
    lr_limit_max = 1e-4 if optimizer_type in ['adam', 'adamw'] else 5e-2
    lr_limit_min = 3e-5 if optimizer_type in ['adam', 'adamw'] else 5e-4
    Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    optimizer = {
        'adam': optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
        'adamw': optim.AdamW(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
        'sgd': optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True,
                         weight_decay=weight_decay)
    }[optimizer_type]
    if model_path != '':
        optimizer.load_state_dict(checkpoint['optimizer'])

    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, total_epoch)

    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size

    train_dataset = SegmentationDataset_train(train_lines, input_shape, num_classes, True, data_path)
    val_dataset = SegmentationDataset_val(val_lines, input_shape, num_classes, False, data_path)

    train_sampler = None
    val_sampler = None
    shuffle = True

    gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                     pin_memory=False,
                     drop_last=True, collate_fn=seg_dataset_collate, sampler=train_sampler)
    gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                         pin_memory=False,
                         drop_last=True, collate_fn=seg_dataset_collate, sampler=val_sampler)

    if local_rank == 0:
        eval_callback = EvalCallback(model, input_shape, num_classes, val_lines, data_path, log_dir, Cuda, \
                                     eval_flag=eval_flag, period=eval_period)
    else:
        eval_callback = None

    for epoch in range(init_epoch, total_epoch):
        nbs = 16
        lr_limit_max = 1e-4 if optimizer_type in ['adam', 'adamw'] else 5e-2
        lr_limit_min = 3e-5 if optimizer_type in ['adam', 'adamw'] else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, total_epoch)

        for param in model.backbone.parameters():
            param.requires_grad = True

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                         pin_memory=True,
                         drop_last=True, collate_fn=seg_dataset_collate, sampler=train_sampler)
        gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=True,
                             drop_last=True, collate_fn=seg_dataset_collate, sampler=val_sampler)

        UnFreeze_flag = True

        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

        fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val,
                      gen, gen_val, total_epoch, Cuda,
                      dice_loss, focal_loss, cls_weights, num_classes, save_period, save_dir,
                      local_rank)

    if local_rank == 0:
        loss_history.writer.close()

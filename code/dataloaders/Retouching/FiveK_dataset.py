import os
import torch
import random

import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from PIL import Image
from torch.utils.data import Dataset


class FiveK_train_dataset(Dataset):
    def __init__(self, cfg, expert='c'):
        self.cfg = cfg
        self.expert = expert
        self.flip = cfg.dataset['random_flip']
        self.crop = cfg.dataset['random_crop']
        self.color = cfg.dataset['color_jitter']
        self.rotate = cfg.dataset['random_rotate']

        root_dir = os.path.join(cfg.dataset_root, 'Retouching/FiveK')

        datalist_txt = os.path.join(root_dir, f'test.txt')
        with open(datalist_txt, 'r') as fid:
            datanames = fid.readlines()
        testnames = [dataname.strip() for dataname in datanames]

        img_dir = os.path.join(root_dir, 'input/JPG/sizeShort480')
        gt_dir = os.path.join(root_dir, f'expert{expert.capitalize()}/JPG/sizeShort480')
        datalist = os.listdir(img_dir)

        self.img_list = list()
        self.gt_list = list()
        for img_name in datalist:
            name, ext = os.path.splitext(img_name)
            tag = name.split('-')[0]
            if tag not in testnames:
                self.img_list.append(os.path.join(img_dir, f'{img_name}'))
                self.gt_list.append(os.path.join(gt_dir, f'{img_name}'))

        self.transform = transforms.Compose([transforms.ToTensor()])
        self.normalize = transforms.Normalize(mean=cfg.dataset['mean'], std=cfg.dataset['std'])

    def get_image(self, idx, flip=0, rotate=0, crop=0, color=0):
        img = Image.open(os.path.join(self.img_list[idx])).convert("RGB")
        gt = Image.open(os.path.join(self.gt_list[idx])).convert("RGB")

        # Augmentations
        # random horizontal flip
        if flip == 1:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            gt = gt.transpose(Image.FLIP_LEFT_RIGHT)

        # random rotate
        if rotate != 0:
            img = img.rotate(90 * rotate, expand=1)
            gt = gt.rotate(90 * rotate, expand=1)

        # random crop
        if crop == 1:
            img, gt = self.random_crop(img, gt)

        # color jitter
        if color == 1:
            brightness = np.random.uniform(0.9, 1.1)
            saturation = np.random.uniform(0.9, 1.1)
            img = TF.adjust_brightness(img, brightness)
            img = TF.adjust_saturation(img, saturation)

        return self.transform(img), self.transform(gt)

    def random_crop(self, img, gt):
        w, h = img.size
        ratio_h = np.random.uniform(0.8, 1.0)
        ratio_w = np.random.uniform(0.8, 1.0)

        crop_h = round(h * ratio_h)
        crop_w = round(w * ratio_w)
        i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(crop_h, crop_w))

        img_crop = TF.crop(img, i, j, h, w)
        gt_crop = TF.crop(gt, i, j, h, w)
        return img_crop, gt_crop

    def __getitem__(self, idx):
        seed = random.randint(0, 2 ** 32)
        np.random.seed(seed)
        flip = np.random.randint(0, 2) if self.flip else 0
        crop = np.random.randint(0, 2) if self.crop else 0
        color = np.random.randint(0, 2) if self.color else 0
        rotate = np.random.randint(0, 4) if self.rotate else 0

        img_path = self.img_list[idx]
        img_name = os.path.split(img_path)[1]
        img, gt = self.get_image(idx, flip=flip, rotate=rotate, crop=crop, color=color)
        batch = {'img_path': img_path,
                 'img_name': img_name,
                 'img': self.normalize(img),
                 'gt': self.normalize(gt)}
        return batch

    def __len__(self):
        return len(self.img_list)


class FiveK_test_dataset(Dataset):
    def __init__(self, cfg, expert='c', resol='480p'):
        assert resol in ['480p', '4k']
        self.cfg = cfg
        self.expert = expert

        root_dir = os.path.join(cfg.dataset_root, f'Retouching/FiveK')

        datalist_txt = os.path.join(root_dir, f'test.txt')
        with open(datalist_txt, 'r') as fid:
            datanames = fid.readlines()
        testnames = [dataname.strip() for dataname in datanames]

        if resol == '480p':
            img_dir = os.path.join(root_dir, 'input/JPG/sizeShort480')
            gt_dir = os.path.join(root_dir, f'expert{expert.capitalize()}/JPG/sizeShort480')
        else:
            img_dir = os.path.join(root_dir, 'input/JPG/sizeOri')
            gt_dir = os.path.join(root_dir, f'expert{expert.capitalize()}/JPG/sizeOri')
        datalist = os.listdir(img_dir)

        self.img_list = list()
        self.gt_list = list()
        for img_name in datalist:
            name, ext = os.path.splitext(img_name)
            tag = name.split('-')[0]
            if tag in testnames:
                self.img_list.append(os.path.join(img_dir, f'{img_name}'))
                self.gt_list.append(os.path.join(gt_dir, f'{img_name}'))

        self.transform = transforms.Compose([transforms.ToTensor()])
        self.normalize = transforms.Normalize(mean=cfg.dataset['mean'], std=cfg.dataset['std'])

    def get_image(self, idx):
        img = Image.open(os.path.join(self.img_list[idx])).convert("RGB")
        gt = Image.open(os.path.join(self.gt_list[idx])).convert("RGB")
        return self.transform(img), self.transform(gt)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        img_name = os.path.split(img_path)[1]
        img, gt = self.get_image(idx)
        batch = {'img_path': img_path,
                 'img_name': img_name,
                 'img': self.normalize(img),
                 'gt': self.normalize(gt)}
        return batch

    def __len__(self):
        return len(self.img_list)

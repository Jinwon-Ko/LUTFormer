import os
import torch
import random

import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from PIL import Image
from torch.utils.data import Dataset


class EUVP_train_dataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.flip = cfg.dataset['random_flip']
        self.crop = cfg.dataset['random_crop']
        self.color = cfg.dataset['color_jitter']
        self.rotate = cfg.dataset['random_rotate']

        train_dirs = [os.path.join(cfg.dataset_root, 'Underwater/EUVP/underwater_imagenet'),
                      os.path.join(cfg.dataset_root, 'Underwater/EUVP/underwater_scenes'),
                      os.path.join(cfg.dataset_root, 'Underwater/EUVP/underwater_dark')]

        self.img_list = list()
        self.gt_list = list()
        for train_dir in train_dirs:
            img_dir = os.path.join(train_dir, 'trainA')
            gt_dir = os.path.join(train_dir, 'trainB')
            img_list = os.listdir(img_dir)

            for img_name in img_list:
                self.img_list.append(os.path.join(img_dir, f'{img_name}'))
                self.gt_list.append(os.path.join(gt_dir, f'{img_name}'))

        self.transform = transforms.Compose([transforms.ToTensor()])
        self.normalize = transforms.Normalize(mean=cfg.dataset['mean'], std=cfg.dataset['std'])

    def get_image(self, idx, flip=0, rotate=0, color=0):
        img = Image.open(os.path.join(self.img_list[idx])).convert("RGB")
        gt = Image.open(os.path.join(self.gt_list[idx])).convert("RGB")

        # Augmentations
        # random horizontal flip
        if flip == 1:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            gt = gt.transpose(Image.FLIP_LEFT_RIGHT)

        # random crop
        img, gt = self.random_crop(img, gt)

        # random rotate
        if rotate != 0:
            img = img.rotate(90 * rotate, expand=1)
            gt = gt.rotate(90 * rotate, expand=1)

        # color jitter
        if color == 1:
            brightness = np.random.uniform(0.9, 1.1)
            saturation = np.random.uniform(0.9, 1.1)
            img = TF.adjust_brightness(img, brightness)
            img = TF.adjust_saturation(img, saturation)

        return self.transform(img), self.transform(gt)

    def random_crop(self, img, gt):
        width, height = img.size
        if (width < 256) or (height < 256):
            img = TF.resize(img, size=[256, 256])
            gt = TF.resize(gt, size=[256, 256])

        i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(256, 256))

        img_crop = TF.crop(img, i, j, h, w)
        gt_crop = TF.crop(gt, i, j, h, w)
        return img_crop, gt_crop

    def __getitem__(self, idx):
        seed = random.randint(0, 2 ** 32)
        np.random.seed(seed)
        flip = np.random.randint(0, 2) if self.flip else 0
        color = np.random.randint(0, 2) if self.color else 0
        rotate = np.random.randint(0, 4) if self.rotate else 0

        img_path = self.img_list[idx]
        img_name = os.path.split(img_path)[1]
        img, gt = self.get_image(idx, flip=flip, rotate=rotate, color=color)
        batch = {'img_path': img_path,
                 'img_name': img_name,
                 'img': self.normalize(img),
                 'gt': self.normalize(gt)}
        return batch

    def __len__(self):
        return len(self.img_list)


class EUVP_test_dataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg

        test_dir = os.path.join(cfg.dataset_root, 'Underwater/EUVP/test_samples')

        img_dir = os.path.join(test_dir, 'Inp')
        gt_dir = os.path.join(test_dir, 'GTr')
        datalist = os.listdir(img_dir)

        self.img_list = list()
        self.gt_list = list()
        for img_name in datalist:
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

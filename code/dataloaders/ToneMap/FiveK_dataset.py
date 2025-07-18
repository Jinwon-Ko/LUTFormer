import os
import cv2
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

        img_dir = os.path.join(root_dir, 'input/PNG/480p_16bits_XYZ_WB')
        gt_dir = os.path.join(root_dir, f'expert{expert.capitalize()}/JPG/sizeShort480')
        datalist = os.listdir(img_dir)

        self.img_list = list()
        self.gt_list = list()
        for img_name in datalist:
            name, ext = os.path.splitext(img_name)
            tag = name.split('-')[0]
            if tag not in testnames:
                self.img_list.append(os.path.join(img_dir, f'{img_name}'))
                self.gt_list.append(os.path.join(gt_dir, f'{name}.jpg'))

        self.transform = transforms.Compose([transforms.ToTensor()])
        self.normalize = transforms.Normalize(mean=cfg.dataset['mean'], std=cfg.dataset['std'])

    def get_image(self, idx, flip=0, rotate=0, crop=0, color=0):
        img = cv2.imread(os.path.join(self.img_list[idx]), -1)
        gt = Image.open(os.path.join(self.gt_list[idx])).convert("RGB")
        img = np.array(img)[:, :, [2, 1, 0]]

        # Augmentations
        # random horizontal flip
        if flip == 1:
            img = cv2.flip(img, 1)
            gt = gt.transpose(Image.FLIP_LEFT_RIGHT)

        # random rotate
        if rotate != 0:
            rot = [None, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_CLOCKWISE]
            img = cv2.rotate(img, rot[rotate])
            gt = gt.rotate(90 * rotate, expand=1)

        # random crop
        if crop == 1:
            img, gt = self.random_crop(img, gt)

        img = img.astype(np.float32) / 65535
        img = torch.from_numpy(img.transpose(2, 0, 1))

        # color jitter
        if color == 1:
            brightness = np.random.uniform(0.6, 1.4)
            img = TF.adjust_brightness(img, brightness)

        return img, self.transform(gt)

    def random_crop(self, img, gt):
        w, h = gt.size
        ratio_h = np.random.uniform(0.9, 1.0)
        ratio_w = np.random.uniform(0.9, 1.0)

        crop_h = round(h * ratio_h)
        crop_w = round(w * ratio_w)
        i, j, h, w = transforms.RandomCrop.get_params(gt, output_size=(crop_h, crop_w))

        img_crop = img[i:i + h, j:j + w, :]
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

        img_dir = os.path.join(root_dir, 'input/PNG/480p_16bits_XYZ_WB')
        gt_dir = os.path.join(root_dir, f'expert{expert.capitalize()}/JPG/sizeShort480')
        datalist = os.listdir(img_dir)

        self.img_list = list()
        self.gt_list = list()
        for img_name in datalist:
            name, ext = os.path.splitext(img_name)
            tag = name.split('-')[0]
            if tag in testnames:
                self.img_list.append(os.path.join(img_dir, f'{img_name}'))
                self.gt_list.append(os.path.join(gt_dir, f'{name}.jpg'))

        self.transform = transforms.Compose([transforms.ToTensor()])
        self.normalize = transforms.Normalize(mean=cfg.dataset['mean'], std=cfg.dataset['std'])

    def get_image(self, idx):
        img = cv2.imread(os.path.join(self.img_list[idx]), -1)
        gt = Image.open(os.path.join(self.gt_list[idx])).convert("RGB")
        img = np.array(img)[:, :, [2, 1, 0]]

        img = img.astype(np.float32) / 65535
        img = torch.from_numpy(img.transpose(2, 0, 1))
        return img, self.transform(gt)

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

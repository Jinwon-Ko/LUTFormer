import os
import cv2
import random
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from PIL import Image
from torch.utils.data import Dataset


class PPR10K_train_dataset(Dataset):
    def __init__(self, cfg, expert='c'):
        self.cfg = cfg
        self.expert = expert
        self.flip = cfg.dataset['random_flip']
        self.crop = cfg.dataset['random_crop']
        self.color = cfg.dataset['color_jitter']
        self.rotate = cfg.dataset['random_rotate']

        data_root = os.path.join(cfg.dataset_root, f'Retouching/PPR10K/train')

        img_dir = os.path.join(data_root, 'source_aug')
        gt_dir = os.path.join(data_root, f'target_{expert}')
        mask_dir = os.path.join(data_root, f'masks')
        img_list = os.listdir(img_dir)
        gt_list = os.listdir(gt_dir)

        self.img_list = list()
        self.gt_list = list()
        self.mask_list = list()
        for img_name in img_list:
            name, ext = os.path.splitext(img_name)
            if len(img_list) == len(gt_list):
                self.img_list.append(os.path.join(img_dir, f'{img_name}'))
                self.gt_list.append(os.path.join(gt_dir, f'{img_name}'))
                self.mask_list.append(os.path.join(mask_dir, f'{name}.png'))
            else:
                split_name = name.split('_')
                if len(split_name) == 2:
                    self.img_list.append(os.path.join(img_dir, f'{img_name}'))
                    self.gt_list.append(os.path.join(gt_dir, f'{img_name}'))
                    self.mask_list.append(os.path.join(mask_dir, f'{name}.png'))
                else:
                    source_name = '_'.join(split_name[:-1])
                    self.img_list.append(os.path.join(img_dir, f'{img_name}'))
                    self.gt_list.append(os.path.join(gt_dir, f'{source_name}{ext}'))
                    self.mask_list.append(os.path.join(mask_dir, f'{source_name}.png'))

        self.transform = transforms.Compose([transforms.ToTensor()])
        self.normalize = transforms.Normalize(mean=cfg.dataset['mean'], std=cfg.dataset['std'])

    def get_image(self, idx, flip=0, rotate=0, crop=0, color=0):
        mask = Image.open(os.path.join(self.mask_list[idx])).convert("RGB")
        img = cv2.imread(os.path.join(self.img_list[idx]), -1)
        gt = cv2.imread(os.path.join(self.gt_list[idx]), -1)
        img = np.array(img)[:, :, [2, 1, 0]]
        gt = np.array(gt)[:, :, [2, 1, 0]]

        if flip == 1:
            img = cv2.flip(img, 1)
            gt = cv2.flip(gt, 1)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        if crop == 1:
            img, gt, mask = self.random_crop(img, gt, mask)

        # resize
        img = cv2.resize(img, (448, 448))
        gt = cv2.resize(gt, (448, 448))
        mask = TF.resize(mask, (448, 448))

        img = img.astype(np.float32) / 65535
        img = torch.from_numpy(img.transpose(2, 0, 1))

        gt = gt.astype(np.float32) / 255
        gt = torch.from_numpy(gt.transpose(2, 0, 1))

        return img, gt, self.transform(mask)

    def random_crop(self, img, gt, mask):
        w, h = mask.size
        ratio_h = np.random.uniform(0.6, 1.0)
        ratio_w = np.random.uniform(0.6, 1.0)

        crop_h = round(h * ratio_h)
        crop_w = round(w * ratio_w)
        i, j, h, w = transforms.RandomCrop.get_params(mask, output_size=(crop_h, crop_w))

        img_crop = img[i:i + h, j:j + w, :]
        gt_crop = gt[i:i + h, j:j + w, :]
        mask_crop = TF.crop(mask, i, j, h, w)

        return img_crop, gt_crop, mask_crop

    def __getitem__(self, idx):
        seed = random.randint(0, 2 ** 32)
        np.random.seed(seed)
        flip = np.random.randint(0, 2) if self.flip else 0
        crop = np.random.randint(0, 2) if self.crop else 0
        color = np.random.randint(0, 2) if self.color else 0
        rotate = np.random.randint(0, 4) if self.rotate else 0

        img_path = self.img_list[idx]
        img_name = os.path.split(img_path)[1]
        img, gt, mask = self.get_image(idx, flip=flip, rotate=rotate, crop=crop, color=color)

        return {'img_path': img_path,
                'img_name': img_name,
                'img': self.normalize(img),
                'gt': self.normalize(gt),
                'mask': self.normalize(mask)}

    def __len__(self):
        return len(self.img_list)


class PPR10K_test_dataset(Dataset):
    def __init__(self, cfg, expert='c'):
        self.cfg = cfg
        self.expert = expert

        root_dir = os.path.join(cfg.dataset_root, f'Retouching/PPR10K/val')

        img_dir = os.path.join(root_dir, 'source')
        gt_dir = os.path.join(root_dir, f'target_{expert}')
        mask_dir = os.path.join(root_dir, f'masks')
        datalist = os.listdir(img_dir)

        self.img_list = list()
        self.gt_list = list()
        self.mask_list = list()
        for img_name in datalist:
            name, ext = os.path.splitext(img_name)
            self.img_list.append(os.path.join(img_dir, f'{img_name}'))
            self.gt_list.append(os.path.join(gt_dir, f'{img_name}'))
            self.mask_list.append(os.path.join(mask_dir, f'{name}.png'))

        self.transform = transforms.Compose([transforms.ToTensor()])
        self.normalize = transforms.Normalize(mean=cfg.dataset['mean'], std=cfg.dataset['std'])

    def get_image(self, idx):
        mask = Image.open(os.path.join(self.mask_list[idx])).convert("RGB")
        img = cv2.imread(os.path.join(self.img_list[idx]), -1)
        gt = cv2.imread(os.path.join(self.gt_list[idx]), -1)
        img = np.array(img)[:, :, [2, 1, 0]]
        gt = np.array(gt)[:, :, [2, 1, 0]]

        img = img.astype(np.float32) / 65535
        img = torch.from_numpy(img.transpose(2, 0, 1))

        gt = gt.astype(np.float32) / 255
        gt = torch.from_numpy(gt.transpose(2, 0, 1))

        return img, gt, self.transform(mask)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        img_name = os.path.split(img_path)[1]
        img, gt, mask = self.get_image(idx)

        return {'img_path': img_path,
                'img_name': img_name,
                'img': self.normalize(img),
                'gt': self.normalize(gt),
                'mask': self.normalize(mask)}

    def __len__(self):
        return len(self.img_list)

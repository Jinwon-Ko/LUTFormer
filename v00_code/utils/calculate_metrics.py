import os
import cv2
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from utils.util import to_np, to_tensor
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


class Evaluator(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.mean = np.array([cfg.dataset['mean']], dtype=np.float32)
        self.std = np.array([cfg.dataset['std']], dtype=np.float32)
        self.mse = nn.MSELoss()

    def measure_PSNR(self, pred, gt):
        assert len(pred) == len(gt)

        PSNR = 0
        for b_idx in range(len(pred)):
            image_true = self.make_data2img(gt[b_idx])
            image_test = self.make_data2img(pred[b_idx])
            PSNR += psnr(image_true, image_test)
        return PSNR

    def make_data2img(self, data):
        data = to_np(data.permute(1, 2, 0))
        data = (data * self.std + self.mean)
        data = np.clip(data, a_min=0.0, a_max=1.0)
        data = np.round(data * 255)
        data = np.uint8(data)[:, :, [2, 1, 0]]
        return data


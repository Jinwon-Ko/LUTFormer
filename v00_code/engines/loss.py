import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from utils.util import *


class Loss_Function(nn.Module):
    def __init__(self, cfg, weight_dict):
        super().__init__()
        self.cfg = cfg
        self.weight_dict = weight_dict

        self.loss_L1 = nn.L1Loss()
        self.loss_L2 = nn.MSELoss()
        self.relu = nn.ReLU()

        perceptual_backbone = models.vgg16(pretrained=True)
        self.mean = nn.Parameter(torch.tensor([0.485, 0.456, 0.406]), requires_grad=False)
        self.std = nn.Parameter(torch.tensor([0.229, 0.224, 0.225]), requires_grad=False)

        blocks = []
        blocks.append(perceptual_backbone.features[:4].eval())
        blocks.append(perceptual_backbone.features[4:9].eval())
        blocks.append(perceptual_backbone.features[9:16].eval())
        blocks.append(perceptual_backbone.features[16:23].eval())
        blocks.append(perceptual_backbone.features[23:30].eval())
        self.blocks = nn.ModuleList(blocks)
        for name, p in self.named_parameters():
            p.requires_grad = False

    def get_reconsturction_loss(self, pred, gt, l_name='recon'):
        return {l_name: self.weight_dict['loss_recon'] * self.loss_L2(pred, gt)}

    def get_grad_loss(self, pred, gt, l_name='grad'):
        grad_y_pred = (pred[:, :, 1:, :] - pred[:, :, :-1, :])      # H
        grad_x_pred = (pred[:, :, :, 1:] - pred[:, :, :, :-1])      # W

        grad_y_gt = (gt[:, :, 1:, :] - gt[:, :, :-1, :])            # H
        grad_x_gt = (gt[:, :, :, 1:] - gt[:, :, :, :-1])            # W

        grad_y = self.loss_L2(grad_y_pred, grad_y_gt)               # H
        grad_x = self.loss_L2(grad_x_pred, grad_x_gt)               # W
        return {l_name: self.weight_dict['loss_recon'] * (grad_y + grad_x)}

    def get_perceptual_loss(self, pred, gt, feature_layers=[0, 1, 2, 3], l_name='perceptual'):
        x = (pred - self.mean.view(1, -1, 1, 1)) / self.std.view(1, -1, 1, 1)
        y = (gt - self.mean.view(1, -1, 1, 1)) / self.std.view(1, -1, 1, 1)

        loss = 0.0
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += self.loss_L1(x, y)
        return {l_name: self.weight_dict['loss_perceptual'] * loss / len(pred)}

    def get_LUT_range_constraint_loss(self, LUT, l_name='constraint'):
        upper_bound = LUT - 1
        lower_bound = 0 - LUT
        constraint = (torch.sum(self.relu(upper_bound)) / ((upper_bound > 0).sum() + 1e-6) +
                      torch.sum(self.relu(lower_bound)) / ((lower_bound > 0).sum() + 1e-6))
        constraint = self.weight_dict['loss_constraint'] * constraint
        return {l_name: constraint}

    def get_LUT_regularization_mn_loss(self, LUT, l_name='regular_mn'):
        # LUT : [[R, G, B], B_idx, G_idx, R_idx]
        diff_r = LUT[:, 0, :, :, :-1] - LUT[:, 0, :, :, 1:]
        diff_g = LUT[:, 1, :, :-1, :] - LUT[:, 1, :, 1:, :]
        diff_b = LUT[:, 2, :-1, :, :] - LUT[:, 2, 1:, :, :]

        mn = (torch.sum(self.relu(diff_r)) / ((diff_r > 0).sum() + 1e-6) +
              torch.sum(self.relu(diff_g)) / ((diff_g > 0).sum() + 1e-6) +
              torch.sum(self.relu(diff_b)) / ((diff_b > 0).sum() + 1e-6))
        return {l_name: self.weight_dict['loss_regular_mn'] * mn}

    def get_LUT_regularization_tv_loss(self, LUT, l_name='regular_tv'):
        # LUT : [[R, G, B], B_idx, G_idx, R_idx]
        diff_r = LUT[:, [1, 2], :, :, :-1] - LUT[:, [1, 2], :, :, 1:]
        diff_g = LUT[:, [0, 2], :, :-1, :] - LUT[:, [0, 2], :, 1:, :]
        diff_b = LUT[:, [0, 1], :-1, :, :] - LUT[:, [0, 1], 1:, :, :]

        tv = torch.mean((diff_r ** 2)) + torch.mean(diff_g ** 2) + torch.mean(diff_b ** 2)
        return {l_name: self.weight_dict['loss_regular_tv'] * tv}


def get_loss_function(args):
    weight_dict = dict()
    weight_dict['loss_recon'] = args.training['loss_recon_coef']
    weight_dict['loss_perceptual'] = args.training['loss_perceptual_coef']
    # weight_dict['loss_constraint'] = args.training['loss_constraint_coef']
    # weight_dict['loss_regular_mn'] = args.training['loss_regular_mn_coef']
    # weight_dict['loss_regular_tv'] = args.training['loss_regular_tv_coef']
    criterion = Loss_Function(args, weight_dict=weight_dict)
    return criterion

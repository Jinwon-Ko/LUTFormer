import math
import torch
import random

from engines.utils_train import update_and_logging, logging
from utils.util import to_np, logger, save_final_model
from utils.calculate_metrics import Evaluator
from utils.viz_utils import Visualizer


def get_loss_names(cfg):
    loss_names = ['recon', 'grad', 'perceptual']    #, 'regular_mn', 'regular_tv', 'constraint'
    loss_t = {'Total': 0}
    loss_t.update({key: 0 for key in loss_names})
    return loss_names, loss_t


def get_total_loss(cfg, criterion, outputs, gt, l_names=[]):
    loss_dict = {}
    for l_name in l_names:
        if l_name == 'recon':
            enhanced = outputs['enhanced']
            if cfg.dataset_name == 'PPR10K':
                gt = gt * outputs['weights']
                enhanced = enhanced * outputs['weights']
            loss_dict.update(criterion.get_reconsturction_loss(enhanced, gt, l_name=l_name))
        elif l_name == 'grad':
            loss_dict.update(criterion.get_grad_loss(outputs['enhanced'], gt, l_name=l_name))
        elif l_name == 'perceptual':
            loss_dict.update(criterion.get_perceptual_loss(outputs['enhanced'], gt, l_name=l_name))

        elif l_name == 'constraint':
            loss_dict.update(criterion.get_LUT_range_constraint_loss(outputs['low_density_LUT'], l_name=l_name))
        elif l_name == 'regular_mn':
            loss_dict.update(criterion.get_LUT_regularization_mn_loss(outputs['low_density_LUT'], l_name=l_name))
        elif l_name == 'regular_tv':
            loss_dict.update(criterion.get_LUT_regularization_tv_loss(outputs['low_density_LUT'], l_name=l_name))

    return loss_dict


def train_one_epoch(cfg, epoch, model, train_loader, criterion, optimizer):
    print('Epoch %03d' % epoch)
    model.train()
    criterion.train()
    viz_tools = Visualizer(cfg)
    eval_tools = Evaluator(cfg)

    num = 0
    psnr = 0
    l_names, loss_t = get_loss_names(cfg)
    for i, batch in enumerate(train_loader):
        # Load data
        img = batch['img'].cuda()
        gt = batch['gt'].cuda()
        img_path = batch['img_path']
        img_name = batch['img_name']

        # Forward model
        outputs = model(img)
        if cfg.dataset_name == 'PPR10K':
            mask = batch['mask'].cuda()
            mask = torch.sum(mask, dim=1, keepdim=True)
            weights = torch.ones_like(mask)
            weights[mask > 0] = 3
            outputs.update({'weights': weights})

        # Backpropagation
        loss_dict = get_total_loss(cfg, criterion, outputs, gt, l_names=l_names)
        losses = sum(loss_dict[k] for k in loss_dict.keys())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # Visualize
        if cfg.viz and (i % 10 == 0):
            viz_contents = {'img': img[0],
                            'gt': gt[0],
                            'enhanced': outputs['enhanced'][0]}
            viz_tools.viz_train(cfg, viz_contents, n_iter=num)

        psnr += eval_tools.measure_PSNR(outputs['enhanced'], gt)

        # Logging
        num += img.shape[0]
        loss_t, log = update_and_logging(loss_t, loss_dict)
        loss_t['Total'] += losses.item()
        txt = 'Total: %.5f  %s' % (losses.item(), log)
        print('[Epoch %d][%d/%d][Losses %s]' % (epoch, i, len(train_loader), txt), end='\r')

    # logging
    log = logging(loss_t, num)
    logger("[Epoch %d Average Losses] %s\n" % (epoch, log), f'{cfg.save_dir}/losses_image_enhancement.txt')
    print('\n[Epoch %d Average Losses] %s' % (epoch, log))
    save_final_model(cfg, model, optimizer, epoch)

    PSNR = psnr / num
    logger("[Epoch %d]  Train ==> PSNR %5f\n" % (epoch, PSNR), f'{cfg.save_dir}/train_performances.txt')
    print('[Epoch %d]  Train ==> PSNR %5f' % (epoch, PSNR))
    return model, optimizer, loss_t



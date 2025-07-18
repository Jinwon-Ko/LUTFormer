import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from engines.utils_train import update_dict, update_LRs, save_losses_and_performances
from dataloaders.factory import load_dataset
from engines.train.forward_train import train_one_epoch
from engines.test.forward_test import evaluation


def do_train_process(cfg, model, criterion, optimizer, lr_scheduler):
    train_dataset = load_dataset(cfg, mode='train')
    test_dataset = load_dataset(cfg, mode='test')
    train_loader = DataLoader(train_dataset, batch_size=cfg.dataset['batch_size'], num_workers=cfg.dataset['num_workers'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=cfg.dataset['num_workers'], shuffle=False)

    start_epoch = 0
    best = {'PSNR': 0, 'COST': 10}
    Losses = {'Total': [], 'recon': [], 'grad': [], 'perceptual': []}   # , 'regular_mn': [], 'regular_tv': [], 'constraint': []
    Performances = {'PSNR': []}
    LRs = []

    if cfg.load:
        checkpoint = torch.load(cfg.checkpoint['model'])
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        start_epoch = checkpoint['epoch']
        checkpoint = None

    model.cuda()
    criterion.cuda()

    if cfg.load:
        best, now = evaluation(cfg, start_epoch-1, model, test_loader, best)
        lr_scheduler.step(start_epoch-1)
        
    for epoch in range(start_epoch, cfg.training['epochs']):
        LRs = update_LRs(LRs, optimizer.param_groups[0]['lr'])
        model, optimizer, losses = train_one_epoch(cfg, epoch, model, train_loader, criterion, optimizer)
        best, now = evaluation(cfg, epoch, model, test_loader, best)
        lr_scheduler.step()

        Losses, Performances = update_dict(Losses, losses, Performances, now)
        save_losses_and_performances(cfg, Losses, Performances['PSNR'], LRs)

    print('best_Performances : ', best)



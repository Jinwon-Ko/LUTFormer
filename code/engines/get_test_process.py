import os
import torch

from torch.utils.data import DataLoader
from dataloaders.factory import load_dataset
from engines.test.forward_test import analysis, test_speed


def do_test_process(cfg, model):
    if cfg.load:
        checkpoint = torch.load(cfg.checkpoint['model'])
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        checkpoint = None

    if cfg.run_mode == 'test_paper':
        checkpoint = torch.load(cfg.checkpoint['model'])
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        checkpoint = None

    model.cuda()

    test_dataset = load_dataset(cfg, mode='test')    # load_train_dataset(cfg), load_eval_dataset(cfg, task=task)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=cfg.dataset['num_workers'], shuffle=False)
    analysis(cfg, model, test_loader)


def do_check_runtime(cfg, model):
    if cfg.load:
        checkpoint = torch.load(cfg.checkpoint['model'])
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        checkpoint = None

    model.cuda()
    test_speed(cfg, model)

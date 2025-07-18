import os
import math
import time
import torch
import numpy as np

from utils.calculate_metrics import Evaluator
from utils.util import to_np, save_best_model
from utils.viz_utils import Visualizer



def evaluation(cfg, epoch, model, test_loader, best):
    if not epoch >= cfg.testing['start_eval_epoch']:
        return best

    model.eval()

    eval_tools = Evaluator(cfg)
    viz_tools = Visualizer(cfg)

    n = 0
    psnr = 0
    for i, batch in enumerate(test_loader):
        print('[Epoch: %d][%d/%d]' % (epoch, i, len(test_loader)), end='\r')

        # load data
        img = batch['img'].cuda()
        gt = batch['gt'].cuda()
        img_path = batch['img_path']
        img_name = batch['img_name']

        with torch.no_grad():
            outputs = model(img)

        psnr += eval_tools.measure_PSNR(outputs['enhanced'], gt)
        n += len(img)

        if cfg.viz:
            viz_contents = {'img': img[0],
                            'gt': gt[0],
                            'enhanced': outputs['enhanced'][0]}
            viz_tools.viz_test(cfg, viz_contents, img_name[0])

    PSNR = psnr / n
    print('%s Test ==> PSNR %5f' % (cfg.dataset_name, PSNR))

    now = {'PSNR': PSNR}
    best = save_best_model(cfg, model, epoch, now, best, metric='PSNR')
    return best, now


def analysis(cfg, model, test_loader):
    model.eval()

    eval_tools = Evaluator(cfg)
    viz_tools = Visualizer(cfg)

    identity = model.LUT_estimator.identity
    viz_tools.viz_identity_LUT(cfg, identity)

    n = 0
    psnr = 0
    with torch.no_grad():
        torch.cuda.empty_cache()

        for i, batch in enumerate(test_loader):
            print('Processing [%04d/%04d]...' % (i, len(test_loader)), end='\r')
            # load data
            img = batch['img'].cuda()
            gt = batch['gt'].cuda()
            img_path = batch['img_path']
            img_name = batch['img_name']

            outputs = model(img)
            n += len(img)

            PSNR = eval_tools.measure_PSNR(outputs['enhanced'], gt)
            psnr += PSNR
            if cfg.viz:
                viz_contents = {'img': img[0],
                                'gt': gt[0],
                                'enhanced': outputs['enhanced'][0]}
                viz_tools.viz_analysis(cfg, viz_contents, img_name=img_name[0], mode='one') # mode : ['one', 'dict']

    PSNR = psnr / n
    print('%s ==> [Prediction] Total [PSNR %.5f]' % (cfg.dataset_name, PSNR))
    print('# of trainable parameters : %.3f K' % (sum(p.numel() for p in model.parameters() if p.requires_grad) * 0.001))


def test_speed(cfg, model):
    resolutions = {'480p': (480, 640),
                   '720p': (720, 1280),
                   '4k': (2160, 3840),
                   '8k': (4320, 7680)}

    with torch.no_grad():
        torch.cuda.empty_cache()

        dummy = torch.randn(1, 3, 256, 256).cuda()
        for _ in range(20):
            _ = model(dummy)

        for resol, size in resolutions.items():
            n = 0
            spend_time = 0
            img = torch.rand(1, 3, size[0], size[1]).cuda()

            for i in range(10):
                print(f'Processing {resol} resolution... [{i:02d}/{10}]', end='\r')

                torch.cuda.synchronize()
                t0 = time.time()

                for _ in range(10):
                    _ = model(img)
                    n += 1

                torch.cuda.synchronize()
                spend_time += (time.time() - t0)

            runtime = spend_time / n
            print(f'[{resol} resolution] Runtime in millisecond: {runtime * 1000:.5f} ms')

    print('# of trainable parameters : %.3f K' % (sum(p.numel() for p in model.parameters() if p.requires_grad) * 0.001))
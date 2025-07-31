import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from utils.util import to_np, to_tensor


############################################
class Visualizer(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.mean = np.array([cfg.dataset['mean']], dtype=np.float32)
        self.std = np.array([cfg.dataset['std']], dtype=np.float32)
        self.show = {}

    def update_image(self, img, name='img'):
        try:
            img = to_np(img.permute(1, 2, 0))
            img = (img * self.std + self.mean)
            img = np.clip(img, a_min=0.0, a_max=1.0)
            if 'error' in name:
                img = img ** 0.5
            img = np.round(img * 255)
            img = np.uint8(img)[:, :, [2, 1, 0]]
            if 'error' in name:
                img  = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        except:
            pass

        self.show[name] = img

    def saveimg_one(self, dir_name, file_name, show_name):
        name, ext = os.path.splitext(file_name)
        os.makedirs(dir_name, exist_ok=True)
        cv2.imwrite(os.path.join(dir_name, name + '.png'), self.show[show_name])

    def saveimg_list(self, dir_name, file_name, show_list):
        # boundary line

        if self.show[show_list[0]].shape[-1] == 3:
            line = np.zeros((self.show[show_list[0]].shape[0], 3, 3), dtype=np.uint8)
            line[:, :, :] = 255
        else:
            line = np.zeros((self.show[show_list[0]].shape[0], 3), dtype=np.uint8)
            line[:, :] = 255
        disp = line

        for i in range(len(show_list)):
            if show_list[i] not in self.show.keys():
                continue
            disp = np.concatenate((disp, self.show[show_list[i]], line), axis=1)

        os.makedirs(dir_name, exist_ok=True)
        cv2.imwrite(os.path.join(dir_name, file_name), disp)

    def saveimg_dict(self, dir_name, file_name, show_dict):
        for idx, key in enumerate(show_dict.keys()):
            show_list = show_dict[key]

            # boundary line
            if self.show[show_list[0]].shape[-1] == 3:
                line = np.zeros((self.show[show_list[0]].shape[0], 3, 3), dtype=np.uint8)
                line[:, :, :] = 255
            else:
                line = np.zeros((self.show[show_list[0]].shape[0], 3), dtype=np.uint8)
                line[:, :] = 255

            # stack images by column direction
            col_disp = line
            for i in range(len(show_list)):
                if show_list[i] not in self.show.keys():
                    continue
                col_disp = np.concatenate((col_disp, self.show[show_list[i]], line), axis=1)

            # stack images by row direction
            self.row_line = np.ones((3, col_disp.shape[1], 3), dtype=np.uint8) * 255
            if idx == 0:
                disp = self.row_line
            disp = np.concatenate((disp, col_disp, self.row_line), axis=0)

        # save image
        os.makedirs(dir_name, exist_ok=True)
        cv2.imwrite(os.path.join(dir_name, file_name), disp)

    def viz_train(self, cfg, viz_contents, n_iter):
        ######################################################################################
        # Draw input & output & GT
        self.update_image(img=viz_contents['img'], name='img')
        self.update_image(img=viz_contents['gt'], name='gt')
        self.update_image(img=viz_contents['enhanced'], name='enhanced')
        self.update_image(img=abs(viz_contents['gt'] - viz_contents['enhanced']), name='error')

        show_dict = {'row1': ['img', 'enhanced'],
                     'row2': ['gt', 'error']}

        dir_name = os.path.join(cfg.viz_dir, f'train/results')
        self.saveimg_dict(dir_name=os.path.join(dir_name), file_name=f'iter_{n_iter:04d}.jpg', show_dict=show_dict)

    def viz_test(self, cfg, viz_contents, img_name):
        ######################################################################################
        # Draw input & output & GT
        self.update_image(img=viz_contents['img'], name='img')
        self.update_image(img=viz_contents['gt'], name='gt')
        self.update_image(img=viz_contents['enhanced'], name='enhanced')
        self.update_image(img=abs(viz_contents['gt'] - viz_contents['enhanced']), name='error')

        show_dict = {'row1': ['img', 'enhanced'],
                     'row2': ['gt', 'error']}

        name, ext = os.path.splitext(img_name)
        dir_name = os.path.join(cfg.viz_dir, f'test/results')
        self.saveimg_dict(dir_name=os.path.join(dir_name), file_name=name + '.jpg', show_dict=show_dict)

    def viz_analysis(self, cfg, viz_contents, img_name, mode='dict'):
        ######################################################################################
        # Draw input & output & GT
        self.update_image(img=viz_contents['img'], name='img')
        self.update_image(img=viz_contents['gt'], name='gt')
        self.update_image(img=viz_contents['enhanced'], name='enhanced')
        self.update_image(img=abs(viz_contents['gt'] - viz_contents['enhanced']), name='error')

        show_dict = {'row1': ['img', 'enhanced'],
                     'row2': ['gt', 'error']}

        name, ext = os.path.splitext(img_name)
        if mode == 'dict':
            dir_name = os.path.join(cfg.viz_dir, f'analysis/results')
            self.saveimg_dict(dir_name=os.path.join(dir_name), file_name=name + '.jpg', show_dict=show_dict)

        elif mode == 'one':
            self.saveimg_one(dir_name=os.path.join(cfg.viz_dir, f'analysis/img'), file_name=img_name, show_name='img')
            self.saveimg_one(dir_name=os.path.join(cfg.viz_dir, f'analysis/gt'), file_name=name + '.png', show_name='gt')
            self.saveimg_one(dir_name=os.path.join(cfg.viz_dir, f'analysis/pred'), file_name=name + '.png', show_name='enhanced')

    def viz_demo(self, cfg, enhanced, img_name):
        ######################################################################################
        # Draw input & output & GT
        name, ext = os.path.splitext(img_name)
        self.update_image(img=enhanced, name='enhanced')
        self.saveimg_one(dir_name=cfg.output_dir, file_name=name + '.png', show_name='enhanced')

    def viz_identity_LUT(self, cfg, identity):
        N = cfg.network['n_points']

        # LUT : [[R, G, B], B_idx, G_idx, R_idx]
        R = torch.linspace(0., 1., N)
        G = torch.linspace(0., 1., N)
        B = torch.linspace(0., 1., N)
        Z, Y, X = torch.meshgrid(R, G, B, indexing='ij')
        X = X.flatten()
        Y = Y.flatten()
        Z = Z.flatten()

        LUT = torch.clamp(identity, min=0., max=1.)
        LUT = to_np(LUT.flatten(1).T)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X, Y, Z, c=LUT, marker='o', alpha=0.5)

        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        ax.set_zlim([-0.05, 1.05])
        ax.set_xlabel('R')
        ax.set_ylabel('G')
        ax.set_zlabel('B')

        img_name = f'identity_LUT{N}.png'
        dir_name = os.path.join(cfg.viz_dir)
        os.makedirs(dir_name, exist_ok=True)

        plt.savefig(os.path.join(dir_name, img_name))
        plt.close()

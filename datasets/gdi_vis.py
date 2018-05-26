#!/usr/bin/env python

import collections
import os

import numpy as np
import PIL.Image
import scipy.io
import torch
from torch.utils import data

class GDI_Vis_Base(data.Dataset):

    mean_bgr = np.array([104.00699, 116.66877, 122.67892])

    def __init__(self, split='train', transform=False, binarize=False):
        self.split = split
        self._transform = transform
        self.binarize = binarize

    def list_files(self, dataset_dir, image_dir, imp_dir, image_ext):
        self.files = collections.defaultdict(list)
        for split in ['train', 'valid']:
            imgsets_file = os.path.join(dataset_dir, '%s.txt' % split)
            for did in open(imgsets_file):
                did = did.strip()
                img_file = os.path.join(dataset_dir, image_dir, '{}.{}'.format(did, image_ext))
                lbl_file = os.path.join(dataset_dir, imp_dir, '{}.png'.format(did))
                self.files[split].append({
                    'img': img_file,
                    'lbl': lbl_file,
                })

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        data_file = self.files[self.split][index]
        # load image
        img_file = data_file['img']
        img = PIL.Image.open(img_file)
        img = np.array(img, dtype=np.uint8)
        assert len(img.shape) == 3 # assumes color images and no alpha channel

        # load label
        lbl_file = data_file['lbl']
        lbl = PIL.Image.open(lbl_file)
        lbl = np.array(lbl)
        assert lbl.max() < 256
        lbl = lbl.astype(np.uint8)
        if self._transform:
            return self.transform(img, lbl), img_file, lbl_file
        else:
            return img, lbl, img_file, lbl_file

    def transform(self, img, lbl):
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float32)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1) # C x H x W

        if self.binarize:
            lbl = lbl > 255.0 * 2 / 3
        else:
            lbl = lbl / 255.0
        lbl = lbl[:, :, None]  # add channel axis
        lbl = lbl.transpose(2, 0, 1) # C x H x W
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).float()
        return img, lbl

    def untransform(self, img, lbl):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img += self.mean_bgr
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]
        lbl = lbl.numpy()
        lbl = lbl[0, :, :]
        if self.binarize:
            lbl = (lbl * 3 / 2) >= (1 - 0.0001) * 255 # lbl > 255.0 * 2 / 3
        else:
            lbl = lbl * 255.0
        lbl = lbl.astype(np.uint8)
        return img, lbl


class GDI(GDI_Vis_Base):
    def __init__(self, root, image_dir, imp_dir, split='train', transform=False, binarize=False):
        super(GDI, self).__init__(split=split, transform=transform, binarize=binarize)

        image_ext = 'jpg'
        dataset_dir = os.path.join(root, 'gdi')
        self.list_files(dataset_dir, image_dir, imp_dir, image_ext)


class Massvis(GDI_Vis_Base):
    def __init__(self, root, image_dir, imp_dir, split='train', transform=False, binarize=False):
        super(Massvis, self).__init__(split=split, transform=transform, binarize=binarize)

        image_ext = 'png'
        dataset_dir = os.path.join(root, 'massvis')
        self.list_files(dataset_dir, image_dir, imp_dir, image_ext)

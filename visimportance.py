#!/usr/bin/env python

import argparse
# import datetime
import os
import sys
# import shlex
# import subprocess
# import pickle

import torch
import yaml
import torch.nn as nn

from datasets.gdi_vis import GDI, Massvis
import models
from trainer import Trainer
import utils

configurations = {
    # massvis FCN32s
    1: dict(
        max_iteration=100000,
        lr=1.0e-4,
        momentum=0.9,
        weight_decay=0.0005,
        iter_size=1,
        gamma=0.1,
        step_size=20000,
        interval_validate=200,
    ),
    # GDI FCN32s
    2: dict(
        max_iteration=100000,
        lr=1.0e-3,
        momentum=0.9,
        weight_decay=0.0005,
        iter_size=20,
        gamma=0.1,
        step_size=5000,
        interval_validate=200,
    ),
}

def get_parameters(model, bias=False, fcn_type='fcn32'):
    for k, m in model._modules.items():
        if k == "score_sal" and isinstance(m, nn.Conv2d):
            if bias:
                yield m.bias
            else:
                yield m.weight
        if fcn_type == 'fcn16':
            if k == "score_pool4" and isinstance(m, nn.Conv2d):
                if bias:
                    yield m.bias
                else:
                    yield m.weight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='gdi', help='name of dataset, gdi or massvis (default: gdi)')
    parser.add_argument('--dataset_dir', type=str, default='/path/to/datase_dirt', help='dataset directory')
    parser.add_argument('--fcn_type', type=str, help='FCN type, fcn32 or fcn16 (default: gdi)', default='fcn32',
                        choices=['fcn32', 'fcn16'])
    parser.add_argument('--overlaid_img_dir', type=str, default='/path/to/overlaid_img_dir',
                        help='output directory path for images with heatpmap overlaid onto input images')
    parser.add_argument('--pretrained_model', type=str, default='/path/to/pretrained_model',
                        help='pretrained model converted from Caffe models')
    parser.add_argument('--config', type=int, default=1, choices=configurations.keys(),
                        help='configuration for training where several hyperparameters are defined')
    parser.add_argument('--log_file', type=str, default='F:/dataset/visimportance/log', help='/path/to/log_file')
    parser.add_argument('--resume', type=str, default='',
                        help='checkpoint file to be loaded when retraining models')
    parser.add_argument('--checkpoint_dir', type=str, default='/path/to/checkpoint_dir',
                        help='checkpoint file to be saved in each epoch')
    parser.add_argument('--eval_only', action='store_true', help='evaluation only')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id (default: 0)')
    args = parser.parse_args()

    utils.create_dir(os.path.join(args.overlaid_img_dir, "train"))
    utils.create_dir(os.path.join(args.overlaid_img_dir, "valid"))
    if not args.eval_only:
        utils.create_dir(args.checkpoint_dir)
    print(args)

    gpu = args.gpu
    cfg = configurations[args.config]
    log_file = args.log_file
    resume = args.resume

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    cuda = torch.cuda.is_available()
    args.cuda = cuda
    if args.cuda:
        print("torch.backends.cudnn.version: {}".format(torch.backends.cudnn.version()))

    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)

    # 1. dataset

    root = os.path.expanduser(args.dataset_dir)
    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    train_loader = None
    if not args.eval_only: # training + validation
        if args.dataset == 'gdi':
            dt = GDI(root, image_dir="gd_train", imp_dir="gd_imp_train", split='train', transform=True)
        else:
            dt = Massvis(root, image_dir="train", imp_dir="train_imp", split='train', transform=True)
        train_loader = torch.utils.data.DataLoader(dt, batch_size=1, shuffle=True, **kwargs)
        print("no of images in training", len(train_loader))

    if args.dataset == 'gdi': # validation
        dv = GDI(root, image_dir="gd_val", imp_dir="gd_imp_val", split='valid', transform=True)
    else:
        dv = Massvis(root, image_dir="valid", imp_dir="valid_imp", split='valid', transform=True)
    val_loader = torch.utils.data.DataLoader(dv, batch_size=1, shuffle=False, **kwargs)
    print("no of images in evaluation", len(val_loader))


    # 2. model

    model = models.FCN32s() if args.fcn_type == 'fcn32' else models.FCN16s()

    start_epoch = 0
    start_iteration = 0
    if resume:
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        start_iteration = checkpoint['iteration']
        if args.fcn_type == 'fcn32':
            assert checkpoint['arch'] == 'FCN32s'
        else:
            assert checkpoint['arch'] == 'FCN16s'
    else:
        if args.fcn_type in ['fcn32', 'fcn16']:
            model_weight = torch.load(args.pretrained_model)
            model.load_state_dict(model_weight)
            if not args.eval_only:
                model._initialize_weights()
        else:
            fcn32s = models.FCN32s()
            checkpoint = torch.load(args.pretrained_model)
            fcn32s.load_state_dict(checkpoint['model_state_dict'])
            model.copy_params_from_fcn32s(fcn32s)
            model._initialize_weights()

    if cuda:
        model = model.cuda()

    # 3. optimizer

    optim = torch.optim.SGD(
        [
            {'params': get_parameters(model, bias=False, fcn_type=args.fcn_type)},
            {'params': get_parameters(model, bias=True, fcn_type=args.fcn_type), 'lr': cfg['lr'] * 2, 'weight_decay': 0},
        ],
        lr=cfg['lr'],
        momentum=cfg['momentum'],
        weight_decay=cfg['weight_decay'])
    if resume:
        optim.load_state_dict(checkpoint['optim_state_dict'])

    # lr_policy: step
    last_epoch = start_iteration if resume else -1
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim,  cfg['step_size'], gamma=cfg['gamma'], last_epoch=last_epoch)

    trainer = Trainer(
        cuda=cuda,
        model=model,
        optimizer=optim,
        lr_scheduler=lr_scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir=args.checkpoint_dir,
        log_file=log_file,
        max_iter=cfg['max_iteration'],
        iter_size=cfg['iter_size'],
        interval_validate=cfg.get('interval_validate', len(train_loader)) if not args.eval_only else 0,
        overlaid_img_dir=args.overlaid_img_dir,
        dataset=args.dataset,
        eval_only=args.eval_only,
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    if not args.eval_only:
        trainer.train()
    else:
        trainer.validate()


if __name__ == '__main__':
    main()

import datetime
import math
import os
import shutil
import psutil
import gc

import numpy as np
from scipy.special import expit
import scipy.misc
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

import utils
import tqdm

class Trainer(object):

    def __init__(self, cuda, model, optimizer,
                 train_loader, val_loader, checkpoint_dir, log_file, max_iter, iter_size=1,
                 size_average=True, interval_validate=None, lr_scheduler=None, overlaid_img_dir=None,
                 dataset=None, eval_only=False):

        self.cuda = cuda
        self.eval_only = eval_only

        self.model = model
        self.optim = optimizer
        self.optim.zero_grad()
        self.lr_scheduler = lr_scheduler

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.timestamp_start = datetime.datetime.now()
        self.size_average = size_average

        if interval_validate is None:
            self.interval_validate = len(self.train_loader)
        else:
            self.interval_validate = interval_validate

        self.epoch = 0
        self.iteration = 0
        self.bwd_counter = 0

        self.max_iter = max_iter
        self.best_mean_rmse = 1e+20
        self.best_mean_r2 = 0
        self.iter_size = iter_size
        self.dataset = dataset

        self.overlaid_img_dir = overlaid_img_dir
        self.checkpoint_dir = checkpoint_dir
        self.log_file = log_file

        self.log_headers = [
            'epoch',
            'iteration',
            'train/fname',
            'train/loss',
            'train/kl',
            'train/kl_01',
            'train/cc',
            'train/rmse',
            'train/r2',
            'train/spearman',
            'valid/fname',
            'valid/loss',
            'valid/kl',
            'valid/kl_01',
            'valid/cc',
            'valid/rmse',
            'valid/r2',
            'valid/spearman',
            'elapsed_time',
        ]
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                f.write(','.join(self.log_headers) + '\n')

    def print_log(self, image_name, loss, metrics, is_valid=True):
        with open(self.log_file, 'a') as f:
            elapsed_time = (datetime.datetime.now() - self.timestamp_start).total_seconds()
            if is_valid:
                log = [self.epoch, self.iteration] + [image_name] + [''] * 8 + [loss] + list(metrics) + [elapsed_time]
            else:
                log = [self.epoch, self.iteration] + [image_name] + [loss] + list(metrics) + [''] * 8 + [elapsed_time]
            log = map(str, log)
            f.write(','.join(log) + '\n')


    def validate(self):
        training = self.model.training
        self.model.eval()
        metrics = []

        val_loss_sum = 0
        for batch_idx, ((data, target), data_files, target_files) in tqdm.tqdm(
                enumerate(self.val_loader), total=len(self.val_loader),
                desc='Valid iteration={} epoch={}'.format(self.iteration, self.epoch), ncols=80, leave=False):

            gc.collect()
            assert data.size(0) == 1, "Set batch size to one for validation!"

            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)

            score = self.model(data)
            loss = nn.BCEWithLogitsLoss(size_average=self.size_average)(score, target)

            if np.isnan(float(loss.data[0])):
                raise ValueError('loss is nan while validating')
            val_loss = float(loss.data[0])
            val_loss_sum += val_loss

            imgs = data.data.cpu()
            lbl_preds = (expit(score.data.cpu().numpy()) * 255).astype(np.uint8)
            lbl_trues = target.data.cpu()
            for img, lbl_true, lbl_pred, data_file, target_file in zip(imgs, lbl_trues, lbl_preds, data_files, target_files):
                img, lbl_true = self.val_loader.dataset.untransform(img, lbl_true)
                lbl_pred = lbl_pred[0]
                assert lbl_true.ndim == 2 and lbl_pred.ndim == 2
                if self.overlaid_img_dir is not None:
                    image_name, _ = os.path.splitext(os.path.split(data_file)[1])
                    fname = os.path.join(self.overlaid_img_dir, "valid", image_name + "_target.png")
                    utils.overlay_imp_on_img(img, lbl_true, fname, colormap='jet')
                    fname = os.path.join(self.overlaid_img_dir, "valid", image_name + "_{:05d}.png".format(self.epoch))
                    utils.overlay_imp_on_img(img, lbl_pred, fname, colormap='jet')

                kl, kl_01, cc, rmse, r2, spearman = utils.label_accuracy(lbl_true, lbl_pred)
                metrics.append((kl, kl_01, cc, rmse, r2, spearman))
                # print("\nkl, kl_01, cc, rmse, r2, spearman", kl, kl_01, cc, rmse, r2, spearman)

                self.print_log(image_name, val_loss, metrics[-1], is_valid=True)


        metrics = np.mean(metrics, axis=0)
        print("valid metrics:", metrics)

        val_loss_sum /= len(self.val_loader)
        self.print_log("summary_valid", val_loss_sum, metrics, is_valid=True)

        if self.eval_only:
            return

        mean_rmse, mean_r2 = metrics[3],  metrics[4]
        is_best = mean_rmse < self.best_mean_rmse
        self.best_mean_rmse = min(mean_rmse, self.best_mean_rmse)
        self.best_mean_r2 = max(mean_r2, self.best_mean_r2)
        checkpoint_file = os.path.join(self.checkpoint_dir, 'checkpoint-{}.pth.tar'.format(self.dataset))
        torch.save({
            'epoch': self.epoch,
            'iteration': self.iteration,
            'arch': self.model.__class__.__name__,
            'metrics': metrics,
            'optim_state_dict': self.optim.state_dict(),
            'model_state_dict': self.model.state_dict(),
            'best_mean_rmse': self.best_mean_rmse,
            'best_mean_r2': self.best_mean_r2,
        }, checkpoint_file)
        if is_best:
            shutil.copy(checkpoint_file, os.path.join(self.checkpoint_dir, 'model_best-{}.pth.tar'.format(self.dataset)))
        if (self.epoch + 1) % 10 == 0:
            shutil.copy(checkpoint_file,
                        os.path.join(self.checkpoint_dir, 'checkpoint-{}-{}.pth.tar'.format(self.dataset, self.epoch)))

        if training:
            self.model.train()


    def train_epoch(self):
        # https://discuss.pytorch.org/t/how-to-implement-accumulated-gradient/3822/8
        self.model.train()
        self.optim.zero_grad()

        loss_sum = 0
        metrics = []

        for batch_idx, ((data, target), data_files, target_files) in tqdm.tqdm(
                enumerate(self.train_loader), total=len(self.train_loader),
                desc='Train epoch={}, iter={}'.format(self.epoch, self.iteration), ncols=80, leave=False):
            iteration = batch_idx + self.epoch * len(self.train_loader)

            gc.collect()

            if self.iteration != 0 and (iteration - 1) != self.iteration:
                continue
            self.iteration = iteration

            if self.iteration % self.interval_validate == 0:
                self.validate()

            assert self.model.training
            assert data.size(0) == 1, "Set batch size to one for training!"

            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)

            score = self.model(data)
            assert target.data.cpu().numpy().min() >= 0 and target.data.cpu().numpy().max() <= 1
            loss = nn.BCEWithLogitsLoss(size_average=self.size_average)(score, target)

            loss = loss / self.iter_size
            if np.isnan(float(loss.data[0])):
                raise ValueError('loss is nan while training')
            train_loss = float(loss.data[0])
            loss_sum += train_loss * self.iter_size

            loss.backward()
            self.bwd_counter += 1

            if self.bwd_counter % self.iter_size == 0:
                # https://github.com/intel/caffe/blob/master/src/caffe/solver.cpp#L269
                self.optim.step()
                self.optim.zero_grad()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            imgs = data.data.cpu()
            lbl_preds =(expit(score.data.cpu().numpy()) * 255).astype(np.uint8)
            lbl_trues = target.data.cpu()
            for img, lbl_true, lbl_pred, data_file, target_file in zip(imgs, lbl_trues, lbl_preds, data_files, target_files):
                img, lbl_true = self.train_loader.dataset.untransform(img, lbl_true)
                lbl_pred = lbl_pred[0]
                assert lbl_true.ndim == 2 and lbl_pred.ndim == 2
                if self.overlaid_img_dir is not None:
                    image_name, _ = os.path.splitext(os.path.split(data_file)[1])
                    fname = os.path.join(self.overlaid_img_dir, "train", image_name + "_target.png")
                    utils.overlay_imp_on_img(img, lbl_true, fname, colormap='jet')
                    fname = os.path.join(self.overlaid_img_dir, "train", image_name + "_{:05d}.png".format(self.epoch))
                    utils.overlay_imp_on_img(img, lbl_pred, fname, colormap='jet')


                kl, kl_01, cc, rmse, r2, spearman = utils.label_accuracy(lbl_true, lbl_pred)
                # print("\nkl, kl_01, cc, rmse, r2, spearman", kl, kl_01, cc, rmse, r2, spearman)

                metrics.append((kl, kl_01, cc, rmse, r2, spearman))
                self.print_log(image_name, train_loss, metrics[-1], is_valid=False)

        metrics = np.mean(metrics, axis=0)
        print("train metrics:", metrics)

        loss_sum /= len(self.train_loader)
        self.print_log("summary_train", loss_sum, metrics, is_valid=False)


    def train(self):
        max_epoch = int(math.ceil(1. * self.max_iter / len(self.train_loader)))
        for epoch in tqdm.trange(self.epoch, max_epoch, desc='Train', ncols=80):
            self.epoch = epoch
            self.train_epoch()
            if self.iteration >= self.max_iter:
                break

#!/usr/bin/env python3

# Copyright 2020 The Johns Hopkins University (author: Jiatong Shi)


import os
import torch
import numpy as np
import copy
import time
import librosa
from scipy import signal
from librosa.display import specshow
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(train_loader, model, device, optimizer, criterion, args):
    losses = AverageMeter()
    model.train()
    start = time.time()
    for step, (phone, mean_list, duration, length) in enumerate(train_loader, 1):
        phone = phone.to(device)
        mean_list = mean_list.to(device).float()
        duration = duration.to(device)
        length = length.to(device).long()
        length_mask = length.ne(0).float().to(device)


        if args.model_type == "Transformer":
            output, att = model(phone, mean_list, pos=length)
        elif args.model_type == "LSTM":
            output, _ = model(phone, mean_list, src_key_padding_mask=length)
        elif args.model_type == "DNN":
            phone = phone.float()
            output = model(phone, mean_list)

        train_loss = criterion(output, duration, length_mask)

        optimizer.zero_grad()
        train_loss.backward()
        if args.gradclip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradclip)
        if args.optimizer == "noam":
            optimizer.step_and_update_lr()
        else:
            optimizer.step()
        losses.update(train_loss.item(), phone.size(0))
        if step % 500 == 0:
            end = time.time()
            print("step {}: train_loss {} -- sum_time: {}s".format(step, losses.avg, end - start))

    info = {'loss': losses.avg}
    return info


def validate(dev_loader, model, device, criterion, args):
    losses = AverageMeter()
    model.eval()

    with torch.no_grad():
        for step, (phone, mean_list, duration, length) in enumerate(dev_loader, 1):
            phone = phone.to(device)
            mean_list = mean_list.to(device).float()
            duration = duration.to(device)
            length = length.to(device).long()
            length_mask = length.ne(0).float().to(device)

            if args.model_type == "Transformer":
                output, att = model(phone, mean_list, pos=length)
            elif args.model_type == "LSTM":
                output, _ = model(phone, mean_list, src_key_padding_mask=length)
            elif args.model_type == "DNN":
                phone = phone.float()
                output = model(phone, mean_list)
            val_loss = criterion(output, duration, length_mask)
            losses.update(val_loss.item(), phone.size(0))
            if step % 100 == 0 and args.model_type == "Transformer":
                length = length.cpu().detach().numpy()[0]
                att = att.cpu().detach().numpy()[0]
                att = att[:, :length, :length]
                plt.subplot(1, 4, 1)
                specshow(att[0])
                plt.subplot(1, 4, 2)
                specshow(att[1])
                plt.subplot(1, 4, 3)
                specshow(att[2])
                plt.subplot(1, 4, 4)
                specshow(att[3])
                plt.savefig(os.path.join(args.model_save_dir, '{}_att.png'.format(step)))
                print("step {}: {}".format(step, losses.avg))

    info = {'loss': losses.avg}
    return info


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, model_filename):
    torch.save(state, model_filename)
    return 0


def record_info(train_info, dev_info, epoch, logger):
    loss_info = {
        "train_loss": train_info['loss'],
        "dev_loss": dev_info['loss']}
    logger.add_scalars("losses", loss_info, epoch)
    return 0



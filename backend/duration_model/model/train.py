#!/usr/bin/env python3

# Copyright 2020 The Johns Hopkins University (author: Jiatong Shi)


import os
import sys
import numpy as np
import torch
import time
from model.gpu_util import use_single_gpu
from model.duration_dataset import DurationDataset, DurationCollator
from model.network import TransformerDuration
from model.transformer_optim import ScheduledOptim
from model.loss import MaskedLoss
from model.utils import train_one_epoch, save_checkpoint, validate, record_info


def train(args):
    if args.gpu > 0 and torch.cuda.is_available():
        cvd = use_single_gpu()
        print(f"GPU {cvd} is used")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = DurationDataset(duration_file=args.train,
                             max_len=args.max_len)

    dev_set = DurationDataset(duration_file=args.val,
                             max_len=args.max_len)

    collate_fn = DurationCollator(args.max_len)
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=args.batchsize,
                                               shuffle=True,
                                               num_workers=args.num_workers,
                                               collate_fn=collate_fn,
                                               pin_memory=True)
    dev_loader = torch.utils.data.DataLoader(dataset=dev_set,
                                               batch_size=args.batchsize,
                                               shuffle=True,
                                               num_workers=args.num_workers,
                                               collate_fn=collate_fn,
                                               pin_memory=True)
    # prepare model
    if args.model_type == "Transformer":
        model = TransformerDuration(dim_feedforward=args.dim_feedforward,
                                phone_size=args.phone_size,
                                embed_size=args.embedding_size,
                                d_model=args.hidden_size,
                                dropout=args.dropout,
                                d_output=1,
                                nhead=args.nhead,
                                num_block=args.num_block,
                                pos_enc=True)
    else:
        raise ValueError('Not Support Model Type %s' % args.model_type)
    print(model)
    model = model.to(device)

    # load weights for pre-trained model
    if args.initmodel != '':
        pretrain = torch.load(args.initmodel, map_location=device)
        pretrain_dict = pretrain['state_dict']
        model_dict = model.state_dict()
        state_dict_new = {}
        para_list = []
        for k, v in pretrain_dict.items():
            assert k in model_dict
            if model_dict[k].size() == pretrain_dict[k].size():
                state_dict_new[k] = v
            else:
                para_list.append(k)
        print("Total {} parameters, Loaded {} parameters".format(
            len(pretrain_dict), len(state_dict_new)))
        if len(para_list) > 0:
            print("Not loading {} because of different sizes".format(
                ", ".join(para_list)))
        model_dict.update(state_dict_new)
        model.load_state_dict(model_dict)
        print("Loaded checkpoint {}".format(args.initmodel))
        print("")


    # setup optimizer
    if args.optimizer == 'noam':
        optimizer = ScheduledOptim(torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.98),
            eps=1e-09),
            args.hidden_size,
            args.noam_warmup_steps,
            args.noam_scale)
    else:
        raise ValueError('Not Support Optimizer')

    # Setup tensorborad logger
    if args.use_tfboard:
        from tensorboardX import SummaryWriter
        logger = SummaryWriter("{}/log".format(args.model_save_dir))
    else:
        logger = None

    if args.loss == "l1":
        loss = MaskedLoss("l1")
    elif args.loss == "mse":
        loss = MaskedLoss("mse")
    else:
        raise ValueError("Not Support Loss Type")
    
    # Training
    for epoch in range(1, 1 + args.max_epochs):
        start_t_train = time.time()
        train_info = train_one_epoch(train_loader, model, device, optimizer, loss, args)
        end_t_train = time.time()

        print(
            'Train epoch: {:04d}, lr: {:.6f}, '
            'loss: {:.4f}, time: {:.2f}s'.format(
                epoch, optimizer._optimizer.param_groups[0]['lr'],
                train_info['loss'], end_t_train - start_t_train))

        start_t_dev = time.time()
        dev_info = validate(dev_loader, model, device, loss, args)
        end_t_dev = time.time()

        print("Valid loss: {:.4f}, time: {:.2f}s".format(
            dev_info['loss'], end_t_dev - start_t_dev))
        print("")
        sys.stdout.flush()
        
        if not os.path.exists(args.model_save_dir):
            os.makedirs(args.model_save_dir)

        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer._optimizer.state_dict(),
        }, "{}/epoch_{}.pth.tar".format(args.model_save_dir, epoch))

        # record training and validation information
        if args.use_tfboard:
            record_info(train_info, dev_info, epoch, logger)

    if args.use_tfboard:
        logger.close()

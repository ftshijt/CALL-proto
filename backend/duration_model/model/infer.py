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
from model.loss import MaskedLoss
from model.utils import AverageMeter, create_src_key_padding_mask


def infer(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
                                local_gaussian=args.local_gaussian,
                                pos_enc=True)
    else:
        raise ValueError('Not Support Model Type %s' % args.model_type)
    print(model)
    model = model.to(device)

    test_set = DurationDataset(duration_file=args.test,
                             max_len=args.max_len)

    collate_fn = DurationCollator(args.max_len)
    test_loader = torch.utils.data.DataLoader(dataset=dev_set,
                                               batch_size=1,
                                               shuffle=True,
                                               num_workers=args.num_workers,
                                               collate_fn=collate_fn,
                                               pin_memory=True)

    # Load model weights
    print("Loading pretrained weights from {}".format(args.model_file))
    checkpoint = torch.load(args.model_file, map_location=device)
    state_dict = checkpoint['state_dict']
    model_dict = model.state_dict()
    state_dict_new = {}
    para_list = []
    for k, v in state_dict.items():
        assert k in model_dict
        if model_dict[k].size() == state_dict[k].size():
            state_dict_new[k] = v
        else:
            para_list.append(k)

    print("Total {} parameters, loaded {} parameters".format(len(state_dict), len(state_dict_new)))

    if len(para_list) > 0:
        print("Not loading {} because of different sizes".format(", ".join(para_list)))
    model_dict.update(state_dict_new)
    model.load_state_dict(model_dict)
    print("Loaded checkpoint {}".format(args.model_file))
    model = model.to(device)
    model.eval()

    if args.loss == "l1":
        loss = MaskedLoss("l1")
    elif args.loss == "mse":
        loss = MaskedLoss("mse")
    else:
        raise ValueError("Not Support Loss Type")

    if not os.path.exists(args.prediction_path):
        os.makedirs(args.prediction_path)

    with torch.no_grad():
        for step, (phone, mean_list, duration, length) in enumerate(test_loader, 1):
            phone = phone.to(device)
            mean_list = mean_list.to(device).float()
            duration = duration.to(device)
            length = length.to(device).long()
            length_mask = create_src_key_padding_mask(length, args.max_len)
            length_mask = length_mask.to(device)
            length = length.to(device)

            output = model(phone, mean_list, src_key_padding_mask=length)
            val_loss = criterion(output, duration, length_mask)
            losses.update(val_loss.item(), phone.size(0))
            if step % 1 == 0:
                output = output.cpu().detach().numpy()[0]
                duration = duration.cpu().detach().numpy()[0]
                np.save(os.path.join(args.prediction_path, "output_{}.npy".format(step)), output)
                np.save(os.path.join(args.prediction_path, "duration_{}.npy".foramt(step)), output)
    print("loss avg for test is {}".format(losses.avg))

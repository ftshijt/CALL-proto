#!/usr/bin/env python3

# Copyright 2020 The Johns Hopkins University (author: Jiatong Shi)

import yamlargparse

import sys
sys.path.append("/export/c04/jiatong/project/ai_system/CALL-proto/backend/duration_model")

parser = yamlargparse.ArgumentParser(description='Duration training')
parser.add_argument('-c', '--config', help='config file path',
                    action=yamlargparse.ActionConfigFile)
parser.add_argument('--train',
                    help='train data')
parser.add_argument('--val',
                    help="validation data")
parser.add_argument('--model-save-dir',
                    help='output directory which model file will be saved in.')
parser.add_argument('--model-type', default='Transformer',
                    help='Type of model (Transformer or LSTM)')
parser.add_argument('--initmodel', '-m', default='',
                    help='Initialize the model from given file')
parser.add_argument('--resume', default=False, type=bool,
                    help='Resume the optimization from snapshot')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--max-epochs', default=20, type=int,
                    help='Max. number of epochs to train')
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--optimizer', default='noam', type=str)
parser.add_argument('--gradclip', default=-1, type=int,
                    help='gradient clipping. if < 0, no clipping')
parser.add_argument('--max_len', default=100, type=int,
                    help='number of frames in one utterance')
parser.add_argument('--batchsize', default=1, type=int,
                    help='number of utterances in one batch')
parser.add_argument('--num_workers', default=4, type=int,
                    help='number of cpu workers')

parser.add_argument('--phone_size', default=86, type=int)
parser.add_argument('--dim_feedforward', default=1024, type=int)
parser.add_argument('--embedding-size', default=256, type=int)
parser.add_argument('--hidden-size', default=256, type=int)
parser.add_argument('--dropout', default=0.1, type=float)
parser.add_argument('--num_block', default=6, type=int)
parser.add_argument('--context', default=3, type=int)
parser.add_argument('--nhead', default=4, type=int)
parser.add_argument('--seed', default=666, type=int)
parser.add_argument('--use_tfb', dest='use_tfboard',
                    help='whether use tensorboard',
                    action='store_true')
parser.add_argument('--noam-scale', default=1.0, type=float)
parser.add_argument('--noam-warmup-steps', default=25000, type=float)
parser.add_argument('--loss', default="l1", type=str)
parser.add_argument('--use-pos-enc', default=0, type=int)
parser.add_argument('--gradient-accumulation-steps', default=1, type=int)
parser.add_argument('--local_gaussian', default=False, type=bool)

args = parser.parse_args()

print(args)
from model.train import train
train(args)

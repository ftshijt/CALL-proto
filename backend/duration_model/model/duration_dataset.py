#!/usr/bin/env python3

# Copyright 2020 The Johns Hopkins University (author: Jiatong Shi)

from torch.utils.data import Dataset
import numpy as np
import torch
import os
import librosa


class DurationCollator(object):
    def __init__(self, max_len, model_type=None, context=-1):
        self.max_len = max_len
        self.model_type = model_type
        self.context = context
        # plus 1 for aligner to consider padding char

    def __call__(self, batch):
        batch_size = len(batch)
        # get spectrum dim
        len_list = [len(batch[i][0]) for i in range(batch_size)]
        mean_list = []
        phone = np.zeros((batch_size, self.max_len))
        duration = np.zeros((batch_size, self.max_len))
        length_mask = np.zeros((batch_size, self.max_range))

        for i in range(batch_size):
            length = min(len_list[i], self.max_len)
            length_mask[i, :length] = np.arange(1, length + 1)
            phone[i, :length] = batch[i][0][:length]
            duration[i, :length] = batch[i][1][:length]
            mean_list.append(np.mean(batch[i][1][:length]))

        if self.model_type == "DNN":
            context_phone = np.zeros((batch_size, self.max_len, self.context * 2 + 1))
            for i in range(batch_size):
                for j in range(self.max_len):
                    cdphone = []
                    for k in range(self.context):
                        cdphone.append(phone[i, int(max(j-k-1, 0))])
                    cdphone.reverse()
                    cdphone.append(phone[i, j])
                    for k in range(self.context):
                        cdphone.append(phone[i, int(min(j+k+1, self.max_len - 1))])
                    cdphone = np.array(cdphone)
                    context_phone[i, j, :] = cdphone
            phone = context_phone


        mean_list = np.array(mean_list)

        phone = torch.from_numpy(phone).long()
        duration = torch.from_numpy(duration).float()
        length = torch.from_numpy(length_mask).long()
        mean_list = torch.from_numpy(mean_list).unsqueeze(dim=-1).unsqueeze(dim=-1).float()
        return phone, mean_list, duration, length


def refine_duration_data(info):
    input_info = []
    output_info = []
    for utt in info:
        utt = utt.strip()
        if len(utt) < 1:
            continue
        utt = utt.split(" ")
        head = utt[0]
        data = list(map(int, utt[1:]))
        utt_input = []
        utt_output = []
        
        # remove silence
        while data[0] == 1:
            data.pop(0)
            if len(data) < 1:
                break
        if len(data) < 1:
            continue
        
        while data[-1] == 1:
            data.pop(-1)
        pre = -1
        num_fram = 1
        for frame in data:
            if frame != pre:
                pre = frame
                utt_input.append(frame)
                utt_output.append(1)
            else:
                utt_output[-1] += 1
        input_info.append(utt_input)
        output_info.append(utt_output)
    return input_info, output_info


class DurationDataset(Dataset):
    def __init__(self, duration_file, max_len=100):
        duration_file = open(duration_file, "r")
        info = duration_file.read().split("\n")
        self.input_info, self.output_info = refine_duration_data(info)

    def __len__(self):
        return len(self.input_info)

    def __getitem__(self, i):
        return self.input_info[i], self.output_info[i]


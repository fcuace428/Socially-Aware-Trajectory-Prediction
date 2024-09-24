import numpy as np
import math
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class TITANdataset(Dataset):
    def __init__(self, input_path, inputn_path, target_path, numi):
        self.inputs = torch.from_numpy(np.array(np.load(input_path, allow_pickle=True), dtype='float32'))
        # class + bbox(0~4), cxcywh(5~8) distance(9, 10), speed(11~14), cxcywh_speed(15~18),imu(19, 20)  
        self.inputs = torch.cat((self.inputs[:, :, :5], self.inputs[:, :, 9:11], self.inputs[:, :, 11:15], self.inputs[:, :, 19:21]), -1)

        self.inputs_n = torch.from_numpy(np.array(np.load(inputn_path, allow_pickle=True), dtype='float32'))
        self.inputs_n = self.inputs_n[:, :, :, :5]
        self.targets = torch.from_numpy(np.array(np.load(target_path, allow_pickle=True), dtype='float32'))
        self.targets = self.targets[:, :,:, :13]
        self.num_i = torch.from_numpy(np.array(np.load(numi, allow_pickle=True), dtype='int')).bool()

    def __getitem__(self, idx):
        inputs = self.inputs[idx]
        targets = self.targets[idx]
        inputs_n1 = self.inputs_n[idx]
        num_i = self.num_i[idx]
        return inputs, inputs_n1, targets, num_i

    def __len__(self):
        return len(self.inputs)


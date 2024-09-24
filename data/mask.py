import numpy as np
import math
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.nn as nn

def for_step_wise_y(x, y, data_clip, save_name):
    obs_len = 10
    pre_len = 20
    x_traj = np.concatenate((x[:, :, :5], x[:, :, 9:11], x[:, :, 11:15], x[:, :, 19:22]), 2)
    y_traj = np.concatenate((y[:, :, :5], y[:, :, 9:11], y[:, :, 11:15], y[:, :, 19:22]), 2)
    tot_box = np.concatenate((x_traj, y_traj), 1)
    bs, sl, fd = tot_box.shape
    final_traj = np.zeros((bs, obs_len, pre_len, fd))
    for i in range(bs):
        for j in range(obs_len):
            traj = tot_box[i, j + 1:j + pre_len + 1].reshape(1, 1, 20, 14)
            final_traj[i, j] = traj
    np.save('./data/titan_data/%s/%s.npy'%(data_clip, save_name), final_traj)

def mask_g(input_n, num_i, data_clip, save_name):
    bs, ts, node, fd = input_n.shape 
    ts_mask = np.zeros((bs, ts, node), dtype=int)
    for i in range (bs):
        for j in range(ts):
            for z in range(node):
                num_count = int(num_i[i, j])
                a = np.ones((1, 1, node  - num_count), dtype=int)
                ts_mask[i, j, num_count:] = a
    np.save('./data/titan_data/%s/%s.npy'%(data_clip, save_name), ts_mask)
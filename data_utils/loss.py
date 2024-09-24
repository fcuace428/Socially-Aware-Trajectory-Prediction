import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
import numpy as np

class rmse_loss(nn.Module):
    '''
    Params:
        x_pred: (batch_size, enc_steps, dec_steps, pred_dim)
        x_true: (batch_size, enc_steps, dec_steps, pred_dim)
    Returns:
        rmse: scalar, rmse = \sum_{i=1:batch_size}()
    '''
    def __init__(self):
        super(rmse_loss, self).__init__()
    
    def forward(self, x_pred, x_true):
        L2_diff = torch.sqrt(torch.sum((x_pred - x_true)**2, dim=3))
        # sum over prediction time steps
        L2_all_pred = torch.sum(L2_diff, dim=2)
        # mean of each frames predictions
        L2_mean_pred = torch.mean(L2_all_pred, dim=1)
        # sum of all batches
        L2_mean_pred = torch.mean(L2_mean_pred, dim=0)
        return L2_mean_pred

def cvae_multi(pred_traj, target):
        '''
        CVAE loss use best-of-many
        '''
        K = pred_traj.shape[3]
        target = target.unsqueeze(3).repeat(1, 1, 1, K, 1)
        total_loss = []
        # best_idx_con = []
        for enc_step in range(pred_traj.size(1)):
            traj_rmse = torch.sqrt(torch.sum((pred_traj[:,enc_step,:,:,:] - target[:,enc_step,:,:,:])**2, dim=-1)).sum(dim=1)
            best_idx = torch.argmin(traj_rmse, dim=1)
            loss_traj = traj_rmse[range(len(best_idx)), best_idx].mean()
            total_loss.append(loss_traj)
        #     best_idx_con.append(best_idx.unsqueeze(-1))
        # best_idx = torch.cat(best_idx_con, -1)
        return sum(total_loss)/len(total_loss)

def final_multi(pred_traj, target):
        '''
        CVAE loss use best-of-many
        pred_traj: (batch_size, dec_steps, k, pred_dim)
        target: (batch_size, dec_steps, pred_dim)
        '''
        K = pred_traj.shape[2]
        target = target.unsqueeze(2).repeat(1, 1, K, 1)
        L2_diff = torch.sqrt(torch.sum((pred_traj - target)**2, dim=-1)).sum(dim=1)
        best_idx = torch.argmin(L2_diff, dim=1)
        final_loss = L2_diff[range(len(best_idx)), best_idx]
        L2_mean_pred = torch.mean(final_loss, dim=0)
        # sum of all batches
        return L2_mean_pred
    
def con_multi(x_pred, x_true):
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(x_pred, x_true)

    return loss

# if __name__ == '__main__':
#     # fl = cvae_loss(alpha, gamma, reduction)
#     # - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0. predict
#     # - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.   label
#     x = torch.randn([128, 20, 20, 4])
#     y = torch.randn([128, 20, 20, 4])
    
#     loss = final_multi(x, y)
#     print(loss)
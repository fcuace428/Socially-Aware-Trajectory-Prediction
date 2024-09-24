import numpy as np
import torch
from data_utils.TITANdataset import TITANdataset
from data_utils.multi_traj_cvae_v4 import *
import tqdm
import statistics

def predict(model, inputs, inputs_n, numi, device):
    model.eval()
    with torch.no_grad():
        input_traj = inputs.to(device)
        neigh = inputs_n.to(device)
        mask = numi.to(device)
        goal, cvae_dec_traj, KLD_loss = model(input_traj, neigh, mask,None, training=False)
        cvae_dec_traj = torch.clamp(cvae_dec_traj[:, -1], 0, 1)

    return cvae_dec_traj.cpu().numpy()

def bbox2center(pred, true, look_back = 20):
    true = denorm(true)
    pred = denorm(pred)
    t_tx = true[:, :, :, 0].unsqueeze(3)
    t_ty = true[:, :, :, 1].unsqueeze(3)
    t_bx = true[:, :, :, 2].unsqueeze(3)
    t_by = true[:, :, :, 3].unsqueeze(3)
    p_tx = pred[:, :, :, 0].unsqueeze(3)
    p_ty = pred[:, :, :, 1].unsqueeze(3)
    p_bx = pred[:, :, :, 2].unsqueeze(3)
    p_by = pred[:, :, :, 3].unsqueeze(3)
    
    true_w = abs(t_bx - t_tx)
    true_h = abs(t_by - t_ty)
    true_x = (t_tx + true_w * 0.5)
    true_y = (t_ty + true_h * 0.5)

    pred_w = abs(p_bx - p_tx)
    pred_h = abs(p_by - p_ty)
    pred_x = (p_tx + pred_w * 0.5)
    pred_y = (p_ty + pred_h * 0.5)

    pr7 = np.concatenate((pred_x, pred_y, pred_w, pred_h), 3)
    gt7 = np.concatenate((true_x, true_y, true_w, true_h), 3)
    return torch.from_numpy(pr7), torch.from_numpy(gt7)

def denorm(x):
    x[:, :, :, 0] = x[:, :, :, 0] * 1920
    x[:, :, :, 1] = x[:, :, :, 1] * 1200
    x[:, :, :, 2] = x[:, :, :, 2] * 1920
    x[:, :, :, 3] = x[:, :, :, 3] * 1200
    return x

def ADE(pred, true):
    d = (true[:, :, :, 0] - pred[:, :, :, 0])**2 + (true[:, :, :, 1] - pred[:, :, :, 1])**2
    # mean K=20 find min distance traj then mean bs and ts to get minADE
    ade = torch.sqrt(d).mean(axis=1).min(axis=-1).values.mean()
    return ade


def FDE(pred, true):
    d = (true[:, -1, :, 0] - pred[:, -1, :, 0])**2 + (true[:, -1, :, 1] - pred[:, -1, :, 1])**2
    # mean K=20 find min distance of traj last point then mean bs ans ts to get minADE
    fde = torch.sqrt(d).min(axis=-1).values.mean()
    return fde

def MSE(pred, true):
    displacement = (true[:, 0:2, :] - pred[:, 0:2, :])**2
    mse = torch.mean(displacement)
    rmse = torch.sqrt(mse)
    return mse, rmse

def FIOU(pred, true):
    # input cxcywh
    iouu = []
    for i in range(true.shape[0]):
        xmin = np.max([true[i, 0] - true[i, 2]/2, pred[i, 0] - pred[i, 2]/2]) 
        xmax = np.min([true[i, 0] + true[i, 2]/2, pred[i, 0] + pred[i, 2]/2])
        ymin = np.max([true[i, 1] - true[i, 3]/2, pred[i, 1] - pred[i, 3]/2])
        ymax = np.min([true[i, 1] + true[i, 3]/2, pred[i, 1] + pred[i, 3]/2])
        w_true = true[i, 2]
        h_true = true[i, 3]
        w_pred = pred[i, 2]
        h_pred = pred[i, 3]
        w_inter = np.max([0, xmax - xmin])
        h_inter = np.max([0, ymax - ymin])
        intersection = w_inter * h_inter
        union = (w_true * h_true + w_pred * h_pred) - intersection
        iou = (intersection/union).numpy()
        iouu.append(float(iou))
    return max(iouu)



import time
import os
import torch
from torch.nn.modules.loss import CrossEntropyLoss
import tqdm
from data_utils.loss import cvae_multi, rmse_loss, final_multi
import logging

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter("[%(asctime)s] %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger

def train_model(Gen, train_loader, val_loader, EPOCH, train_len, val_len, device):
    G_optimizer = torch.optim.Adam(Gen.parameters(), lr=1e-3 ,weight_decay=1e-4)
    G_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(G_optimizer, factor=0.1, patience=5, min_lr=1e-10, verbose=1)
    logger = get_logger('model_info.log')
    criterion = rmse_loss().to(device)
    if not os.path.isdir('./pre_train/'):
        os.mkdir('./pre_train/')
    best_val_loss = float("inf")
    with torch.set_grad_enabled(True):
        for epoch in range(EPOCH):
            #  trainning
            Gen.train()
            total_goal_loss = 0 
            total_cvae_loss = 0
            total_KLD_loss = 0
            total_final_loss = 0
            start_time = time.time()
            for _, (inputs, inputs_n, targets, numi) in enumerate(tqdm.tqdm(train_loader, desc=f"Training Epoch {epoch}")):
                # cla + bbox(0~4), cxcywh(5~8) distance(9, 10), speed(11~14), cxcywh_speed(15~18),imu(19, 20)放入TENSOR
                batch_size = inputs.shape[0]
                inputs = inputs.to(device, non_blocking=True)
                inputn = inputs_n.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                mask = numi.to(device, non_blocking=True)
                #  generator loss calculate
                all_goal_traj, cvae_dec_traj, KLD_loss = Gen(inputs, inputn, mask, targets)
                cvae_loss = cvae_multi(cvae_dec_traj, targets[:, :, :, 1:5])
                final_loss = final_multi(cvae_dec_traj[:, -1, :, :], targets[:, -1, :, 1:5])
                goal_loss = criterion(all_goal_traj, targets[:, :, :, 1:5])
                train_loss = cvae_loss + goal_loss + KLD_loss.mean()

                total_goal_loss += goal_loss.item() * batch_size
                total_cvae_loss += cvae_loss.item() * batch_size
                total_KLD_loss += KLD_loss.mean() * batch_size#
                total_final_loss += final_loss.item() * batch_size

                G_optimizer.zero_grad()  # clear gradients for this training step
                train_loss.backward()  # backpropagation, compute gradients
                G_optimizer.step()  # apply gradients
            total_goal_loss /= train_len
            total_cvae_loss /= train_len
            total_KLD_loss /= train_len
            total_final_loss /= train_len

            # val
            Gen.eval()
            total_goal_lossv = 0
            total_cvae_lossv = 0
            total_KLD_lossv = 0
            total_final_lossv = 0
            with torch.set_grad_enabled(False):
                for _, (inputsv, inputs_nv, targetsv, numiv) in enumerate(val_loader):
                    #   predict
                    # cla + bbox(0~4), cxcywh(5~8) distance(9, 10), speed(11~14), cxcywh_speed(15~18),imu(19, 20)  
                    batch_sizev = inputsv.shape[0]
                    inputsv = inputsv.to(device, non_blocking=True)
                    inputnv = inputs_nv.to(device, non_blocking=True)
                    targetsv = targetsv.to(device, non_blocking=True)
                    maskv = numiv.to(device, non_blocking=True)
                    # 一起
                    all_goal_trajv, cvae_dec_trajv, KLD_lossv = Gen(inputsv, inputnv, maskv, None, training=False)
                    cvae_lossv = cvae_multi(cvae_dec_trajv, targetsv[:, :, :, 1:5])
                    final_lossv = final_multi(cvae_dec_trajv[:, -1, :, :], targetsv[:, -1, :, 1:5])
                    goal_lossv = criterion(all_goal_trajv, targetsv[:, :, :, 1:5])

                    total_final_lossv += final_lossv.item() * batch_sizev
                    total_goal_lossv += goal_lossv.item() * batch_sizev
                    total_cvae_lossv += cvae_lossv.item() * batch_sizev
                    total_KLD_lossv += KLD_lossv.mean() * batch_sizev       

            total_goal_lossv /= val_len
            total_cvae_lossv /= val_len
            total_KLD_lossv /= val_len
            total_final_lossv/= val_len

            G_scheduler.step(total_final_lossv)

            print('time : {:5.2f}s | G_learning rate: {:5.9f}'\
                .format((time.time() - start_time), G_optimizer.param_groups[0]['lr']))
            print('| train goal loss: {:5.5f} | train cvae loss: {:5.5f} | train KLD loss: {:5.5f} | train final loss: {:5.5f}'\
                .format(total_goal_loss, total_cvae_loss, total_KLD_loss, total_final_loss))
            print('val goal loss: {:5.5f} | val cvae loss: {:5.5f} | val KLD loss: {:5.5f} | val final loss: {:5.5f}'\
                .format(total_goal_lossv, total_cvae_lossv, total_KLD_lossv, total_final_lossv))
            logger.info('Epoch:[{}]\t train_loss={:.5f}\t val_loss={:.5f}'.format(epoch,total_final_loss,total_final_lossv))
            if total_final_lossv < best_val_loss:
                best_val_loss = total_final_lossv
                MODEL_PATH = ('./pre_train/mtl_traj_%s.pt' % epoch)
                torch.save(Gen.state_dict(), MODEL_PATH)

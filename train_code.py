import torch
from data_utils.TITANdataset import TITANdataset
from data_utils.multi_traj_cvae_v4 import *
from data_utils.train_v4 import *

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # hyper parameters
    EPOCH = 10
    BATCH_SIZE_LOADER = 128
    train_dataset = TITANdataset('data/titan_data/train/x_train.npy','data/titan_data/train/x_train_n.npy',\
        'data/titan_data/train/y_train_ts.npy', 'data/titan_data/train/x_mask.npy')
    val_dataset = TITANdataset('data/titan_data/val/x_val.npy','data/titan_data/val/x_val_n.npy',\
        'data/titan_data/val/y_val_ts.npy', 'data/titan_data/val/x_mask.npy')

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE_LOADER, shuffle=True, num_workers=12, \
        pin_memory=False, prefetch_factor=10)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE_LOADER, shuffle=True, num_workers=12, \
        pin_memory=False, prefetch_factor=10)

    # ======    training    ======
    Gen = Social_Goal_Attention_Networks().to(device)
    #pretrained_weights = torch.load("")
    #Gen.load_state_dict(pretrained_weights, strict=False)
    train_model(Gen, train_loader, val_loader, EPOCH, len(train_dataset), len(val_dataset), device)
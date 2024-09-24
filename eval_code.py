import torch
from data_utils.TITANdataset import TITANdataset
from data_utils.multi_traj_cvae_v4 import *
from data_utils.train_v4 import *
from data_utils.eval_final import *

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_dataset = TITANdataset('data/titan_data/test/x_test.npy','data/titan_data/test/x_test_n.npy','data/titan_data/test/y_test_ts.npy', 'data/titan_data/test/x_mask.npy')
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)

    model = Social_Goal_Attention_Networks().to(device)
    model.load_state_dict(torch.load("./pre_train/mtl_traj_9.pt"))
    tra, cla = [], []
    trav, clav = [], []

    for step, (inputsv, inputs_nv, targetsv, numiv) in enumerate(tqdm.tqdm(test_loader)):
        real_boxv = predict(model, inputsv, inputs_nv, numiv, device)
        if step == 0:
            trav = real_boxv
            yt_test = targetsv[:, -1, :, 1:5].unsqueeze(2).repeat(1, 1, 20, 1)
        else:
            trav = np.concatenate((trav, real_boxv), 0)
            yt_test = np.concatenate((yt_test, targetsv[:, -1, :, 1:5].unsqueeze(2).repeat(1, 1, 20, 1)), 0)
    
    trav = torch.from_numpy(trav)
    trav, tra_test_gt = bbox2center(trav, torch.from_numpy(yt_test.astype(float)))
    av = ADE(trav, tra_test_gt)
    fv = FDE(trav, tra_test_gt)
    tmp_FIOUv = []
    for i in range(trav.shape[0]):
        tmp_FIOUv.append(FIOU(trav[i,-1,:], tra_test_gt[i,-1,:]))
    fiouv = statistics.mean(tmp_FIOUv)
    print('ADE:', float(av))
    print('FDE:', float(fv))
    print('FIOU:', float(fiouv))
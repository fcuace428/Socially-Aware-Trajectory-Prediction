#from cv2 import add
import torch
import torch.nn as nn
from data_utils.CVAE_v2 import CVAE

import math
class Social_Goal_Attention_Networks(nn.Module):
    def __init__(self):
        super(Social_Goal_Attention_Networks, self).__init__()
        self.cvae = CVAE()
        self.bbox_emb_size = 512
        self.hidden_size = 256  # GRU hidden size
        self.goal_emb_size = 128
        self.social = 1
        self.enc_steps = 10  # observation step
        self.dec_steps = 20  # prediction step
        self.dropout = 0
        self.pred_dim = 4
        self.K = 20
        self.drop = nn.Dropout(self.dropout)

        # input embeding(bbox speed imu)
        self.b_emb = nn.Sequential(nn.Linear(13, self.bbox_emb_size), nn.ELU(inplace=True))
        self.bbox_pos = PositionalEncoding(self.bbox_emb_size)

        # social gat
        self.social_emb = nn.Sequential(nn.Linear(5, self.social), nn.ELU(inplace=True))
        self.sg_mtl = nn.MultiheadAttention(self.social, 1, self.dropout, batch_first=True)
        self.norm0 = nn.LayerNorm(self.social)
        self.ffn = nn.Sequential(nn.Linear(self.social, 512), nn.ELU(inplace=True), nn.Dropout(self.dropout), 
                                    nn.Linear(512, self.social))
        self.norm1 = nn.LayerNorm(self.social)
        self.social_att = nn.Sequential(nn.Linear(self.social, 1), nn.ELU(inplace=True))

        # encoder
        self.enc_to_goal_hidden = nn.Sequential(nn.Linear(self.bbox_emb_size, self.goal_emb_size), nn.ELU(inplace=True))
        self.traj_enc_cell = nn.LSTMCell(self.bbox_emb_size + self.goal_emb_size+ self.social, self.bbox_emb_size)
        self.enc_to_dec_hidden = nn.Sequential(nn.Linear(self.bbox_emb_size, self.bbox_emb_size), nn.ELU(inplace=True))
        self.cvae_to_dec_hidden = nn.Sequential(nn.Linear(self.bbox_emb_size + 32, 512), nn.ELU(inplace=True))

        # goal
        self.goal_hidden_to_input = nn.Sequential(nn.Linear(self.goal_emb_size, self.goal_emb_size), nn.ELU(inplace=True))
        self.goal_cell = nn.LSTMCell(self.goal_emb_size, self.goal_emb_size)
        # self.goal_hidden_to_traj = nn.Sequential(nn.Linear(self.goal_emb_size, self.hidden_size), nn.ELU(inplace=True))
        self.goal_regressor = nn.Sequential(nn.Linear(self.goal_emb_size, self.pred_dim))
        self.goal_to_dec = nn.Sequential(nn.Linear(self.goal_emb_size, self.goal_emb_size), nn.ELU(inplace=True))
        self.goal_pos = PositionalEncoding(self.goal_emb_size)
        self.goal_mtl = nn.MultiheadAttention(self.goal_emb_size, 8, self.dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(self.goal_emb_size)
        self.goal_sum = nn.Sequential(nn.Linear(self.goal_emb_size, 1), nn.ELU(inplace=True))
        self.goal_sum1 = nn.Sequential(nn.Linear(self.goal_emb_size, 1), nn.ELU(inplace=True))
        self.linear = nn.Sequential(nn.Linear(self.goal_emb_size+20, self.goal_emb_size), nn.ELU(inplace=True))

        # decoder
        self.dec_for_rec = nn.Sequential(nn.Linear(512, 512), nn.ELU(inplace=True))
        self.dec_goal_mtl = nn.MultiheadAttention(512, 4, self.dropout, batch_first=True)
        self.dec_norm1 = nn.LayerNorm(512)
        self.dec_ffn = nn.Sequential(nn.Linear(512, 1024), nn.ELU(inplace=True), nn.Dropout(self.dropout), 
                                        nn.Linear(1024, 512))
        self.dec_norm2 = nn.LayerNorm(512)
        self.dec_regressor = nn.Sequential(nn.Linear(512, self.pred_dim))

    def box_embed(self, box):
        traj = self.b_emb(box)
        return traj

    def Social_Gat(self, neigh, mask):
        neigh = self.social_emb(neigh)
        att, w = self.sg_mtl(neigh, neigh, neigh, key_padding_mask=mask)
        addnorm = self.norm0(self.drop(att) + neigh)
        m = self.ffn(addnorm)
        addnorm1 = self.norm1(self.drop(m) + addnorm)
        sat = self.social_att(addnorm1)
        sat = torch.softmax(sat, dim=1)
        social = torch.bmm(addnorm1.transpose(1, 2), sat).squeeze(-1)
        return social, w

    def Social_Goal(self, goal_hidden):
        goal_input = goal_hidden.new_zeros(goal_hidden.size(0), self.goal_emb_size)
        goal_list = []
        device = goal_hidden.device
        for dec_step in range(self.dec_steps):
            goal_input = self.goal_hidden_to_input(goal_hidden)
            noise = torch.randn(20).repeat(goal_input.size(0),1).to(device)
            goal_input = torch.cat((goal_input, noise), 1)
            goal_input = self.linear(goal_input)               
            goal_hidden, _ = self.goal_cell(self.drop(goal_input), (goal_hidden,goal_hidden))
            goal_list.append(goal_hidden)

        goal_stack = self.goal_pos(torch.stack([self.goal_to_dec(goal) for goal in goal_list], dim=1))
        goal_mtl, _ = self.goal_mtl(goal_stack, goal_stack, goal_stack)
        goal_norm= self.norm2(self.drop(goal_mtl) + goal_stack)

        goal_traj = self.goal_regressor(goal_norm)

        goal_sum = self.goal_sum(goal_norm)
        goal_att = torch.softmax(goal_sum, dim=1)
        goal_for_enc = torch.bmm(goal_norm.transpose(1, 2), goal_att).squeeze(-1)

        goal_sum1 = self.goal_sum1(goal_norm)
        goal_att1 = torch.softmax(goal_sum1, dim=1)
        goal_for_dec = torch.bmm(goal_norm.transpose(1, 2), goal_att1).squeeze(-1)
        return goal_for_dec, goal_for_enc, goal_traj

    def cvae_decoder(self, dec_hidden):
        batch_size = dec_hidden.size(0)
        K = dec_hidden.shape[1]
        dec_traj = dec_hidden.new_zeros(batch_size, self.dec_steps, K, self.pred_dim)
        dec_goal_cvae = self.bbox_pos(dec_hidden)

        for k_step in range(self.dec_steps):
            dec_goal, _ = self.dec_goal_mtl(dec_goal_cvae, dec_goal_cvae, dec_goal_cvae)
            dec_norm1 = self.dec_norm1(self.drop(dec_goal) + dec_goal_cvae)
            goal_d = self.dec_ffn(dec_norm1)
            dec_norm2 = self.dec_norm2(self.drop(goal_d) + dec_norm1)
            dec_traj[:, k_step, :, :] = self.dec_regressor(dec_norm2)
            # recusive
            dec_goal_cvae = self.bbox_pos(self.dec_for_rec(dec_norm2))
        return dec_traj
    
    def encoder(self, raw_inputs, neigh, mask, raw_targets, traj_input):
        # initial output tensor
        # first enc step dec later
        all_goal_traj = traj_input.new_zeros(traj_input.size(0), self.enc_steps, self.dec_steps, self.pred_dim)
        all_cvae_dec_traj = traj_input.new_zeros(traj_input.size(0), self.enc_steps, self.dec_steps, self.K, self.pred_dim)
        # initial encoder goal with zeros
        goal_for_enc = traj_input.new_zeros((traj_input.size(0), self.goal_emb_size))
        # initial encoder hidden with zeros
        traj_enc_hidden = traj_input.new_zeros((traj_input.size(0), 512))
        total_KLD = 0
        for enc_step in range(self.enc_steps):
            social, w = self.Social_Gat(neigh[:, enc_step], mask[:, enc_step])
            traj_enc_hidden,_ = self.traj_enc_cell(self.drop(torch.cat((traj_input[:, enc_step, :], goal_for_enc,social), 1)), (traj_enc_hidden,traj_enc_hidden))#, social
            enc_hidden = traj_enc_hidden
            goal_hidden = self.enc_to_goal_hidden(enc_hidden)
            goal_for_dec, goal_for_enc, goal_traj = self.Social_Goal(goal_hidden)
            all_goal_traj[:, enc_step, :, :] = goal_traj
            dec_hidden = self.enc_to_dec_hidden(enc_hidden)
            if self.training:
                cvae_hidden, KLD = self.cvae(dec_hidden, goal_for_dec, raw_inputs[:, enc_step, :], raw_targets[:, enc_step, :, :])
            else:
                cvae_hidden, KLD = self.cvae(dec_hidden, goal_for_dec, raw_inputs[:, enc_step, :], None)
            total_KLD += KLD
            cvae_dec_hidden = self.cvae_to_dec_hidden(cvae_hidden)
            all_cvae_dec_traj[:, enc_step, :, :, :] = self.cvae_decoder(cvae_dec_hidden)
        return all_goal_traj, all_cvae_dec_traj, total_KLD

    def forward(self, inputs, neigh, mask, targets=None, training=True):
        self.training = training
        traj_input = self.box_embed(inputs) # x_train(batch_size, time_sequence[10], features[13])
        all_goal_traj, all_cvae_dec_traj, KLD= self.encoder(inputs, neigh, mask, targets, traj_input)
        return all_goal_traj, all_cvae_dec_traj, KLD


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x.transpose(0, 1)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x.transpose(0, 1))

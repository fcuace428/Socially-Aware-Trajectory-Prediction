import torch
import torch.nn as nn
from torch.distributions import Normal
class CVAE(nn.Module):
    def __init__(self, hidden_size = 256, emb_size = 128, nu = 0.0, sigma = 1.5, k = 20):
        super(CVAE, self).__init__()
        self.input_dim = 4
        self.pred_dim = 4
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.latent = 32
        self.nu = nu
        self.sigma = sigma
        self.k = k
        self.node_future_encoder_h = nn.Sequential(nn.Linear(13, hidden_size),nn.ELU(inplace=True))
        self.gt_goal_encoder = nn.LSTM(input_size=13,
                                        hidden_size= hidden_size,
                                        bidirectional = True,
                                        batch_first=True)
        self.p_z_x = nn.Sequential(nn.Linear(512 + self.emb_size, self.emb_size),
                                    nn.ELU(inplace=True),
                                    nn.Linear(self.emb_size, 64),
                                    nn.ELU(inplace=True),
                                    nn.Linear(64, 64))
        # posterior
        self.q_z_xy = nn.Sequential(nn.Linear(512 + 512, self.emb_size),
                                    nn.ELU(inplace=True),
                                    nn.Linear(self.emb_size, 64),
                                    nn.ELU(inplace=True),
                                    nn.Linear(64, 64))

    #(bs, 512), (bs, 11), (bs, 20, 4)
    def gaussian_latent_net(self, dec_ht, goal, raw_input, target = None):
        # get mu, sigma
        # 1. sample z from piror(obs 先驗)
        z_mu_logvar_p = self.p_z_x(torch.cat((dec_ht, goal), -1))
        z_mu_p = z_mu_logvar_p[:, :self.latent]
        z_logvar_p = z_mu_logvar_p[:, self.latent:]
        if target is not None:
            # 2. sample z from posterior, for training only
            initial_h = self.node_future_encoder_h(raw_input)
            initial_h = torch.stack([initial_h, torch.zeros_like(initial_h, device=initial_h.device)], dim=0)
            c_0 = torch.zeros_like(initial_h, device=initial_h.device)
            self.gt_goal_encoder.flatten_parameters()
            # target_h(2, bs, 128)
            _, (target_h, _) = self.gt_goal_encoder(target, (initial_h, c_0))
            # target_h(bs, 2, 128)
            target_h = target_h.permute(1,0,2)
            # target_h(bs, 128*2)
            target_h = target_h.reshape(-1, target_h.shape[1] * target_h.shape[2])
            z_mu_logvar_q = self.q_z_xy(torch.cat((dec_ht, target_h), dim=-1))
            z_mu_q = z_mu_logvar_q[:, :self.latent]
            z_logvar_q = z_mu_logvar_q[:, self.latent:]
            Z_mu = z_mu_q
            Z_logvar = z_logvar_q
            # 3. compute KL(q_z_xy||p_z_x)
            KLD = 0.5 * ((z_logvar_q.exp()/z_logvar_p.exp()) + \
                        (z_mu_p - z_mu_q).pow(2)/z_logvar_p.exp() - \
                        1 + (z_logvar_p - z_logvar_q))
            KLD = KLD.sum(dim=-1).mean()
            KLD = torch.clamp(KLD, min=0.001)
            
        else:
            Z_mu = z_mu_p
            Z_logvar = z_logvar_p
            KLD = torch.as_tensor(0.0, device=Z_logvar.device)
        
        # 4. Draw sample
        with torch.set_grad_enabled(False):
            K_samples = torch.normal(self.nu, self.sigma, size = (dec_ht.shape[0], self.k, self.latent)).cuda()

        Z_std = torch.exp(0.5 * Z_logvar)
        Z = Z_mu.unsqueeze(1).repeat(1, self.k, 1) + K_samples * Z_std.unsqueeze(1).repeat(1, self.k, 1)
        return Z, KLD


    def forward(self, dec_ht, goal, last_raw_input, raw_tar=None):
        Z, KLD = self.gaussian_latent_net(dec_ht, goal, last_raw_input, raw_tar)
        # bs, k, 512+128
        enc_obs_and_z = torch.cat((dec_ht.unsqueeze(1).repeat(1, Z.shape[1], 1), Z), -1)
        # enc_obs_and_z = torch.cat((dec_ht, goal, Z), -1)
        return enc_obs_and_z, KLD
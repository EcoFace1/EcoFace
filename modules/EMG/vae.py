import math
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributions as dist
import numpy as np
import sys

from modules.EMG.flow_base import WN, ResidualCouplingBlock

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class FVAEEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_channels, kernel_size,
                 n_layers, gin_channels=0, p_dropout=0, strides=[4]):
        super().__init__()
        self.strides = strides
        self.hidden_size = hidden_channels
        self.pre_net = nn.Sequential(*[
            nn.Conv1d(in_channels, hidden_channels, kernel_size=s * 2, stride=s, padding=s // 2)
            if i == 0 else
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=s * 2, stride=s, padding=s // 2)
            for i, s in enumerate(strides)
        ])
        self.wn = WN(hidden_channels, kernel_size, 1, n_layers, gin_channels, p_dropout)
        self.out_proj = nn.Conv1d(hidden_channels, latent_channels * 2, 1)

        self.latent_channels = latent_channels

    def forward(self, x, x_mask, g1, g2): 
        x = self.pre_net(x)
        x_mask = x_mask[:, :, ::np.prod(self.strides)][:, :, :x.shape[-1]]
        x = x * x_mask
        x = self.wn(x, x_mask=x_mask, g1=g1, g2=g2) * x_mask
        x = self.out_proj(x)
        m, logs = torch.split(x, self.latent_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs))
        return z, m, logs, x_mask


class FVAEDecoder(nn.Module):
    def __init__(self, latent_channels, hidden_channels, out_channels, kernel_size,
                 n_layers, gin_channels=0, p_dropout=0,
                 strides=[4]):
        super().__init__()
        self.strides = strides
        self.hidden_size = hidden_channels
        self.pre_net = nn.Sequential(*[
            nn.ConvTranspose1d(latent_channels, hidden_channels, kernel_size=s, stride=s)
            if i == 0 else
            nn.ConvTranspose1d(hidden_channels, hidden_channels, kernel_size=s, stride=s)
            for i, s in enumerate(strides)
        ])
        self.wn = WN(hidden_channels, kernel_size, 1, n_layers, gin_channels, p_dropout)
        self.out_proj = nn.Conv1d(hidden_channels, out_channels, 1)

    def forward(self, x, x_mask, g1, g2, g3):
        x = self.pre_net(x)
        x = x * x_mask
        if g1.shape[2]>x.shape[2]:
            g1 = g1[:,:,:x.shape[2]]
            g2 = g2[:,:,:x.shape[2]]
            g3 = g3[:,:,:x.shape[2]]
        x = self.wn(x, x_mask=x_mask, g1=g1, g2=g2, g3=g3) * x_mask
        x = self.out_proj(x)
        return x

class FVAE(nn.Module):
    def __init__(self,
                 in_out_channels=64, hidden_channels=256, latent_size=16,
                 kernel_size=3, enc_n_layers=5, dec_n_layers=5, gin_channels=80, strides=[4,],
                 use_prior_glow=True, glow_hidden=256, glow_kernel_size=3, glow_n_blocks=5):
        super(FVAE, self).__init__()
        self.in_out_channels = in_out_channels
        self.strides = strides
        self.hidden_size = hidden_channels
        self.latent_size = latent_size
        self.use_prior_glow = use_prior_glow
        
        self.g_pre_net1 = nn.Sequential(*[
            nn.Conv1d(gin_channels, gin_channels, kernel_size=s * 2, stride=s, padding=s // 2)
            for i, s in enumerate(strides)
        ]) #for hubert
        self.g_pre_net2 = nn.Sequential(*[
            nn.Conv1d(gin_channels, gin_channels, kernel_size=s * 2, stride=s, padding=s // 2)
            for i, s in enumerate(strides)
        ]) #for emotion

        self.encoder = FVAEEncoder(in_out_channels, hidden_channels, latent_size, kernel_size,
                                   enc_n_layers, gin_channels, strides=strides)
        self.prior_flow = ResidualCouplingBlock(
            latent_size, glow_hidden, glow_kernel_size, 1, glow_n_blocks, 4, gin_channels=gin_channels)
    
        self.decoder = FVAEDecoder(latent_size, hidden_channels, in_out_channels, kernel_size,
                                    dec_n_layers, gin_channels, strides=strides)

        self.prior_dist = dist.Normal(0, 1)

    def forward(self, x=None, x_mask=None, g1=None, g2=None, g3=None, infer=False, temperature=1. , **kwargs):
        """

        :param x: [B, T,  C_in_out]
        :param x_mask: [B, T]
        :param g: [B, T, C_g]
        :return:
        """
        x_mask = x_mask[:, None, :] # [B, 1, T]
        g1 = g1.transpose(1,2) # [B, C_g, T]
        g2 = g2.transpose(1,2) # [B, C_g, T]
        g3 = g3.transpose(1,2) # [B, C_g, T]
        
        g_for_sqz1 = g1
        g_for_sqz2 = g2
        #g_for_sqz3 = g3
        
        g_sqz1 = self.g_pre_net1(g_for_sqz1)
        g_sqz2 = self.g_pre_net2(g_for_sqz2)
        #g_sqz3 = self.g_pre_net3(g_for_sqz3)

        if not infer:
            x = x.transpose(1,2) # [B, C, T]
            z_q, m_q, logs_q, x_mask_sqz = self.encoder(x, x_mask, g_sqz1, g_sqz2)
            
            x_recon = self.decoder(z_q, x_mask, g1, g2, g3)
            q_dist = dist.Normal(m_q, logs_q.exp())
         
            logqx = q_dist.log_prob(z_q)
            z_p = self.prior_flow(z_q, x_mask_sqz, g_sqz1, g_sqz2)
            logpx = self.prior_dist.log_prob(z_p)
            loss_kl = ((logqx - logpx) * x_mask_sqz).sum() / x_mask_sqz.sum() / logqx.shape[1]
            
            return x_recon.transpose(1,2), loss_kl, z_p.transpose(1,2), m_q.transpose(1,2), logs_q.transpose(1,2)
        else:
            latent_shape = [g_sqz1.shape[0], self.latent_size, g_sqz1.shape[2]]
            z_p = self.prior_dist.sample(latent_shape).to(g1.device) * temperature # [B, latent_size, T_sqz]

            z_p = self.prior_flow(z_p, 1, g_sqz1, g_sqz2, reverse=True)
            
            x_recon = self.decoder(z_p, 1, g1, g2, g3)
            
            return x_recon.transpose(1,2), z_p.transpose(1,2)
    
    def predict(self, g1=None, g2=None, g3=None, temperature=1. , **kwargs):
        g1 = g1.transpose(1,2) # [B, C_g, T]
        g2 = g2.transpose(1,2) # [B, C_g, T]
        g3 = g3.transpose(1,2) # [B, C_g, T]
        
        g_for_sqz1 = g1
        g_for_sqz2 = g2
        #g_for_sqz3 = g3
        
        g_sqz1 = self.g_pre_net1(g_for_sqz1)
        g_sqz2 = self.g_pre_net2(g_for_sqz2)
        #g_sqz3 = self.g_pre_net3(g_for_sqz3)
        latent_shape = [g_sqz1.shape[0], self.latent_size, g_sqz1.shape[2]]
        z_p = self.prior_dist.sample(latent_shape).to(g1.device) * temperature # [B, latent_size, T_sqz]

        z_p = self.prior_flow(z_p, 1, g_sqz1, g_sqz2, reverse=True)
        
        x_recon = self.decoder(z_p, 1, g1, g2, g3)
        return x_recon.transpose(1,2), z_p.transpose(1,2)


class VAEModel(nn.Module):
    def __init__(self, in_out_dim=64, cond_drop=False, use_prior_flow=True):
        super().__init__()
        mel_feat_dim = 64
        mel_in_dim = 1024 # hubert
        
        cond_dim = mel_feat_dim

        person_dic = np.load('./checkpoint/person_dic.npz')
        person_dict = {key: value for key, value in person_dic.items()}
        #person id
        self.one_hot_person = torch.eye(len(person_dic.keys())).cuda()
        self.obj_vector_person = nn.Linear(len(person_dic.keys()), cond_dim)
        
        self.mel_encoder = nn.Sequential(*[
                nn.Conv1d(mel_in_dim, 64, 3, 1, 1, bias=False),
                nn.BatchNorm1d(64),
                nn.GELU(),
                nn.Conv1d(64, mel_feat_dim, 3, 1, 1, bias=False)
            ]) #for hubert
        self.emo_encoder = nn.Sequential(*[
                nn.Conv1d(mel_in_dim, 64, 3, 1, 1, bias=False),
                nn.BatchNorm1d(64),
                nn.GELU(),
                nn.Conv1d(64, mel_feat_dim, 3, 1, 1, bias=False)
            ]) #for emo
        self.cond_drop = cond_drop
        if self.cond_drop:
            self.dropout = nn.Dropout(0.5)

        self.in_dim, self.out_dim = in_out_dim, in_out_dim
        self.use_prior_flow = use_prior_flow
        self.vae = FVAE(in_out_channels=in_out_dim, hidden_channels=256, latent_size=16, kernel_size=5,
            enc_n_layers=8, dec_n_layers=4, gin_channels=cond_dim, strides=[4,],
            use_prior_glow=self.use_prior_flow, glow_hidden=64, glow_kernel_size=3, glow_n_blocks=4)
        self.downsampler = LambdaLayer(lambda x: F.interpolate(x.transpose(1,2), scale_factor=0.5, mode='nearest').transpose(1,2))

    def num_params(self, model, print_out=True, model_name="model"):
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        if print_out:
            print(f'| {model_name} Trainable Parameters: %.3fM' % parameters)
        return parameters
    
    @property
    def device(self):
        return self.vae.parameters().__next__().device

    def forward(self, batch, ret, train=True, return_latent=False, temperature=1.):
        infer = not train
        mask = batch['y_mask'].to(self.device)
        mel = batch['hubert'].to(self.device)
        emo = batch['emovec'].to(self.device)

        mel = self.downsampler(mel)
        emo = self.downsampler(emo)
        cond_feat1 = self.mel_encoder(mel.transpose(1,2)).transpose(1,2)
        cond_feat2 = self.emo_encoder(emo.transpose(1,2)).transpose(1,2)
        
        person = batch['person'].to(self.device) #batch,person id
        
        onehot_person = self.one_hot_person[person]
        obj_embedding_person = self.obj_vector_person(onehot_person)
        cond_feat3=obj_embedding_person[:,None,:].repeat(1, cond_feat1.shape[1], 1)
        
        if self.cond_drop:
            cond_feat1 = self.dropout(cond_feat1)
            cond_feat2 = self.dropout(cond_feat2)
        
        if not infer:
            exp = batch['y'].to(self.device)
            x = exp
            x_recon, loss_kl, z_p, m_q, logs_q = self.vae(x=x, x_mask=mask, g1=cond_feat1, g2=cond_feat2, g3=cond_feat3, infer=False)
            x_recon = x_recon * mask.unsqueeze(-1)
            ret['pred'] = x_recon
            ret['mask'] = mask
            ret['loss_kl'] = loss_kl
            if return_latent:
                ret['m_q'] = m_q
                ret['z_p'] = z_p
            return x_recon, loss_kl, m_q, logs_q
        else:
            x_recon, z_p = self.vae(x=None, x_mask=mask, g1=cond_feat1, g2=cond_feat2, g3=cond_feat3, infer=True, temperature=temperature)
            x_recon = x_recon * mask.unsqueeze(-1)
            ret['pred'] = x_recon
            ret['mask'] = mask

            return x_recon
    
    def predict(self, g1, g2, g3, temperature=1.):

        mel = g1
        emo = g2

        mel = self.downsampler(mel)
        emo = self.downsampler(emo)
        cond_feat1 = self.mel_encoder(mel.transpose(1,2)).transpose(1,2)
        cond_feat2 = self.emo_encoder(emo.transpose(1,2)).transpose(1,2)
        
        person = g3
        
        onehot_person = self.one_hot_person[person]
        obj_embedding_person = self.obj_vector_person(onehot_person)
        cond_feat3=obj_embedding_person[:,None,:].repeat(1, cond_feat1.shape[1], 1)
        
        if self.cond_drop:
            cond_feat1 = self.dropout(cond_feat1)
            cond_feat2 = self.dropout(cond_feat2)
        
        
        x_recon, z_p = self.vae.predict(g1=cond_feat1, g2=cond_feat2, g3=cond_feat3, temperature=temperature)

        return x_recon


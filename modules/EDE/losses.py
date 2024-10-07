
import torch
import torch.nn as nn
import sys

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0]
       
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)+ 1e-6)
        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(1, batch_size).mean()

        return loss

class TripletLoss(nn.Module):
    def __init__(self):
        super(TripletLoss, self).__init__()
        self.triplet_loss = nn.TripletMarginLoss(margin= 0.5, reduction='mean')
    
    def forward(self, features, labels=None, level_all=None):
        total_loss = 0
        for i in range(len(features)):
            l = level_all[i]
            chose = (labels == labels[i])
            chose[i] = False
            fea = features[chose]
            level = level_all[chose]
            if(level.numel() == 0):
                total_loss += 0
                continue
            fea_same = fea[level == l] #(b_s,1024)
            fea_contra = fea[level != l] #(b_c,1024)
            if(fea_same.numel()==0 or fea_contra.numel()==0):
                total_loss += 0
                continue
            a = features[i].repeat(fea_contra.shape[0] * fea_same.shape[0],1) #(b_c * b_s, 1024) 
            #b = fea_same.repeat(fea_contra.shape[0],1)  #(b_c * b_s, 1024) 
            b = torch.repeat_interleave(fea_same,fea_contra.shape[0],dim=0)
            c = fea_contra.repeat(fea_same.shape[0], 1) #(b_c * b_s, 1024)
            loss = self.triplet_loss(a, b, c)
            total_loss += loss
        
        return total_loss/len(features)



            

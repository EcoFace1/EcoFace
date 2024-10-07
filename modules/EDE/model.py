import os
import sys
import argparse

from collections import OrderedDict
import torch
import torch.nn as nn

#from modules.VAembedding.Videobackbone.c3d import C3D
#from modules.VAembedding.Audiobackbone.vggish import VGGish 

from modules.EDE.Videobackbone.resnet import VideoEncoder #video feature network
from transformers import HubertModel #audio feature network

from modules.EDE.losses import SupConLoss,TripletLoss #loss function

class EmoEmb(nn.Module):
    def __init__(self):
        super().__init__()
        
        # define networks
        #self.net_vid = C3D()
        #self.net_aud = VGGish()
        
        self.net_vid = VideoEncoder()
        self.net_aud = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")

        #load the pretrained model from AV-Hubert
        video_ckp = torch.load('./checkpoint/base_lrs3_iter5.pt')['model']
        video_new_state_dict = OrderedDict()
        for k, v in video_ckp.items():
            if k[:23] == 'feature_extractor_video':
                if 'proj' in k:
                    continue
                video_new_state_dict[k[24:]] = v
        self.net_vid.load_state_dict(video_new_state_dict,strict=False)

        #only train hubert transformer encoder
        self.net_aud.feature_extractor._freeze_parameters()
        for param in self.net_aud.feature_projection.parameters():
            param.requires_grad = False
        
        #define loss function
        self.criterion1 = SupConLoss()
        self.criterion2 = TripletLoss()

    def forward(self,batch, ret, train=True, infer_type='audio'):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if train:
            #train together
            pic=batch['pic'] #(batch, C=1, time, H=128, W=128)
            video_features=self.net_vid(pic) #[batch_size, time, 1024]
            video_features_mean = torch.mean(video_features,dim=1)
            video_loss1 = self.criterion1(video_features_mean,batch['label'])
            video_loss2 = self.criterion2(video_features_mean,batch['label'],batch['level'])
            
            audio=batch['mel']
            audio_features=self.net_aud.forward(audio).last_hidden_state #[batch_size, time*2, 1024]
            audio_features_mean = torch.mean(audio_features,dim=1)
            audio_loss1 = self.criterion1(audio_features_mean,batch['label'])
            audio_loss2 = self.criterion2(audio_features_mean,batch['label'],batch['level'])

            video_features_double = self.linear_interpolate_batch(video_features)

            if audio_features.shape[1]<video_features_double.shape[1]:
                video_features_double = video_features_double[:,:audio_features.shape[1],:]
            else:
                audio_features = audio_features[:,:video_features_double.shape[1],:]
            inter_loss = self.mse_loss(video_features_double,audio_features)

            ret['video_features']=video_features
            ret['audio_features']=audio_features
            ret['video_loss1']=video_loss1
            ret['video_loss2']=video_loss2
            ret['audio_loss1']=audio_loss1
            ret['audio_loss2']=audio_loss2
            '''ret['inter_loss1']=inter_loss1
            ret['inter_loss2']=inter_loss2'''
            ret['inter_loss']=inter_loss
            return inter_loss
        else:
            if infer_type=='video':
                pic=batch['pic'] #(batch, C=1, time, H=128, W=128)
                features=self.net_vid(pic) #[batch_size, time, 1024]
                features = self.linear_interpolate_batch(features)
                ret['features']=features
            else:
                audio=batch['mel'] #(batch, C=1, H=time, W=64)
                features=self.net_aud.forward(audio).last_hidden_state #[batch_size, time*2, 1024]
                ret['features']=features
            return features
    
    def linear_interpolate_batch(self,input_tensor):
        batch_size, seq_length, feature_dim = input_tensor.size()
        output_tensor = torch.zeros((batch_size, 2 * seq_length, feature_dim), dtype=input_tensor.dtype,
                                    device=input_tensor.device)
        # (batch,time,feature_dim) -> (batch,2*time,feature_dim)
        tempA = input_tensor[:, :-1, :]
        tempB = input_tensor[:, 1:, :]
        tempC = (tempA + tempB) / 2
        output_tensor[:, ::2, :] = input_tensor
        output_tensor[:, 1:-1:2, :] = tempC
        output_tensor[:, -1, :] = output_tensor[:, -2, :]

        return output_tensor

    def mse_loss(self, x_gt, x_pred):
            # mean squared error, l2 loss
            error = (x_pred - x_gt)
            num_frame = x_gt.shape[0]*x_gt.shape[1] #batch*time
            n_dim = x_pred.shape[-1]
            return (error ** 2).sum() / (num_frame * n_dim)
    
    def mae_loss(self, x_gt, x_pred):
        # mean absolute error, l1 loss
        error = (x_pred - x_gt) 
        num_frame = x_gt.shape[0]*x_gt.shape[1]
        n_dim = x_gt.shape[-1]
        return error.abs().sum() / (num_frame * n_dim)
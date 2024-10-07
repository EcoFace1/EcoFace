import scipy
from scipy import linalg
from torch.nn import functional as F
import torch
from torch import nn
import numpy as np

def fused_add_tanh_sigmoid_multiply(input_a, input_b, input_c, style, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b + input_c + style
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts

class WN(torch.nn.Module):
    def __init__(self, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0,
                 p_dropout=0, share_cond_layers=False):
        super(WN, self).__init__()
        assert (kernel_size % 2 == 1)
        assert (hidden_channels % 2 == 0)
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout
        self.share_cond_layers = share_cond_layers

        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        
        self.drop = nn.Dropout(p_dropout)

        #if multi-condition, use more cond_layer ***************************
        if gin_channels != 0 and not share_cond_layers:
            cond_layer1 = torch.nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1)
            cond_layer2 = torch.nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1)
            cond_layer3 = torch.nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1)
            self.cond_layer1 = torch.nn.utils.weight_norm(cond_layer1, name='weight') #for hubert
            self.cond_layer2 = torch.nn.utils.weight_norm(cond_layer2, name='weight') #for emotion
            self.cond_layer3 = torch.nn.utils.weight_norm(cond_layer3, name='weight') #for person

        for i in range(n_layers):
            dilation = dilation_rate ** i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = torch.nn.Conv1d(hidden_channels, 2 * hidden_channels, kernel_size,
                                       dilation=dilation, padding=padding)
            in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

            res_skip_layer = torch.nn.Conv1d(hidden_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x, x_mask=None, g1=None, g2=None, g3=None, **kwargs):
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_channels])
        if g1 is not None and not self.share_cond_layers:
            g1 = self.cond_layer1(g1)
        if g2 is not None and not self.share_cond_layers:
            g2 = self.cond_layer2(g2)
        if g3 is not None and not self.share_cond_layers:
            g3 = self.cond_layer3(g3)
        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            x_in = self.drop(x_in)
            if g1 is not None and g2 is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l1 = g1[:, cond_offset:cond_offset + 2 * self.hidden_channels, :]
                g_l2 = g2[:, cond_offset:cond_offset + 2 * self.hidden_channels, :]
            else:
                g_l1 = torch.zeros_like(x_in)
                g_l2 = torch.zeros_like(x_in)
            if g3 is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l3 = g3[:, cond_offset:cond_offset + 2 * self.hidden_channels, :]
            else:
                g_l3 = torch.zeros_like(x_in)
            
            acts = fused_add_tanh_sigmoid_multiply(x_in, g_l1, g_l2, g_l3, n_channels_tensor)
            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                x = (x + res_skip_acts[:, :self.hidden_channels, :]) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels:, :]
            else:
                output = output + res_skip_acts
        return output * x_mask

    def remove_weight_norm(self):
        def remove_weight_norm(m):
            try:
                nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(remove_weight_norm)
    
    def enable_adapters(self):
        if not self.use_adapters:
            return
        for adapter_layer in self.adapter_layers:
            adapter_layer.enable()

    def disable_adapters(self):
        if not self.use_adapters:
            return
        for adapter_layer in self.adapter_layers:
            adapter_layer.disable()

class Flip(nn.Module):
    def forward(self, x, *args, reverse=False, **kwargs):
        x = torch.flip(x, [1])
        logdet = torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
        return x, logdet

    def store_inverse(self):
        pass

class ResidualCouplingLayer(nn.Module):
    def __init__(self,
                 channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 p_dropout=0,
                 gin_channels=0,
                 mean_only=False,
                 nn_type='wn'):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        if nn_type == 'wn':
            self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=p_dropout,
                          gin_channels=gin_channels)
        # elif nn_type == 'conv':
        #     self.enc = ConditionalConvBlocks(
        #         hidden_channels, gin_channels, hidden_channels, [1] * n_layers, kernel_size,
        #         layers_in_block=1, is_BTC=False)
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g1=None, g2=None,  reverse=False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask=x_mask, g1=g1, g2=g2)
        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)
        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = -torch.sum(logs, [1, 2])
            return x, logdet


class ResidualCouplingBlock(nn.Module):
    def __init__(self,
                 channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 n_flows=4,
                 gin_channels=0,
                 nn_type='wn'):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers,
                                      gin_channels=gin_channels, mean_only=True, nn_type=nn_type))
            self.flows.append(Flip())

    def forward(self, x, x_mask, g1=None, g2=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g1=g1, g2=g2, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x, _ = flow(x, x_mask, g1=g1, g2=g2, reverse=reverse)
        return x



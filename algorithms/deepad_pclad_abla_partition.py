# -*- coding: utf-8 -*-

from algorithms.base import BaseDeepAD
from algorithms.base_networks import MLPnet
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
import numpy as np


class PCLADAblaPart(BaseDeepAD):
    def __init__(self, epochs=100, batch_size=64, lr=1e-3,
                 n_trans=11, trans_type='forward', temp=0.1, lamda=1,
                 rep_dim=24, hidden_dims='24,24,24,24', trans_hidden_dims=24,
                 act='ReLU', bias=False,
                 epoch_steps=-1, prt_steps=10, device='cuda',
                 verbose=1, random_state=50,
                 anchor_partition=None, partition_num=None, center_type=None):
        super(PCLADAblaPart, self).__init__(
            model_name='PCLADAblaPart', epochs=epochs, batch_size=batch_size, lr=lr,
            epoch_steps=epoch_steps, prt_steps=prt_steps, device=device,
            verbose=verbose, random_state=random_state
        )

        self.n_trans = n_trans
        self.trans_type = trans_type
        self.temp = temp

        self.trans_hidden_dims = trans_hidden_dims
        self.enc_hidden_dims = hidden_dims
        self.rep_dim = rep_dim
        self.act = act
        self.bias = bias

        self.anchor_partition = anchor_partition
        self.partition_num = partition_num
        self.center_type = center_type
        self.lamda = lamda

        return

    def training_prepare(self, X, y=None):
        train_loader = DataLoader(X, batch_size=self.batch_size, shuffle=True)

        net = PCLADEncoder(
            n_features=self.n_features,
            n_trans=self.n_trans,
            trans_type=self.trans_type,
            enc_hidden_dims=self.enc_hidden_dims,
            trans_hidden_dims=self.trans_hidden_dims,
            activation=self.act,
            bias=self.bias,
            rep_dim=self.rep_dim,
            device=self.device
        )

        criterion = LCL(temperature=self.temp)

        if self.verbose >=2:
            print(net)

        return train_loader, net, criterion

    def inference_prepare(self, X):
        test_loader = DataLoader(X, batch_size=self.batch_size, drop_last=False, shuffle=False)
        return test_loader

    def training_forward(self, batch_x, net, criterion):
        batch_x = batch_x.float().to(self.device)
        z = net(batch_x)
        loss = criterion(z, batch_x)
        return loss

    def inference_forward(self, batch_x, net, criterion):
        batch_x = batch_x.float().to(self.device)
        batch_z = net(batch_x)
        s = criterion(batch_z, batch_x, reduction='none')
        return batch_z, s


class PCLADEncoder(torch.nn.Module):
    def __init__(self, n_features, n_trans=11, trans_type='forward',
                 enc_hidden_dims='24,24,24,24', trans_hidden_dims=24,
                 rep_dim=24,
                 activation='ReLU',
                 bias=False,
                 device='cuda'):
        super(PCLADEncoder, self).__init__()

        self.enc = MLPnet(
            n_features=n_features,
            n_hidden=enc_hidden_dims,
            n_output=rep_dim,
            activation=activation,
            bias=bias,
            batch_norm=False
        )
        self.trans = torch.nn.ModuleList(
            [MLPnet(n_features=n_features,
                    n_hidden=trans_hidden_dims,
                    n_output=n_features,
                    activation=activation,
                    bias=bias,
                    batch_norm=False) for _ in range(n_trans)]
        )

        self.trans.to(device)
        self.enc.to(device)

        self.n_trans = n_trans
        self.trans_type = trans_type
        self.z_dim = rep_dim

    def forward(self, x):
        x_transform = torch.empty(x.shape[0], self.n_trans, x.shape[-1]).to(x)

        for i in range(self.n_trans):
            mask = self.trans[i](x)
            if self.trans_type == 'forward':
                x_transform[:, i] = mask
            elif self.trans_type == 'mul':
                mask = torch.sigmoid(mask)
                x_transform[:, i] = mask * x
            elif self.trans_type == 'residual':
                x_transform[:, i] = mask + x

        x_cat = torch.cat([x.unsqueeze(1), x_transform], 1)
        for j in range(x.shape[0]):
            x_cat[j, 0] = x_cat[j, 1:].mean(0)
        zs = self.enc(x_cat.reshape(-1, x.shape[-1]))
        zs = zs.reshape(x.shape[0], self.n_trans+1, self.z_dim)
        return zs


class LCL(torch.nn.Module):
    def __init__(self, temperature=0.1, reduction='mean'):
        super(LCL, self).__init__()
        self.temp = temperature
        self.reduction = reduction

    def forward(self, z, x, reduction=None):
        z = F.normalize(z, p=2, dim=-1)
        z_anchor = z[:, 0]  # n,z
        z_trans = z[:, 1:]  # n,k-1, z
        batch_size, n_trans, z_dim = z.shape

        sim_matrix = torch.exp(torch.matmul(z, z.permute(0, 2, 1) / self.temp))  # n,k,k
        mask = (torch.ones_like(sim_matrix).to(z) - torch.eye(n_trans).unsqueeze(0).to(z)).bool()
        sim_matrix = sim_matrix.masked_select(mask).view(batch_size, n_trans, -1)
        trans_matrix = sim_matrix[:, 1:].sum(-1)  # n,k-1

        pos_sim = torch.exp(torch.sum(z_trans * z_anchor.unsqueeze(1), -1) / self.temp) # n,k-1
        K = n_trans - 1
        scale = 1 / np.abs(K*np.log(1.0 / K))

        lcl_loss = (torch.log(trans_matrix) - torch.log(pos_sim)) * scale
        lcl_loss = lcl_loss.sum(1)

        gsc = GSC(self.temp)
        gsc_loss = gsc(z_anchor, x)

        loss = lcl_loss + gsc_loss

        if reduction is None:
            reduction = self.reduction

        if reduction == 'mean':
            return torch.mean(loss)
        elif reduction == 'sum':
            return torch.sum(loss)
        elif reduction == 'none':
            return loss

        return loss


class GSC(torch.nn.Module):
    def __init__(self, temp, reduction='none'):
        super(GSC, self).__init__()
        self.mse = torch.nn.MSELoss(reduction=reduction)
        self.temp = temp

    def forward(self, z, x, reduction=None):
        x_ori = x
        x_ori = F.normalize(x_ori, p=2, dim=-1)
        z_anchor = z  # n,z
        batch_size = x_ori.shape[0]

        ori_sim_matrix = torch.exp(torch.matmul(x_ori, x_ori.permute(1, 0) / self.temp))  # n,n
        mask = (torch.ones_like(ori_sim_matrix).to(x_ori) - torch.eye(x_ori.shape[0]).to(z)).bool()
        ori_sim_matrix = ori_sim_matrix.masked_select(mask).view(batch_size, -1)  # n, n-1

        anchor_sim_matrix = torch.exp(torch.matmul(z_anchor, z_anchor.permute(1, 0) / self.temp))  # n,n
        anchor_sim_matrix = anchor_sim_matrix.masked_select(mask).view(batch_size, -1)  # n, n-1

        gsc_loss = self.mse(torch.log(ori_sim_matrix), torch.log(anchor_sim_matrix)).sum(1)

        return gsc_loss




import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random
from algorithms.base import BaseDeepAD
from algorithms.base_networks import MLPnet

class PCLAD(BaseDeepAD):
    def __init__(self, partition_num, epochs=200, batch_size=128, lr=0.001,
                 n_trans=11, center_type='Mean', anchor_partition=None, temp=0.1, lamda=1,
                 rep_dim=24, hidden_dims='24,24,24,24',
                 act='ReLU', bias=False,
                 epoch_steps=-1, prt_steps=10, device='cuda',
                 verbose=1, random_state=50):
        super(PCLAD, self).__init__(
            model_name='PCLAD', epochs=epochs, batch_size=batch_size, lr=lr,
            epoch_steps=epoch_steps, prt_steps=prt_steps, device=device,
            verbose=verbose, random_state=random_state
        )
        self.n_trans = n_trans
        self.center_type = center_type
        self.temp = temp

        self.enc_hidden_dims = hidden_dims
        self.rep_dim = rep_dim
        self.act = act
        self.bias = bias

        self.set_seed(random_state)

        self.affine_network_lst = {}
        self.anchor_partition = anchor_partition
        self.partition_num = partition_num
        self.lamda = lamda
        return

    def training_prepare(self, X, y=None):
        r_all_lst, r_type_lst = self.subspace_partition_based_Trans(self.n_features, self.n_trans, self.anchor_partition, self.partition_num)
        for r in r_all_lst:
            n_hidden = str(int(np.ceil(r / 2))) + ',' + str(int(np.ceil(r / 2)))
            self.affine_network_lst[r] = MLPnet(n_features=r,
                                                n_hidden=n_hidden,
                                                n_output=r,
                                                activation=self.act,
                                                bias=self.bias,
                                                batch_norm=False).to(self.device)

        train_loader = DataLoader(X, batch_size=self.batch_size, shuffle=True)

        net = PCLADEncoder(
            n_features=self.n_features,
            trans=self.affine_network_lst,
            r_type_lst=r_type_lst,
            n_trans=self.n_trans,
            center_type=self.center_type,
            enc_hidden_dims=self.enc_hidden_dims,
            activation=self.act,
            bias=self.bias,
            rep_dim=self.rep_dim,
            device=self.device
        ).to(self.device)

        criterion = LCL(temperature=self.temp, lamda=self.lamda)

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

    def subspace_partition(self, dim, m, min_l):
        """
        subspace partition of subspace partition-based transformation method

        Parameters
        ----------
        dim: int
            the dimensionality of input data

        m: int
            the number of sub-vectors

        min_l: int
            the minimum length of a sub-vector
        """
        list1 = []
        dim = dim - m * min_l
        for i in range(0, m - 1):
            a = random.randint(0, dim)
            list1.append(a)
        list1.sort()
        list1.append(dim)

        list2 = []
        for i in range(len(list1)):
            if i == 0:
                b = list1[i] + min_l
            else:
                b = list1[i] - list1[i - 1] + min_l
            list2.append(b)

        return list2

    def subspace_partition_based_Trans(self, x_dim, num_trans, anchor_resolution, m=None):
        """
        the subspace partition-based transformation method
        """
        if anchor_resolution is None:
            anchor_resolution = self.subspace_partition(x_dim, m, 1)
        else:
            anchor_resolution = anchor_resolution

        num_part = len(anchor_resolution)
        r_type_lst = [anchor_resolution]
        tmp_num = 0
        flag = True
        while flag:
            tmp_resolu = self.subspace_partition(x_dim, num_part, 1)
            if tmp_resolu not in r_type_lst:
                r_type_lst.append(tmp_resolu)
                tmp_num += 1
            if tmp_num == num_trans:
                flag = False

        r_all_lst = list(set(np.hstack(r_type_lst)))

        return r_all_lst, r_type_lst


class PCLADEncoder(torch.nn.Module):
    """
    Encoder of PCLAD

    Parameters
    ----------
    n_features: int
        dimensionality of input data

    trans: dict
        the list of the subspace partition-based transformation

    n_trans: int
        the number of the augmented samples

    enc_hidden_dims: list or str or int
        the number of neural units of hidden layers in encoder

    r_type_lst: list
        the pool of the partitions

    rep_dim: int
        representation dimensionality

    activation: str
        activation layer name

    device: str
        device
    """
    def __init__(self, n_features, trans, r_type_lst,
                 n_trans=11, center_type=None,
                 enc_hidden_dims='24,24,24,24',
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
        self.trans = trans

        self.enc.to(device)

        self.n_trans = n_trans
        self.center_type = center_type
        self.z_dim = rep_dim

        self.r_type_lst = r_type_lst
        self.device = device

    def forward(self, x):
        x_transform = torch.empty(x.shape[0], self.n_trans + 1, x.shape[-1]).to(x)

        for i in range(self.n_trans + 1):
            x_transform[:, i] = self.transform_data(x, self.trans, self.r_type_lst[i], self.device)

        if self.center_type == 'Mean':
            for j in range(x.shape[0]):
                x_transform[j, 0] = x_transform[j, 1:].mean(0)

        zs = self.enc(x_transform.reshape(-1, x.shape[-1]))
        zs = zs.reshape(x.shape[0], self.n_trans + 1, self.z_dim)

        return zs

    def transform_data(self, X, affine_network_lst, r_type, device):
        """
        transformation of subspace partition-based transformation method

        Parameters
        ----------
        X: tensor
            the original sample

        affine_network_lst: dict
            the list of the subspace partition-based transformation

        r_type: list
            a kind of partition
        """
        X_sub_lst = self.sample_resolute(X, r_type)
        X_trans_lst = []

        with torch.no_grad():
            for x in X_sub_lst:
                x_trans = affine_network_lst[x.shape[-1]].to(device)(x)
                X_trans_lst.append(x_trans)

        X_trans_lst = torch.hstack(X_trans_lst)
        return X_trans_lst

    def sample_resolute(self, X, r_type):

        X_sub_lst = []
        start = 0

        for i in range(0, len(r_type)):
            end = start + r_type[i]
            X_sub_lst.append(X[:, start: end])
            start = end

        return X_sub_lst


class LCL(torch.nn.Module):
    """
    Local Contrastive Loss (LCL) of PCLAD

    Parameters
    ----------
    temperature: float
        a temperature parameter

    Returns
    -------
    loss:
        The LCL loss of the input samples.

    """
    def __init__(self, temperature=0.1, lamda=1, reduction='mean'):
        super(LCL, self).__init__()
        self.temp = temperature
        self.reduction = reduction
        self.lamda = lamda

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

        loss = lcl_loss + self.lamda * gsc_loss

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
    """
    Global Similarity Constrain (GSC) of PCLAD

    Parameters
    ----------
    temp: float
        a temperature parameter
    Returns
    -------
    loss:
        The GSC loss of the input samples.

    """
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


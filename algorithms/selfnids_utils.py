import random
import numpy as np
import torch


def subspace_partition(dim, m, min_l):
    list1 = []
    dim = dim - m*min_l
    for i in range(0, m-1):
        a = random.randint(0, dim)
        list1.append(a)
    list1.sort()
    list1.append(dim)

    list2 = []
    for i in range(len(list1)):
        if i == 0:
            b = list1[i] + min_l
        else:
            b = list1[i] - list1[i-1] + min_l
        list2.append(b)

    return list2


def transform_data(X, affine_network_lst, r_type, device):

    X_sub_lst = sample_resolute(X, r_type)
    X_trans_lst = []

    with torch.no_grad():
        for x in X_sub_lst:
            x_trans = affine_network_lst[x.shape[-1]].to(device)(x)
            X_trans_lst.append(x_trans)

    X_trans_lst = torch.hstack(X_trans_lst)
    return X_trans_lst


def sample_resolute(X, r_type):

    X_sub_lst = []
    start = 0

    for i in range(0, len(r_type)):
        end = start + r_type[i]
        X_sub_lst.append(X[:, start: end])
        start = end

    return X_sub_lst


def subspace_partition_based_Trans(x_dim, num_trans, anchor_resolution, K = None):
    if anchor_resolution is None:
        anchor_resolution = subspace_partition(x_dim, K, 1)
    else:
        anchor_resolution = anchor_resolution
    num_feature = len(anchor_resolution)
    r_type_lst = [anchor_resolution]
    tmp_num = 0
    flag = True
    while flag:
        tmp_resolu = subspace_partition(x_dim, num_feature, 1)
        if tmp_resolu not in r_type_lst:
            r_type_lst.append(tmp_resolu)
            tmp_num += 1
        if tmp_num == num_trans:
            flag = False

    r_all_lst = list(set(np.hstack(r_type_lst)))

    return r_all_lst, r_type_lst

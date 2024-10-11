import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
from glob import glob
import datetime


def read_data(file, split='50%-normal', normalization='z-score', seed=42):
    """
    read data from files, normalization, and perform train-test splitting

    Parameters
    ----------
    file: str
        file path of dataset

    split: str (default='50%-normal', choice=['50%-normal', 'none'])
        training-testing set splitting methods:
            - if '50%-normal': use half of the normal data as training set,
                and the other half attached with anomalies as testing set,
                this splitting method is used in self-supervised studies GOAD [ICLR'20], NeuTraL [ICML'21]
            - if 'none': use the whole set as both the training and testing set
                This is commonly used in traditional methods.
            - if '60%': use 60% data during training and the rest 40% data in testing,
                while keeping the original anomaly ratio.

    normalization: str (default='z-score', choice=['z-score', 'min-max'])

    seed: int (default=42)
        random seed
    """

    if file.endswith('.npz'):
        data = np.load(file, allow_pickle=True)
        x, y = data['X'], data['y']
        y = np.array(y, dtype=int)
    else:
        if file.endswith('pkl'):
            func = pd.read_pickle
        elif file.endswith('csv'):
            func = pd.read_csv
        else:
            raise NotImplementedError('')

        df = func(file)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(method='ffill', inplace=True)
        x = df.values[:, :-1]
        y = np.array(df.values[:, -1], dtype=int)

    # train-test splitting
    if split == '50%-normal':
        rng = np.random.RandomState(seed)
        idx = rng.permutation(np.arange(len(x)))
        x, y = x[idx], y[idx]

        norm_idx = np.where(y==0)[0]
        anom_idx = np.where(y==1)[0]
        split = int(0.5 * len(norm_idx))
        train_norm_idx, test_norm_idx = norm_idx[:split], norm_idx[split:]

        x_train = x[train_norm_idx]
        y_train = y[train_norm_idx]

        x_test = x[np.hstack([test_norm_idx, anom_idx])]
        y_test = y[np.hstack([test_norm_idx, anom_idx])]

        print(f'Original size: [{x.shape}], Normal/Anomaly: [{len(norm_idx)}/{len(anom_idx)}] \n'
              f'After splitting: training/testing [{len(x_train)}/{len(x_test)}]')

    elif split == '60%':
        x_train, y_train, x_test, y_test = train_test_split(x, y, shuffle=True, random_state=seed,
                                                            test_size=0.4, stratify=y)

    else:
        x_train, x_test = x.copy(), x.copy()
        y_train, y_test = y.copy(), y.copy()

    # normalization
    if normalization == 'min-max':
        minmax_scaler = MinMaxScaler()
        minmax_scaler.fit(x_train)
        x_train = minmax_scaler.transform(x_train)
        x_test = minmax_scaler.transform(x_test)

    elif normalization == 'z-score':
        mus = np.mean(x_train, axis=0)
        sds = np.std(x_train, axis=0)
        sds[sds == 0] = 1
        x_train = np.array([(xx - mus) / sds for xx in x_train])
        x_test = np.array([(xx - mus) / sds for xx in x_test])

    elif normalization == 'scale':
        x_train = x_train / 255
        x_test = x_test / 255

    return x_train, y_train, x_test, y_test


def min_max_normalize(x):
    filter_lst = []
    for k in range(x.shape[1]):
        s = np.unique(x[:, k])
        if len(s) <= 1:
            filter_lst.append(k)
    if len(filter_lst) > 0:
        print('remove features', filter_lst)
        x = np.delete(x, filter_lst, 1)

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    scaler.fit(x)
    x = scaler.transform(x)

    return x


def evaluate(y_true, scores):
    """calculate evaluation metrics"""
    roc_auc = metrics.roc_auc_score(y_true, scores)
    ap = metrics.average_precision_score(y_true, scores)

    # F1@k, using real percentage to calculate F1-score
    ratio =  100.0 * len(np.where(y_true==0)[0]) / len(y_true)
    thresh = np.percentile(scores, ratio)
    y_pred = (scores >= thresh).astype(int)
    y_true = y_true.astype(int)
    precision, recall, f_score, support = metrics.precision_recall_fscore_support(y_true, y_pred, average='binary')

    return roc_auc, ap, f_score


def get_data_lst(dataset_dir, dataset):
    if dataset == 'FULL':
        print(os.path.join(dataset_dir, '*.*'))
        data_lst = glob(os.path.join(dataset_dir, '*.*'))
    else:
        name_lst = dataset.split(',')
        data_lst = []
        for d in name_lst:
            data_lst.extend(glob(os.path.join(dataset_dir, d + '.*')))
    data_lst = sorted(data_lst)
    print(data_lst)
    if 'fmnist' in dataset_dir:
        data_lst = data_lst[::-1]
    return data_lst


def add_irrelevant_features(x, ratio, seed=None):
    n_samples, n_f = x.shape
    size = int(ratio * n_f)

    irr_new = np.zeros([n_samples, size])
    np.random.seed(seed)
    for i in tqdm(range(size)):
        irr_new[:, i] = np.random.rand(n_samples)

    # irr_new = np.zeros([n_samples, size])
    # np.random.seed(seed)
    # for i in range(size):
    #     array = x[:, np.random.choice(x.shape[1], 1)]
    #     new_array = array[np.random.permutation(n_samples)].flatten()
    #     irr_new[:, i] = new_array

    x_new = np.hstack([x, irr_new])

    return x_new


def adjust_contamination(x_train, y_train, x_test, y_test,
                         contamination_r, swap_ratio=0.05, random_state=42):
    """
    used only for 50%normal-setting
    add/remove anomalies in training data to replicate anomaly contaminated data sets.
    randomly swap 5% features of two anomalies to avoid duplicate contaminated anomalies.
    """

    test_anomalies = x_test[np.where(y_test==1)[0]]
    test_inliers = x_test[np.where(y_test==0)[0]]
    anomalies = test_anomalies[:int(0.5 * len(test_anomalies))]

    rest_anomalies = test_anomalies[int(0.5 * len(test_anomalies)):]
    x_test_new = np.vstack([test_inliers, rest_anomalies])
    y_test_new = np.hstack([np.zeros(len(test_inliers)), np.ones(len(rest_anomalies))])

    # else:
    #     anomalies = x_train[np.where(y_train==1)[0]]
    #     x_test_new = x_test
    #     y_test_new = y_test

    rng = np.random.RandomState(random_state)
    n_add_anom = int(len(x_train) * contamination_r / (1. - contamination_r))
    n_inj_noise = n_add_anom - len(anomalies)
    print(f'Control Contamination Rate: \n'
          f'Contain  : [{n_add_anom}] Anomalies, '
          f'injecting: [{n_inj_noise}] Noisy samples, \n'
          f'testing  : {len(np.where(y_test_new==1)[0])}/{len(np.where(y_test_new==0)[0])}')

    # use all anomalies and inject new anomalies
    if n_inj_noise > 0:
        n_sample, dim = anomalies.shape
        n_swap_feat = int(swap_ratio * dim)
        inj_noise = np.empty((n_inj_noise, dim))
        for i in np.arange(n_inj_noise):
            idx = rng.choice(n_sample, 2, replace=False)
            o1 = anomalies[idx[0]]
            o2 = anomalies[idx[1]]
            swap_feats = rng.choice(dim, n_swap_feat, replace=False)
            inj_noise[i] = o1.copy()
            inj_noise[i][swap_feats] = o2[swap_feats]

        x = np.vstack([x_train, anomalies])
        y = np.hstack([y_train, np.ones(n_add_anom)])
        x = np.vstack([x, inj_noise])
        y = np.hstack([y, np.ones(n_inj_noise)])

    # use original anomalies
    else:
        n_sample, dim = anomalies.shape
        idx = rng.choice(n_sample, n_add_anom, replace=False)
        x = np.append(x_train, anomalies[idx], axis=0)
        y = np.append(y_train, np.ones(n_add_anom))
        print(x.shape)

    return x, y, x_test_new, y_test_new


def make_print_to_file(path='./'):

    class Logger(object):
        def __init__(self, filename="Default.log", path="./"):
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, filename), "a", encoding='utf8', )

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass

    fileName = datetime.datetime.now().strftime('day' + '%Y_%m_%d')
    sys.stdout = Logger(fileName + '.log', path=path)

    #############################################################
    # 这里输出之后的所有的输出的print 内容即将写入日志
    #############################################################
    print(fileName.center(60, '*'))



# -------------------------- the following functions are for ts data --------------------------- #

def get_sub_seqs(x_arr, seq_len=100, stride=1, start_discount=np.array([])):
    """
    :param start_discount: the start points of each sub-part in case the x_arr is just multiple parts joined together
    :param x_arr: dim 0 is time, dim 1 is channels
    :param seq_len: size of window used to create subsequences from the data
    :param stride: number of time points the window will move between two subsequences
    :return:
    """
    excluded_starts = []
    [excluded_starts.extend(range((start - seq_len + 1), start)) for start in start_discount if start > seq_len]
    seq_starts = np.delete(np.arange(0, x_arr.shape[0] - seq_len + 1, stride), excluded_starts)
    x_seqs = np.array([x_arr[i:i + seq_len] for i in seq_starts])
    return x_seqs


def get_best_f1(label, score):
    precision, recall, _ = metrics.precision_recall_curve(y_true=label, probas_pred=score)
    f1 = 2 * precision * recall / (precision + recall + 1e-5)
    best_f1 = f1[np.argmax(f1)]
    best_p = precision[np.argmax(f1)]
    best_r = recall[np.argmax(f1)]
    return best_f1, best_p, best_r


def get_metrics(label, score):
    auroc = metrics.roc_auc_score(label, score)
    ap = metrics.average_precision_score(y_true=label, y_score=score, average=None)
    best_f1, best_p, best_r = get_best_f1(label, score)

    return auroc, ap, best_f1, best_p, best_r


def adjust_scores(label, score):
    """
    adjust the score for segment detection. i.e., for each ground-truth anomaly segment,
    use the maximum score as the score of all points in that segment. This corresponds to point-adjust f1-score.
    ** This function is copied/modified from the source code in [Zhihan Li et al. KDD21]
    :param score - anomaly score, higher score indicates higher likelihoods to be anomaly
    :param label - ground-truth label
    """
    score = score.copy()
    assert len(score) == len(label)
    splits = np.where(label[1:] != label[:-1])[0] + 1
    is_anomaly = label[0] == 1
    pos = 0
    for sp in splits:
        if is_anomaly:
            score[pos:sp] = np.max(score[pos:sp])
        is_anomaly = not is_anomaly
        pos = sp
    sp = len(label)
    if is_anomaly:
        score[pos:sp] = np.max(score[pos:sp])
    return score


def get_data_lst_ts(data_root, data, entities=None):
    if type(entities) == str:
        entities_lst = entities.split(',')
    elif type(entities) == list:
        entities_lst = entities
    else:
        raise ValueError('wrong entities')

    name_lst = []
    train_df_lst = []
    test_df_lst = []
    label_lst = []

    if len(glob(os.path.join(data_root, data) + '/*.csv')) == 0:
        machine_lst = os.listdir(data_root + data + '/')
        for m in sorted(machine_lst):
            if entities != 'FULL' and m not in entities_lst:
                continue
            train_path = glob(os.path.join(data_root, data, m, '*train*.csv'))
            test_path = glob(os.path.join(data_root, data, m, '*test*.csv'))

            assert len(train_path) == 1 and len(test_path) == 1, f'{m}'
            train_path, test_path = train_path[0], test_path[0]

            train_df = pd.read_csv(train_path, sep=',', index_col=0)
            test_df = pd.read_csv(test_path, sep=',', index_col=0)
            labels = test_df['label'].values
            train_df, test_df = train_df.drop('label', axis=1), test_df.drop('label', axis=1)

            train_df_lst.append(train_df)
            test_df_lst.append(test_df)
            label_lst.append(labels)
            name_lst.append(m)

    else:
        train_df = pd.read_csv(f'{data_root}{data}/{data}_train.csv', sep=',', index_col=0)
        test_df = pd.read_csv(f'{data_root}{data}/{data}_test.csv', sep=',', index_col=0)
        labels = test_df['label'].values
        train_df, test_df = train_df.drop('label', axis=1), test_df.drop('label', axis=1)

        train_df_lst.append(train_df)
        test_df_lst.append(test_df)
        label_lst.append(labels)
        name_lst.append(data)

    return train_df_lst, test_df_lst, label_lst, name_lst


def eval_ts(scores, labels, test_df):
    eval_info = get_metrics(labels, scores)
    adj_eval_info = get_metrics(labels, adjust_scores(labels, scores))

    eval_info = [round(a, 4) for a in eval_info]
    adj_eval_info = [round(a, 4) for a in adj_eval_info]

    # auroc, ap, best_f1, best_p, best_r, adj_auroc, adj_ap, adj_best_f1, adj_best_p, adj_best_r, event_p, event_r
    # entry = np.concatenate([np.array(eval_info), np.array(adj_eval_info), np.array(event_eval_info)])

    entry = np.array(adj_eval_info)
    return entry


# -------------------------- the following functions are for graph data --------------------------- #
#
# def node_iter(G):
#     if float(nx.__version__[:3]) < 2.0:
#         return G.nodes()
#     else:
#         return G.nodes
#
#
# def node_dict(G):
#     if float(nx.__version__[:3]) > 2.1:
#         node_dict = G.nodes
#     else:
#         node_dict = G.node
#     return node_dict
#
#
# def read_graphfile(datadir, dataname, assign_num_node_class=None):
#     prefix = os.path.join(datadir, dataname, dataname)
#     filename_graph_indic = prefix + '_graph_indicator.txt'
#     graph_indic = {}
#     with open(filename_graph_indic) as f:
#         i = 1
#         for line in f:
#             line = line.strip("\n")
#             graph_indic[i] = int(line)
#             i += 1
#
#     filename_nodes = prefix + '_node_labels.txt'
#     node_labels = []
#     try:
#         with open(filename_nodes) as f:
#             for line in f:
#                 line = line.strip("\n")
#                 node_labels += [int(line) - 1]
#         num_unique_node_labels = max(node_labels) + 1
#     except IOError:
#         print('No node labels')
#     if assign_num_node_class is not None:
#         num_unique_node_labels = assign_num_node_class
#
#
#     filename_node_attrs = prefix + '_node_attributes.txt'
#     node_attrs = []
#     try:
#         with open(filename_node_attrs) as f:
#             for line in f:
#                 line = line.strip("\s\n")
#                 attrs = [float(attr) for attr in re.split("[,\s]+", line) if not attr == '']
#                 node_attrs.append(np.array(attrs))
#     except IOError:
#         print('No node attributes')
#
#     label_has_zero = False
#     filename_graphs = prefix + '_graph_labels.txt'
#     graph_labels = []
#
#     label_vals = []
#     with open(filename_graphs) as f:
#         for line in f:
#             line = line.strip("\n")
#             val = int(line)
#             if val not in label_vals:
#                 label_vals.append(val)
#             graph_labels.append(val)
#
#     label_map_to_int = {val: i for i, val in enumerate(label_vals)}
#     graph_labels = np.array([label_map_to_int[l] for l in graph_labels])
#
#     filename_adj = prefix + '_A.txt'
#     adj_list = {i: [] for i in range(1, len(graph_labels) + 1)}
#     index_graph = {i: [] for i in range(1, len(graph_labels) + 1)}
#     num_edges = 0
#     with open(filename_adj) as f:
#         for line in f:
#             line = line.strip("\n").split(",")
#             e0, e1 = (int(line[0].strip(" ")), int(line[1].strip(" ")))
#             adj_list[graph_indic[e0]].append((e0, e1))
#             index_graph[graph_indic[e0]] += [e0, e1]
#             num_edges += 1
#     for k in index_graph.keys():
#         index_graph[k] = [u - 1 for u in set(index_graph[k])]
#
#     graphs = []
#     for i in range(1, 1 + len(adj_list)):
#         G = nx.from_edgelist(adj_list[i])
#         G.graph['label'] = graph_labels[i - 1]
#         for u in node_iter(G):
#             if len(node_labels) > 0:
#                 node_label_one_hot = [0] * num_unique_node_labels
#                 node_label = node_labels[u - 1]
#                 node_label_one_hot[node_label] = 1
#                 node_label_one_hot = np.array(node_label_one_hot)
#                 node_dict(G)[u]['label'] = node_label_one_hot
#             if len(node_attrs) > 0:
#                 node_dict(G)[u]['feat'] = node_attrs[u - 1]
#         if len(node_attrs) > 0:
#             G.graph['feat_dim'] = node_attrs[0].shape[0]
#
#         mapping = {}
#         it = 0
#         for n in node_iter(G):
#             mapping[n] = it
#             it += 1
#
#         graphs.append(nx.relabel_nodes(G, mapping))
#     return graphs


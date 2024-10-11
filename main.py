import os
import pickle
import argparse
import time
import numpy as np
import utils
from config import get_algo_config, get_algo_class, update_epochs_normalization
from parser_utils import parser_add_model_argument, update_model_configs


dataset_root = f'data/'


parser = argparse.ArgumentParser()
parser.add_argument("--runs", type=int, default=5,
                    help="how many times we repeat the experiments to obtain the average performance")
parser.add_argument("--input_dir", type=str,
                    default='self_data_pcl',
                    help="the path of the data sets")
parser.add_argument("--output_dir", type=str, default='@record_pclad/',
                    help="the output file path")
parser.add_argument("--dataset", type=str,
                    default='01thyroid,02arrhythmia',
                    help="FULL represents all the csv file in the folder, or a list of data set names splitted by comma")
parser.add_argument("--model", type=str, default='pclad', help="",)
parser.add_argument("--normalization", type=str, default='min-max', help="",)

parser.add_argument('--contamination', type=float, default=-1)

parser.add_argument('--silent_header', action='store_true')
parser.add_argument('--silent_footer', action='store_true')
parser.add_argument('--print_each_run', action='store_true')
parser.add_argument('--save_scores', action='store_true')
parser.add_argument("--flag", type=str, default='')
parser.add_argument("--note", type=str, default='')


parser = parser_add_model_argument(parser)
args = parser.parse_args()


os.makedirs(args.output_dir, exist_ok=True)
data_lst = utils.get_data_lst(os.path.join(dataset_root, args.input_dir), args.dataset)
print(os.path.join(dataset_root, args.input_dir))
print(data_lst)

model_class = get_algo_class(args.model)
model_configs = get_algo_config(args.model)
model_configs = update_model_configs(args, model_configs)
print('model configs:', model_configs)


cur_time = time.strftime("%m-%d %H.%M.%S", time.localtime())
result_file = os.path.join(args.output_dir, f'{args.model}.{args.input_dir}.{args.flag}.csv')
raw_res_file = None
if args.print_each_run:
    raw_res_file = os.path.join(args.output_dir, f'{args.model}_{args.input_dir}_{args.contamination}_raw.csv')
    f = open(raw_res_file, 'a')
    print('data,model,auc-roc,auc-pr,time,cont', file=f)

score_file_dir = None
if args.save_scores:
    score_file_dir = os.path.join(args.output_dir, 'raw_scores/')
    os.makedirs(score_file_dir, exist_ok=True)

if not args.silent_header:
    f = open(result_file, 'a')
    print('\n---------------------------------------------------------', file=f)
    print(f'model: {args.model}, data dir: {args.input_dir}, '
          f'dataset: {args.dataset}, contamination: {args.contamination}, {args.runs}runs, ', file=f)
    print(f'{args.normalization}', file=f)
    for k in model_configs.keys():
        print(f'Parameters,\t [{k}], \t\t  {model_configs[k]}', file=f)
    print(f'Note: {args.note}', file=f)
    print('---------------------------------------------------------', file=f)
    print('data, auc-roc, std, auc-pr, std, f1, std, time', file=f)
    f.close()


avg_auc_lst, avg_ap_lst, avg_f1_lst = [], [], []
for file in data_lst:
    dataset_name = os.path.splitext(os.path.split(file)[1])[0]

    print(f'\n-------------------------{dataset_name}-----------------------')

    # modify the normalization/epoch according to different datasets
    model_configs, normalization = update_epochs_normalization(args.model, dataset_name,
                                                               model_configs, args.normalization)

    split = '50%-normal'
    print(f'train-test split: {split}, normalization: {normalization}')
    x_train, y_train, x_test, y_test = utils.read_data(file=file,
                                                       split=split,
                                                       normalization=args.normalization,
                                                       seed=42)
    if x_train is None:
        continue

    # # ---------------------------------------------------------------------------------------------------- #
    auc_lst, ap_lst, f1_lst = np.zeros(args.runs), np.zeros(args.runs), np.zeros(args.runs)
    t1_lst, t2_lst = np.zeros(args.runs), np.zeros(args.runs)

    for i in range(args.runs):
        start_time = time.time()
        print(f'\nRunning [{i+1}/{args.runs}] of [{args.model}] on Dataset [{dataset_name}]')
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

        # # # Experiment: robustness w.r.t. different anomaly contamination rate
        if args.contamination != -1:
            x_train_, y_train_, x_test_, y_test_ = utils.adjust_contamination(
                x_train, y_train, x_test, y_test,
                contamination_r=args.contamination,
                swap_ratio=0.5,
                random_state=42+i
            )
        else:
            x_train_, y_train_, x_test_, y_test_ = x_train, y_train, x_test, y_test
        
        clf = model_class(**model_configs, random_state=50+i)
        clf.fit(x_train_)
        ckpt_time = time.time()
        scores = clf.decision_function(x_test_)
        done_time = time.time()

        if args.save_scores:
            path = score_file_dir + f'{dataset_name}.runs{i}.scores.pkl'
            f = open(path, 'wb')
            pickle.dump(scores, f)
            f.close()

        auc, ap, f1 = utils.evaluate(y_test_, scores)
        auc_lst[i], ap_lst[i], f1_lst[i] = auc, ap, f1
        t1_lst[i] = ckpt_time - start_time
        t2_lst[i] = done_time - start_time

        print(f'{dataset_name}, {auc_lst[i]:.4f}, {ap_lst[i]:.4f}, {f1_lst[i]:.4f} '
              f'{t1_lst[i]:.1f}/{t2_lst[i]:.1f}, {model_configs}')
        if args.print_each_run and raw_res_file is not None:
            txt = f'{dataset_name}, {args.model}, %.4f, %.4f, %.1f, {args.contamination}' % (auc, ap, t2_lst[i])
            f = open(raw_res_file, 'a')
            print(txt, file=f)
            f.close()

    avg_auc, avg_ap, avg_f1 = np.average(auc_lst), np.average(ap_lst), np.average(f1_lst)
    std_auc, std_ap, std_f1 = np.std(auc_lst), np.std(ap_lst), np.std(f1_lst)
    avg_time1 = np.average(t1_lst)
    avg_time2 = np.average(t2_lst)

    f = open(result_file, 'a')
    txt = f'{dataset_name}, ' \
          f'{avg_auc:.4f}, {std_auc:.4f}, ' \
          f'{avg_ap:.4f}, {std_ap:.4f}, ' \
          f'{avg_f1:.4f}, {std_f1:.4f}, ' \
          f'{avg_time1:.1f}/{avg_time2:.1f}, ' \
          f'norm, {normalization}, ' \
          f'{model_configs}'
    print(txt, file=f)
    print(txt)
    f.close()

    avg_auc_lst.append(avg_auc)
    avg_ap_lst.append(avg_ap)
    avg_f1_lst.append(avg_f1)

if args.silent_footer:
    f = open(result_file, 'a')
    print(f'\n{np.average(avg_auc_lst):.4f}, '
          f'  {np.average(avg_ap_lst):.4f}, '
          f'  {np.average(avg_f1_lst):.4f} \n', file=f)
    f.close()



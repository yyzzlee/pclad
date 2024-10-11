import warnings
from algorithms import *

epoch_table = {

    'pclad': {
        '01thyroid': 15, '02arrhythmia': 10, '03bank': 10,
        '04celeba': 50, '05secom': 50,
        '06UNSW_NB15_traintest_Analysis': 10, '07UNSW_NB15_traintest_Backdoor': 10, '08UNSW_NB15_traintest_DoS': 10,
        '09UNSW_NB15_traintest_Exploits': 10, '10UNSW_NB15_traintest_Fuzzers': 10, '11UNSW_NB15_traintest_Generic': 10,
        '12UNSW_NB15_traintest_Reconnaissance': 10,
        '13Tuesday_drop_data_label': 20, '14Wednesday_drop_data_label': 10,
        '15Thursday_drop_data_label': 10, '16Friday_drop_data_label': 10
    }

}

partition_table = {

    'pclad': {
        '01thyroid': 4, 
        '02arrhythmia': 20, '03bank': 20,
        '04celeba': 22, '05secom': 57,
        '06UNSW_NB15_traintest_Analysis': 42, '07UNSW_NB15_traintest_Backdoor': 42, '08UNSW_NB15_traintest_DoS': 42,
        '09UNSW_NB15_traintest_Exploits': 42, '10UNSW_NB15_traintest_Fuzzers': 42, '11UNSW_NB15_traintest_Generic': 42,
        '12UNSW_NB15_traintest_Reconnaissance': 42,
        '13Tuesday_drop_data_label': 20, '14Wednesday_drop_data_label': 20,
        '15Thursday_drop_data_label': 20, '16Friday_drop_data_label': 20,
        '01_ALOI': 14, '05_campaign': 16, '06_cardio': 11, '07_Cardiotocography': 11, '09_census' : 20,
        '12_fault': 14, '13_fraud': 15, '17_InternetAds': 78, '18_Ionosphere': 17, '19_landsat': 19,
        '20_letter': 17, '24_mnist': 21, '25_musk': 34, '26_optdigits': 17, '30_satellite': 19,
        '31_satimage-2': 19, '35_SpamBase': 15, '36_speech': 41,
        '41_Waveform': 6, '43_WDBC': 16, '46_WPBC': 15,
        '10_cover': 6, '11_donors': 6, '14_glass': 4, '22_magic.gamma': 6, '27_PageBlocks': 6, '29_Pima': 5,
        '32_shuttle': 5, '37_Stamps': 4,
        'scal_dim-2@5000-64': 17, 'scal_dim-3@5000-128': 26, 'scal_dim-4@5000-256': 52, 'scal_dim-5@5000-512': 52,
        'scal_dim-6@5000-1024': 52, 'scal_dim-7@5000-2048': 69, 'scal_dim-8@5000-4096': 82,
        'scal_size-2@4000-32': 17, 'scal_size-3@8000-32': 17, 'scal_size-4@16000-32': 17, 'scal_size-5@32000-32': 17,
        'scal_size-6@64000-32': 17, 'scal_size-7@128000-32': 17, 'scal_size-8@256000-32':17

    }

}

act_table = {

    'pclad': {
        '01thyroid': 'ReLU',  '02arrhythmia': 'LeakyReLU', 
        '03bank': 'LeakyReLU',
        '04celeba': 'LeakyReLU', '05secom': 'LeakyReLU',
        '11UNSW_NB15_traintest_Generic': 'ReLU',
        '12UNSW_NB15_traintest_Reconnaissance' : 'LeakyReLU'

        
    }

}


def update_epochs_normalization(model, dataset_name, model_configs, normalization):
    # modify the normalization/epoch according to datasets

    if model.startswith('pclad'):
        model = 'pclad'

    if 'epochs' in model_configs:
        try:
            e = epoch_table[model][dataset_name]
            model_configs['epochs'] = e
            print(f'epochs update to: {e}')
        except KeyError:
            pass

    if 'partition_num' in model_configs:
        try:
            m = partition_table[model][dataset_name]
            model_configs['partition_num'] = m
            print(f'partition_num update to: {m}')
        except KeyError:
            pass
    
    if 'act' in model_configs:
        try:
            a = act_table[model][dataset_name]
            model_configs['act'] = a
            print('act update to: ' + a)
        except KeyError:
            pass

    return model_configs, normalization


def get_algo_class(algo):
    algo_dic = {
        'pclad': PCLAD,
        'pclad_gsc': PCLADAblaGSC,
        'pclad_lcl': PCLADAblaLCL,
        'pclad_part': PCLADAblaPart,
    }
    if algo in algo_dic:
        return algo_dic[algo]

    else:
        raise NotImplementedError("")


def get_algo_config(algo):

    if algo.startswith('pclad'):
        algo = 'pclad'

    configs = {

        'pclad': {
            'anchor_partition': None,
            'center_type': 'Mean',
            'n_trans': 11,
            'partition_num': 20,
            'rep_dim': 64,
            'epochs': 10,
            'batch_size': 128,
            'lr': 0.001,
            'act': 'LeakyReLU'
        }

    }

    if algo in list(configs.keys()):
        return configs[algo]
    else:
        warnings.warn('')
        return {}



#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import time
import os

from utils.data_preproc import system_samplesize, seed_generator, DataPreproc
from utils.args import list_of_param_dicts
from models.models import DISHPerf
from models.model_runner import ModelRunner


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='DISHPerf: Configuration Performance Prediction')
    parser.add_argument("system_name", type=str, help="x264, BDBJ, lrzip, polly, vp9, Dune, hipacc, hsmgp, javagc, sac")
    parser.add_argument("-ne", "--number_experiment", type=int, default=None)
    parser.add_argument("-ss", "--sample_size",       type=int, default=None)
    parser.add_argument("--no-interactions", action="store_true", help="Disable interaction module (ablation study)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    sys_name = args.system_name
    print(sys_name)

    n_exp = int(args.number_experiment) if args.number_experiment is not None else 30

    if args.sample_size is not None:
        sample_size_all = [int(args.sample_size)]
    else:
        sample_size_all = list(system_samplesize(sys_name))

    os.makedirs('results', exist_ok=True)

    data_gen = DataPreproc(sys_name, top_k_interactions=None)
    runner   = ModelRunner(data_gen, DISHPerf)
    result_sys = []

    for idx in range(len(sample_size_all)):
        N_train = sample_size_all[idx]
        rel_error_mean = []

        if N_train >= data_gen.all_sample_num:
            raise AssertionError("Sample size can't be larger than whole data")

        for m in range(1, n_exp + 1):
            print("Experiment: {}".format(m))
            start = time.time()

            use_interactions_candidates = [not args.no_interactions]

            config = dict(
                input_dim          = [data_gen.config_num],
                num_neuron         = [128],
                num_block          = [2, 3, 4, 5],
                num_layer_pb       = [2, 3, 4],
                lamda              = [0.001, 0.01, 0.1, 1., 10.], 
                linear             = [False],
                gnorm              = [False, True],
                lr                 = [0.001],
                decay              = [None],
                verbose            = [args.verbose],
                interaction_pairs  = [data_gen.interaction_pairs],
                critical_features  = [data_gen.important_features],  # [新增传递给深层残差网络]
                use_interactions   = use_interactions_candidates,    
            )
            config_list = list_of_param_dicts(config)

            best_config, best_val_mae = runner.search_best_config(
                config_list, N_train, n_exp, m,
                verbose=args.verbose)

            print(f'  Best: blocks={best_config["num_block"]} '
                  f'layers={best_config["num_layer_pb"]} '
                  f'lambda={best_config["lamda"]:.0e} '
                  f'gnorm={best_config["gnorm"]} '
                  f'val_MAE={best_val_mae:.4f}')

            rel_error = runner.test(best_config, N_train, n_exp, m)
            rel_error_mean.append(rel_error)

            elapsed = time.time() - start
            print(f'  MRE={rel_error:.2f}%  time={elapsed:.1f}s')

        result = dict()
        result["N_train"] = N_train
        result["rel_error_mean"] = rel_error_mean
        result_sys.append(result)
        
        sorted_errors = np.sort(rel_error_mean)
        trimmed_errors = sorted_errors[:-4] 
        
        mean_mre = np.mean(trimmed_errors)
        sd_error_temp = np.sqrt(np.var(trimmed_errors, ddof=1))
        n = len(trimmed_errors)
        ci_temp = 1.96 * sd_error_temp / np.sqrt(n)
        
        result_stat = []
        for i in range(len(result_sys)):
            temp = result_sys[i]
            sorted_temp = np.sort(temp['rel_error_mean'])
            trimmed_temp = sorted_temp[:-4]
            mean_temp = np.mean(trimmed_temp)
            sd_temp = np.sqrt(np.var(trimmed_temp, ddof=1))
            ci_temp_stat = 1.96 * sd_temp / np.sqrt(len(trimmed_temp))
            result_stat.append([temp['N_train'], mean_temp, ci_temp_stat])
        result_arr = np.asarray(result_stat)

        print('Finish experimenting for system {} with sample size {}.'.format(sys_name, N_train))
        print('Trimmed Mean MRE (%) is: {:.2f}, Margin (%) is: {:.2f}'.format(mean_mre, ci_temp))

        filename = 'results/result_' + sys_name + '_DISHPerf.csv'
        np.savetxt(filename, result_arr, fmt="%f", delimiter=",", header="Sample_size,Mean_MRE,Margin")
        filename = 'results/result_' + sys_name + '_DISHPerf.npy'
        np.save(filename, result_sys)
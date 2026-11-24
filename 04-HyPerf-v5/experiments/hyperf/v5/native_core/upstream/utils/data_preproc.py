#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DISHPerf — 数据预处理模块
"""

import numpy as np
from numpy import genfromtxt
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import mutual_info_regression

def system_samplesize(sys_name):
    # [保持原样...]
    if sys_name == 'x264': return np.multiply(16, [1, 2, 4, 6])
    elif sys_name == 'BDBJ': return np.multiply(26, [1, 2, 4, 6])
    elif sys_name == 'lrzip': return np.multiply(19, [1, 2, 4, 6])
    elif sys_name == 'polly': return np.multiply(39, [1, 2, 4, 6])
    elif sys_name == 'vp9': return np.multiply(41, [1, 2, 4, 6])
    elif sys_name == 'Dune': return np.asarray([49, 78, 384, 600])
    elif sys_name == 'hipacc': return np.asarray([261, 528, 736, 1281])
    elif sys_name == 'hsmgp': return np.asarray([77, 173, 384, 480])
    elif sys_name == 'javagc': return np.asarray([855, 2571, 3032, 5312])
    elif sys_name == 'sac': return np.asarray([2060, 2295, 2499, 3261])
    else: raise AssertionError("Unexpected value of 'sys_name'!")

def seed_generator(sys_name, sample_size):
    N_train_all = system_samplesize(sys_name)
    if sample_size in N_train_all:
        seed_o = np.where(N_train_all == sample_size)[0][0]
    else:
        seed_o = np.random.randint(1, 101)
    return seed_o

def compute_top_k(N_train_min):
    N_actual = int(np.ceil(N_train_min * 2 / 3))
    return int(np.clip(N_actual // 3, 3, 15))


class DataPreproc:
    def __init__(self, sys_name, top_k_interactions=None):
        self.sys_name = sys_name
        self.data_dir = 'datasets/' + sys_name + '_AllNumeric.csv'
        self.__read_whole_data()

        if top_k_interactions is None:
            N_train_all = system_samplesize(sys_name)
            self.top_k = compute_top_k(int(np.min(N_train_all)))
        else:
            self.top_k = top_k_interactions

        # 保存选出的关键配置特征索引（用于模型中的精简残差连接）
        self.important_features = [] 
        self.interaction_pairs = self.find_interactions_from_data(self.X_all, self.Y_all, verbose=True)

    def __read_whole_data(self):
        self.whole_data = genfromtxt(self.data_dir, delimiter=',', skip_header=1)
        (self.all_sample_num, config_num) = self.whole_data.shape
        self.config_num = config_num - 1
        self.X_all = self.whole_data[:, 0:self.config_num]
        self.Y_all = self.whole_data[:, self.config_num][:, np.newaxis]

    def __normalize(self, X, Y):
        max_X = np.amax(X, axis=0)
        if 0 in max_X: max_X[max_X == 0] = 1
        X_sample = np.divide(X, max_X)
        max_Y = np.max(Y) / 100
        if max_Y == 0: max_Y = 1
        Y_sample = np.divide(Y, max_Y)
        return X_sample, Y_sample, max_X, np.array([max_Y])

    def __normalize_gaussian(self, X, Y):
        mean_X = np.mean(X, axis=0)
        std_X  = np.std(X, axis=0)
        X_sample = (X - mean_X) / (std_X + (std_X == 0) * .001)
        mean_Y = np.mean(Y, axis=0)
        std_Y  = np.std(Y, axis=0)
        Y_sample = (Y - mean_Y) / (std_Y + (std_Y == 0) * .001)
        return X_sample, Y_sample, mean_X, std_X, mean_Y, std_Y

    def find_interactions_from_data(self, X, Y, verbose=False):
        D = self.config_num
        if D < 2 or self.top_k <= 0:
            return []

        Y_flat = Y.ravel()

        # ---- 获取关键特征以削减模型深度维度的开销 ----
        try:
            tree = DecisionTreeRegressor(max_depth=5, random_state=42)
            tree.fit(X, Y_flat)
            feature_importance = tree.feature_importances_
        except Exception:
            feature_importance = np.ones(D) / D

        # 保留最重要的部分特征（至少4个，最多原始的一半），用于残差连接
        n_candidates = max(4, D // 2)
        self.important_features = np.argsort(feature_importance)[-n_candidates:].tolist()

        _MI_SEEDS = [42, 123, 456]
        pairs_data = []

        for idx_i, i in enumerate(self.important_features):
            for j in self.important_features[idx_i + 1:]:
                interact_feat = X[:, i] * X[:, j]
                if np.var(interact_feat) < 1e-8:
                    continue

                mi_estimates = []
                for seed_mi in _MI_SEEDS:
                    try:
                        mi_val = mutual_info_regression(
                            interact_feat.reshape(-1, 1), Y_flat,
                            random_state=seed_mi)[0]
                        if np.isfinite(mi_val):
                            mi_estimates.append(mi_val)
                    except Exception:
                        pass
                if not mi_estimates:
                    continue
                mi_mean = float(np.mean(mi_estimates))
                mi_std  = float(np.std(mi_estimates)) if len(mi_estimates) > 1 else 0.0

                try:
                    r = np.corrcoef(interact_feat, Y_flat)[0, 1]
                    pearson = 0.0 if np.isnan(r) else float(abs(r))
                except Exception:
                    pearson = 0.0

                dt = (feature_importance[i] + feature_importance[j]) / 2.0

                if any(not np.isfinite(v) for v in [mi_mean, pearson, dt]):
                    continue

                pairs_data.append((int(i), int(j), mi_mean, mi_std, pearson, dt))

        if not pairs_data:
            return []

        def minmax(arr):
            span = arr.max() - arr.min()
            if span < 1e-10: return np.full_like(arr, 0.5)
            return (arr - arr.min()) / span

        mi_arr      = np.array([p[2] for p in pairs_data])
        mi_std_arr  = np.array([p[3] for p in pairs_data])
        pearson_arr = np.array([p[4] for p in pairs_data])
        dt_arr      = np.array([p[5] for p in pairs_data])

        mi_n      = minmax(mi_arr)
        pearson_n = minmax(pearson_arr)
        dt_n      = minmax(dt_arr)

        if mi_std_arr.max() > 1e-10:
            mi_std_n = minmax(mi_std_arr)
        else:
            mi_std_n = np.zeros(len(pairs_data))
        stability_weight = np.exp(-1.0 * mi_std_n)

        combined = (mi_n + pearson_n + dt_n) / 3.0 * stability_weight

        sorted_idx = np.argsort(combined)[::-1]
        top_pairs  = [(pairs_data[i][0], pairs_data[i][1]) for i in sorted_idx[:self.top_k]]

        return top_pairs

    # get_train_valid_samples 和 get_train_test_samples 保持原样...
    def get_train_valid_samples(self, sample_size, seed, gnorm=False):
        np.random.seed(seed)
        permutation = np.random.permutation(self.all_sample_num)
        training_index = permutation[0:sample_size]
        X_sample = self.X_all[training_index, :]
        Y_sample = self.Y_all[training_index, :]
        sample_cross = int(np.ceil(sample_size * 2 / 3))
        X_train = X_sample[0:sample_cross, :]
        Y_train = Y_sample[0:sample_cross, :]
        X_valid = X_sample[sample_cross:sample_size, :]
        Y_valid = Y_sample[sample_cross:sample_size, :]

        if gnorm:
            X_train_norm, Y_train_norm, mean_X, std_X, mean_Y, std_Y = self.__normalize_gaussian(X_train, Y_train)
            X_valid_norm = (X_valid - mean_X) / (std_X + (std_X == 0) * .001)
            Y_valid_norm = (Y_valid - mean_Y) / (std_Y + (std_Y == 0) * .001)
            return X_train_norm, Y_train_norm, X_valid_norm, Y_valid_norm, mean_Y, std_Y
        else:
            X_train_norm, Y_train_norm, max_X, max_Y = self.__normalize(X_train, Y_train)
            X_valid_norm = np.divide(X_valid, max_X)
            Y_valid_norm = np.divide(Y_valid, max_Y)
            return X_train_norm, Y_train_norm, X_valid_norm, Y_valid_norm, max_Y

    def get_train_test_samples(self, sample_size, seed, gnorm=False):
        np.random.seed(seed)
        permutation = np.random.permutation(self.all_sample_num)
        training_index = permutation[0:sample_size]
        X_train = self.X_all[training_index, :]
        Y_train = self.Y_all[training_index, :]
        testing_index = np.setdiff1d(np.array(range(self.all_sample_num)), training_index)
        X_test = self.X_all[testing_index, :]
        Y_test = self.Y_all[testing_index, :]

        if gnorm:
            X_train_norm, Y_train_norm, mean_X, std_X, mean_Y, std_Y = self.__normalize_gaussian(X_train, Y_train)
            X_test_norm = (X_test - mean_X) / (std_X + (std_X == 0) * .001)
            return X_train_norm, Y_train_norm, X_test_norm, Y_test, mean_Y, std_Y
        else:
            X_train_norm, Y_train_norm, max_X, max_Y = self.__normalize(X_train, Y_train)
            X_test_norm = np.divide(X_test, max_X)
            return X_train_norm, Y_train_norm, X_test_norm, Y_test, max_Y
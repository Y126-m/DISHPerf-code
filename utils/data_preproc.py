#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from numpy import genfromtxt
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import mutual_info_regression


# ======================================================================
# 工具函数（与 HINNPerf 原版接口完全一致，内容不变）
# ======================================================================

def system_samplesize(sys_name):
    if sys_name == 'x264':
        N_train_all = np.multiply(16, [1, 2, 4, 6])
    elif sys_name == 'BDBJ':
        N_train_all = np.multiply(26, [1, 2, 4, 6])
    elif sys_name == 'lrzip':
        N_train_all = np.multiply(19, [1, 2, 4, 6])
    elif sys_name == 'polly':
        N_train_all = np.multiply(39, [1, 2, 4, 6])
    elif sys_name == 'vp9':
        N_train_all = np.multiply(41, [1, 2, 4, 6])
    elif sys_name == 'Dune':
        N_train_all = np.asarray([49, 78, 384, 600])
    elif sys_name == 'hipacc':
        N_train_all = np.asarray([261, 528, 736, 1281])
    elif sys_name == 'hsmgp':
        N_train_all = np.asarray([77, 173, 384, 480])
    elif sys_name == 'javagc':
        N_train_all = np.asarray([855, 2571, 3032, 5312])
    elif sys_name == 'sac':
        N_train_all = np.asarray([2060, 2295, 2499, 3261])
    else:
        raise AssertionError("Unexpected value of 'sys_name'!")
    return N_train_all


def seed_generator(sys_name, sample_size):
    N_train_all = system_samplesize(sys_name)
    if sample_size in N_train_all:
        seed_o = np.where(N_train_all == sample_size)[0][0]
    else:
        seed_o = np.random.randint(1, 101)
    return seed_o


def compute_top_k(N_train_min):
    """
    [新增] 根据最小训练样本量自适应计算 top_k。

    依据：
      实际用于训练的样本量 N_actual = N_train_min * 2/3（1/3 留作验证集）。
      每个交互对需要足够的样本支撑，在 L1 正则化下，
      k = N_actual // 3 是"平均每3个训练样本支撑1个交互对"的经验规则，
      这与机器学习中特征选择的经验比例（样本数/特征数 ≥ 3）一致。
      上界 15：超过 15 个交互对后噪声引入风险急剧增加。
      下界 3：少于 3 个交互对时模块统计意义极低。

    各系统最小样本量对应值：
      x264   N_min=16  N_actual=11  k=3
      BDBJ   N_min=26  N_actual=18  k=6
      lrzip  N_min=19  N_actual=13  k=4
      polly  N_min=39  N_actual=26  k=8
      vp9    N_min=41  N_actual=28  k=9
      Dune   N_min=49  N_actual=33  k=11
      hipacc N_min=261 N_actual=174 k=15 (capped)
      hsmgp  N_min=77  N_actual=52  k=15 (capped)
      javagc N_min=855 N_actual=570 k=15 (capped)
      sac    N_min=2060 N_actual=1373 k=15 (capped)
    """
    N_actual = int(np.ceil(N_train_min * 2 / 3))
    return int(np.clip(N_actual // 3, 3, 15))


# ======================================================================
# 数据预处理类
# ======================================================================

class DataPreproc:

    def __init__(self, sys_name, top_k_interactions=None):
        self.sys_name = sys_name
        self.data_dir = 'datasets/' + sys_name + '_AllNumeric.csv'
        self.__read_whole_data()

        # [新增] top_k 自适应计算
        if top_k_interactions is None:
            N_train_all = system_samplesize(sys_name)
            self.top_k = compute_top_k(int(np.min(N_train_all)))
            print(f'  Auto top_k={self.top_k} '
                  f'(N_min={int(np.min(N_train_all))}, '
                  f'N_actual={int(np.ceil(np.min(N_train_all)*2/3))})')
        else:
            self.top_k = top_k_interactions

        # [新增] 交互对筛选
        self.interaction_pairs = self.find_interactions_from_data(
            self.X_all, self.Y_all, verbose=True
        )

    # ------------------------------------------------------------------
    # 数据读取（与 HINNPerf 原版相同）
    # ------------------------------------------------------------------

    def __read_whole_data(self):
        print('Read whole dataset ' + self.sys_name + ' from csv file ...')
        self.whole_data = genfromtxt(self.data_dir, delimiter=',', skip_header=1)
        (self.all_sample_num, config_num) = self.whole_data.shape
        self.config_num = config_num - 1
        self.X_all = self.whole_data[:, 0:self.config_num]
        self.Y_all = self.whole_data[:, self.config_num][:, np.newaxis]

    # ------------------------------------------------------------------
    # 归一化
    # ------------------------------------------------------------------

    def __normalize(self, X, Y):
        
        max_X = np.amax(X, axis=0)
        if 0 in max_X:
            max_X[max_X == 0] = 1
        X_sample = np.divide(X, max_X)

        max_Y = np.max(Y)/100        
        if max_Y == 0:
            max_Y = 1
        Y_sample = np.divide(Y, max_Y)

        return X_sample, Y_sample, max_X, np.array([max_Y])

    def __normalize_gaussian(self, X, Y):
        """与 HINNPerf 原版完全相同"""
        mean_X = np.mean(X, axis=0)
        std_X = np.std(X, axis=0)
        X_sample = (X - mean_X) / (std_X + (std_X == 0) * .001)
        mean_Y = np.mean(Y, axis=0)
        std_Y = np.std(Y, axis=0)
        Y_sample = (Y - mean_Y) / (std_Y + (std_Y == 0) * .001)
        return X_sample, Y_sample, mean_X, std_X, mean_Y, std_Y

    # ------------------------------------------------------------------
    # [新增] 交互特征筛选
    # ------------------------------------------------------------------

    def find_interactions_from_data(self, X, Y, verbose=False):
        
        D = self.config_num
        if D < 2 or self.top_k <= 0:
            return []

        if verbose:
            print(f'  Selecting top-{self.top_k} interaction pairs...')

        Y_flat = Y.ravel()

        # Step 1: 决策树预筛候选特征
        try:
            tree = DecisionTreeRegressor(max_depth=5, random_state=42)
            tree.fit(X, Y_flat)
            feature_importance = tree.feature_importances_
        except Exception:
            feature_importance = np.ones(D) / D

        n_candidates = max(4, D // 2)
        important_feats = np.argsort(feature_importance)[-n_candidates:]

        if verbose:
            print(f'    Candidate features: {len(important_feats)}/{D}')

        # Step 2: 计算候选对的三项原始评分
        pairs_data = []
        for idx_i, i in enumerate(important_feats):
            for j in important_feats[idx_i + 1:]:
                interact_feat = X[:, i] * X[:, j]
                if np.var(interact_feat) < 1e-8:
                    continue

                try:
                    mi = mutual_info_regression(
                        interact_feat.reshape(-1, 1), Y_flat,
                        random_state=42)[0]
                except Exception:
                    mi = 0.0

                try:
                    r = np.corrcoef(interact_feat, Y_flat)[0, 1]
                    pearson = 0.0 if np.isnan(r) else abs(r)
                except Exception:
                    pearson = 0.0

                dt = (feature_importance[i] + feature_importance[j]) / 2.0

                if not any(np.isnan(v) for v in [mi, pearson, dt]):
                    pairs_data.append((int(i), int(j), mi, pearson, dt))

        if not pairs_data:
            if verbose:
                print('    Warning: No valid interaction pairs found')
            return []

        # Step 3: 各指标 min-max 归一化到 [0,1]
        def minmax(arr):
            span = arr.max() - arr.min()
            if span < 1e-10:
                return np.full_like(arr, 0.5)
            return (arr - arr.min()) / span

        mi_n      = minmax(np.array([p[2] for p in pairs_data]))
        pearson_n = minmax(np.array([p[3] for p in pairs_data]))
        dt_n      = minmax(np.array([p[4] for p in pairs_data]))

        # Step 4: 均等权重加权
        combined = (mi_n + pearson_n + dt_n) / 3.0

        # Step 5: 排序取 top-k
        sorted_idx = np.argsort(combined)[::-1]
        top_pairs  = [(pairs_data[i][0], pairs_data[i][1])
                      for i in sorted_idx[:self.top_k]]

        if verbose:
            print(f'    Selected {len(top_pairs)} pairs '
                  f'(from {len(pairs_data)} candidates):')
            for rank, i in enumerate(sorted_idx[:min(5, len(sorted_idx))]):
                fi, fj = pairs_data[i][0], pairs_data[i][1]
                print(f'      ({fi:2d},{fj:2d}) combined={combined[i]:.4f} '
                      f'[MI_n={mi_n[i]:.3f} Pear_n={pearson_n[i]:.3f} '
                      f'DT_n={dt_n[i]:.3f}]')

        return top_pairs

    # ------------------------------------------------------------------
    # 数据集划分（与 HINNPerf 原版接口和逻辑完全相同）
    # ------------------------------------------------------------------

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
            X_train_norm, Y_train_norm, mean_X, std_X, mean_Y, std_Y = \
                self.__normalize_gaussian(X_train, Y_train)
            X_valid_norm = (X_valid - mean_X) / (std_X + (std_X == 0) * .001)
            Y_valid_norm = (Y_valid - mean_Y) / (std_Y + (std_Y == 0) * .001)
            return X_train_norm, Y_train_norm, X_valid_norm, Y_valid_norm, mean_Y, std_Y
        else:
            X_train_norm, Y_train_norm, max_X, max_Y = \
                self.__normalize(X_train, Y_train)
            X_valid_norm = np.divide(X_valid, max_X)
            Y_valid_norm = np.divide(Y_valid, max_Y)
            return X_train_norm, Y_train_norm, X_valid_norm, Y_valid_norm, max_Y

    def get_train_test_samples(self, sample_size, seed, gnorm=False):
        np.random.seed(seed)
        permutation = np.random.permutation(self.all_sample_num)
        training_index = permutation[0:sample_size]
        X_train = self.X_all[training_index, :]
        Y_train = self.Y_all[training_index, :]
        testing_index = np.setdiff1d(
            np.array(range(self.all_sample_num)), training_index)
        X_test = self.X_all[testing_index, :]
        Y_test = self.Y_all[testing_index, :]

        if gnorm:
            X_train_norm, Y_train_norm, mean_X, std_X, mean_Y, std_Y = \
                self.__normalize_gaussian(X_train, Y_train)
            X_test_norm = (X_test - mean_X) / (std_X + (std_X == 0) * .001)
            return X_train_norm, Y_train_norm, X_test_norm, Y_test, mean_Y, std_Y
        else:
            X_train_norm, Y_train_norm, max_X, max_Y = \
                self.__normalize(X_train, Y_train)
            X_test_norm = np.divide(X_test, max_X)
            return X_train_norm, Y_train_norm, X_test_norm, Y_test, max_Y
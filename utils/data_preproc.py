#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HINNPerf Improved - Data Preprocessing (FIXED)
===============================================

修复的核心问题:
1. ✅ Y标签始终保持2D维度（与原版一致）
2. ✅ get_train_test_samples中Y_test不归一化（与原版一致）
3. ✅ 优化交互筛选逻辑（鲁棒性提升）
"""

import numpy as np
from numpy import genfromtxt
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import mutual_info_regression


def system_samplesize(sys_name):
    """定义不同数据集的训练样本大小（与原版一致）"""
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
        raise AssertionError(
            f"Unexpected system name: '{sys_name}'. "
            f"Supported: x264, BDBJ, lrzip, polly, vp9, Dune, hipacc, hsmgp, javagc, sac"
        )
    
    return N_train_all


def seed_generator(sys_name, sample_size):
    """生成每个样本大小的初始种子（与原版一致）"""
    N_train_all = system_samplesize(sys_name)
    
    if sample_size in N_train_all:
        seed_o = np.where(N_train_all == sample_size)[0][0]
    else:
        seed_o = np.random.randint(1, 101)
    
    return seed_o


class DataPreproc:
    """
    改进的数据预处理类（已修复）
    
    核心改进:
    1. 数据驱动的交互筛选（互信息 + Pearson + 决策树）
    2. 保持与原版一致的数据维度处理
    3. 鲁棒的特征工程
    """
    
    def __init__(self, sys_name, top_k_interactions=15):
        """
        初始化数据预处理器
        
        Args:
            sys_name: 系统名称
            top_k_interactions: 保留的交互对数量（默认15）
        """
        self.sys_name = sys_name
        self.data_dir = 'datasets/' + sys_name + '_AllNumeric.csv'
        self.top_k = top_k_interactions
        
        # 读取全量数据
        self.__read_whole_data()
        
        # 筛选重要交互对
        self.interaction_pairs = self.__find_important_interactions()
    
    def __read_whole_data(self):
        """
        从CSV文件读取数据（与原版一致）
        """
        print(f'Reading dataset: {self.sys_name}')
        
        try:
            self.whole_data = genfromtxt(self.data_dir, delimiter=',', skip_header=1)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Dataset not found: {self.data_dir}\n"
                f"Please place the CSV file in the datasets/ directory"
            )
        except Exception as e:
            raise Exception(f"Error reading dataset: {e}")
        
        (self.all_sample_num, config_num) = self.whole_data.shape
        self.config_num = config_num - 1
        
        # ✅ 修复：Y标签保持2D（与原版一致）
        self.X_all = self.whole_data[:, 0:self.config_num]
        self.Y_all = self.whole_data[:, self.config_num][:, np.newaxis]  # 保持2D
        
        print(f'  Samples: {self.all_sample_num}, Features: {self.config_num}')
    
    def __find_important_interactions(self):
        """
        数据驱动的交互对筛选（改进版）
        
        策略:
        1. 互信息（主指标）：捕获非线性关系
        2. 决策树：全局特征重要性
        3. Pearson相关性（辅助）：线性关系
        4. 综合评分 -> Top-K选择
        """
        D = self.config_num
        
        if D < 2 or self.top_k <= 0:
            print('  No interaction pairs (D<2 or top_k=0)')
            return []
        
        print(f'  Selecting top-{self.top_k} interaction pairs...')
        
        # ✅ 修复：Y_all.ravel()确保1D（sklearn要求）
        Y_flat = self.Y_all.ravel()
        
        # 1. 决策树特征重要性
        try:
            tree = DecisionTreeRegressor(max_depth=5, random_state=42)
            tree.fit(self.X_all, Y_flat)
            feature_importance = tree.feature_importances_
        except Exception as e:
            print(f'    Warning: Decision tree failed ({e})')
            feature_importance = np.ones(D) / D
        
        # 2. 选择重要特征
        importance_threshold = np.median(feature_importance)
        important_feats = np.where(feature_importance > importance_threshold)[0]
        
        if len(important_feats) < 2:
            top_n = min(10, D)
            important_feats = np.argsort(feature_importance)[-top_n:]
        
        print(f'    Important features: {len(important_feats)}/{D}')
        
        # 3. 计算交互强度
        pairs = []
        
        for i in important_feats:
            for j in important_feats:
                if i < j:
                    # 构造交互特征
                    interact_feat = self.X_all[:, i] * self.X_all[:, j]
                    
                    # ✅ 关键：检查常量特征
                    feat_var = np.var(interact_feat)
                    if feat_var < 1e-8:
                        continue
                    
                    # 指标1: 互信息
                    try:
                        mi_score = mutual_info_regression(
                            interact_feat.reshape(-1, 1),
                            Y_flat,
                            random_state=42
                        )[0]
                    except Exception:
                        mi_score = 0.0
                    
                    # 指标2: Pearson相关性
                    try:
                        pearson_corr = np.abs(np.corrcoef(
                            interact_feat, 
                            Y_flat
                        )[0, 1])
                        
                        if np.isnan(pearson_corr):
                            pearson_corr = 0.0
                    except Exception:
                        pearson_corr = 0.0
                    
                    # 指标3: 决策树特征重要性
                    dt_score = (feature_importance[i] + feature_importance[j]) / 2
                    
                    # 综合评分
                    w_mi, w_pearson, w_dt = 0.5, 0.3, 0.2
                    combined_score = (
                        w_mi * mi_score + 
                        w_pearson * pearson_corr + 
                        w_dt * dt_score
                    )
                    
                    if not np.isnan(combined_score):
                        pairs.append((i, j, combined_score, mi_score, pearson_corr, dt_score))
        
        # 4. Top-K选择
        if len(pairs) == 0:
            print('    Warning: No valid interaction pairs found')
            return []
        
        pairs_sorted = sorted(pairs, key=lambda x: x[2], reverse=True)
        top_pairs = [(p[0], p[1]) for p in pairs_sorted[:self.top_k]]
        
        # 输出详细信息
        print(f'    Selected {len(top_pairs)} interaction pairs:')
        for idx, (i, j, score, mi, pearson, dt) in enumerate(pairs_sorted[:min(5, len(pairs_sorted))]):
            print(f'      ({i:2d}, {j:2d}): combined={score:.4f} '
                  f'(MI={mi:.3f}, Pearson={pearson:.3f}, DT={dt:.3f})')
        if len(top_pairs) > 5:
            print(f'      ... and {len(top_pairs)-5} more')
        
        return top_pairs
    
    def __normalize(self, X, Y):
        """归一化: 除以最大值（与原版一致）"""
        max_X = np.amax(X, axis=0)
        max_X[max_X == 0] = 1
        X_norm = np.divide(X, max_X)
        
        max_Y = np.max(Y) / 100
        if max_Y == 0:
            max_Y = 1
        Y_norm = np.divide(Y, max_Y)
        
        return X_norm, Y_norm, max_X, np.array([max_Y])
    
    def __normalize_gaussian(self, X, Y):
        """Gaussian归一化（与原版一致）"""
        mean_X = np.mean(X, axis=0)
        std_X = np.std(X, axis=0)
        X_norm = (X - mean_X) / (std_X + (std_X == 0) * 1e-3)
        
        mean_Y = np.mean(Y, axis=0)
        std_Y = np.std(Y, axis=0)
        Y_norm = (Y - mean_Y) / (std_Y + (std_Y == 0) * 1e-3)
        
        return X_norm, Y_norm, mean_X, std_X, mean_Y, std_Y
    
    def get_train_valid_samples(self, sample_size, seed, gnorm):
        """
        生成训练集和验证集（与原版一致）
        
        分割比例: 2/3训练，1/3验证
        """
        np.random.seed(seed)
        permutation = np.random.permutation(self.all_sample_num)
        training_index = permutation[0:sample_size]
        
        X_sample = self.X_all[training_index, :]
        Y_sample = self.Y_all[training_index, :]  # ✅ 保持2D
        
        # 2/3训练，1/3验证
        sample_cross = int(np.ceil(sample_size * 2 / 3))
        X_train = X_sample[0:sample_cross, :]
        Y_train = Y_sample[0:sample_cross, :]  # ✅ 保持2D
        X_valid = X_sample[sample_cross:sample_size, :]
        Y_valid = Y_sample[sample_cross:sample_size, :]  # ✅ 保持2D
        
        if gnorm:
            # Gaussian归一化
            X_train_norm, Y_train_norm, mean_X, std_X, mean_Y, std_Y = \
                self.__normalize_gaussian(X_train, Y_train)
            
            # 验证集使用训练集的统计量
            X_valid_norm = (X_valid - mean_X) / (std_X + (std_X == 0) * 1e-3)
            Y_valid_norm = (Y_valid - mean_Y) / (std_Y + (std_Y == 0) * 1e-3)
            
            return (X_train_norm, Y_train_norm,
                    X_valid_norm, Y_valid_norm,
                    mean_Y, std_Y)
        else:
            # 最大值归一化
            X_train_norm, Y_train_norm, max_X, max_Y = \
                self.__normalize(X_train, Y_train)
            
            X_valid_norm = np.divide(X_valid, max_X)
            Y_valid_norm = np.divide(Y_valid, max_Y)
            
            return (X_train_norm, Y_train_norm,
                    X_valid_norm, Y_valid_norm,
                    max_Y)
    
    def get_train_test_samples(self, sample_size, seed, gnorm=False):
        """
        生成训练集和测试集（与原版一致）
        
        ✅ 关键：测试集标签(Y_test)不归一化（保持原始值）
        """
        np.random.seed(seed)
        permutation = np.random.permutation(self.all_sample_num)
        training_index = permutation[0:sample_size]
        
        X_train = self.X_all[training_index, :]
        Y_train = self.Y_all[training_index, :]  # ✅ 保持2D
        
        testing_index = np.setdiff1d(
            np.array(range(self.all_sample_num)),
            training_index
        )
        X_test = self.X_all[testing_index, :]
        Y_test = self.Y_all[testing_index, :]  # ✅ 保持2D，且不归一化
        
        if gnorm:
            # Gaussian归一化（仅训练集）
            X_train_norm, Y_train_norm, mean_X, std_X, mean_Y, std_Y = \
                self.__normalize_gaussian(X_train, Y_train)
            
            # ✅ 关键：测试集特征归一化，但标签保持原始值
            X_test_norm = (X_test - mean_X) / (std_X + (std_X == 0) * 1e-3)
            
            return (X_train_norm, Y_train_norm,
                    X_test_norm, Y_test,  # Y_test保持原始值
                    mean_Y, std_Y)
        else:
            # 最大值归一化（仅训练集）
            X_train_norm, Y_train_norm, max_X, max_Y = \
                self.__normalize(X_train, Y_train)
            
            # ✅ 关键：测试集特征归一化，但标签保持原始值
            X_test_norm = np.divide(X_test, max_X)
            
            return (X_train_norm, Y_train_norm,
                    X_test_norm, Y_test,  # Y_test保持原始值
                    max_Y)
    
    def get_interaction_info(self):
        """获取交互对信息（用于可视化和分析）"""
        return {
            'system': self.sys_name,
            'num_features': self.config_num,
            'num_samples': self.all_sample_num,
            'top_k': self.top_k,
            'interaction_pairs': self.interaction_pairs,
            'num_interactions': len(self.interaction_pairs)
        }
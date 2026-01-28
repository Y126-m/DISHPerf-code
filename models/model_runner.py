#!/usr/bin/env python
# -*- coding: utf-8 -*-

from utils.data_preproc import system_samplesize, seed_generator
import numpy as np
import time
import tensorflow as tf


class ImprovedModelRunner:
    """改进的模型训练器"""
    
    def __init__(self, data_preproc, model_class):
        self.data_preproc = data_preproc
        self.model_class = model_class
        self.feature_importance_results = []
    
    def train(self, config, sample_size, number_experiment, cur_exp):
        """训练函数"""
        N_train = sample_size
        
        if N_train >= self.data_preproc.all_sample_num:
            raise AssertionError("Sample size can't be larger than whole dataset")
        
        # 1. 数据准备
        seed_init = seed_generator(self.data_preproc.sys_name, N_train)
        seed = seed_init * number_experiment + cur_exp
        use_gnorm = config.get('gnorm', False)
        
        if use_gnorm:
            X_train, Y_train, X_valid, Y_valid, mean_Y, std_Y = \
                self.data_preproc.get_train_valid_samples(N_train, seed, gnorm=True)
        else:
            X_train, Y_train, X_valid, Y_valid, max_Y = \
                self.data_preproc.get_train_valid_samples(N_train, seed, gnorm=False)
        
        # 2. 模型初始化 
        model = self.model_class(config)
        model.build_train()
        
        # 3. 训练配置
        lr_init = config['lr'] if isinstance(config['lr'], (int, float)) else config['lr'][0]
        lr = lr_init
        decay = lr_init / 1000
        
      
        n_features = X_train.shape[-1] if X_train.size > 0 else 100
        
        
        base_patience = 100 * (N_train/100)**0.5 * (n_features/50)**0.3
        patience = max(100, min(300, int(base_patience)))
        
        
        min_epochs = max(300, min(600, int(100 * (N_train/100)**0.7)))
        
       
        val_freq = max(10, min(30, patience // 3))
        
        improvement_threshold = 0.005
        best_val_loss = float('inf')
        wait = 0
        best_weights = None
        loss_history = []
        
        # 5. 训练循环
        for epoch in range(1, 2001):
            _, cur_loss = model.sess.run(
                [model.train_op, model.loss],
                {model.X: X_train, model.Y: Y_train, model.lr: lr}
            )
            
            lr = lr_init * 1.0 / (1 + decay * epoch)
            
           
            if epoch % val_freq == 0:
                Y_pred_val = model.sess.run(model.output, {model.X: X_valid})
                
                # 还原到原始尺度
                if use_gnorm:
                    Y_pred_val_orig = Y_pred_val * (std_Y + 1e-6) + mean_Y
                    Y_valid_orig = Y_valid * (std_Y + 1e-6) + mean_Y
                else:
                    Y_pred_val_orig = Y_pred_val * max_Y
                    Y_valid_orig = Y_valid * max_Y
                
                val_mae = np.mean(np.abs(Y_pred_val_orig - Y_valid_orig))
                loss_history.append(val_mae)
                
                improvement = (best_val_loss - val_mae) / (best_val_loss + 1e-6)
                
               
                trend_ok = True
                if len(loss_history) >= 5:
                    recent_trend = np.polyfit(range(len(loss_history[-5:])), loss_history[-5:], 1)[0]
                    if recent_trend > 0 and epoch > min_epochs:
                        trend_ok = False
                
              
                if val_mae < best_val_loss * (1 - improvement_threshold):
                    best_val_loss = val_mae
                    wait = 0
                    
                    best_weights = {}
                    for v in tf.trainable_variables():
                        best_weights[v.name] = model.sess.run(v)
                else:
                    wait += val_freq
                    
                    if wait >= patience and epoch > min_epochs and not trend_ok:
                       
                        if best_weights is not None:
                            for v in tf.trainable_variables():
                                if v.name in best_weights:
                                    model.sess.run(v.assign(best_weights[v.name]))
                        
                        if config.get('verbose', False):
                            print(f"      Early stopping at epoch {epoch}")
                        break
        
        # 7. 最终评估
        Y_pred_train = model.sess.run(model.output, {model.X: X_train})
        if use_gnorm:
            Y_pred_train_orig = Y_pred_train * (std_Y + 1e-6) + mean_Y
            Y_train_orig = Y_train * (std_Y + 1e-6) + mean_Y
        else:
            Y_pred_train_orig = Y_pred_train * max_Y
            Y_train_orig = Y_train * max_Y
        abs_error_train = np.mean(np.abs(Y_pred_train_orig - Y_train_orig))
        
        Y_pred_val = model.sess.run(model.output, {model.X: X_valid})
        if use_gnorm:
            Y_pred_val_orig = Y_pred_val * (std_Y + 1e-6) + mean_Y
            Y_valid_orig = Y_valid * (std_Y + 1e-6) + mean_Y
        else:
            Y_pred_val_orig = Y_pred_val * max_Y
            Y_valid_orig = Y_valid * max_Y
        abs_error_val = np.mean(np.abs(Y_pred_val_orig - Y_valid_orig))
        
        model.finalize()
        return abs_error_train, abs_error_val
    
    def test(self, best_config, sample_size, number_experiment, cur_exp):
        N_train = sample_size
        
        # 1. 数据准备
        seed_init = seed_generator(self.data_preproc.sys_name, N_train)
        seed = seed_init * number_experiment + cur_exp
        use_gnorm = best_config.get('gnorm', False)
        
        if use_gnorm:
            X_train, Y_train, X_test, Y_test, mean_Y, std_Y = \
                self.data_preproc.get_train_test_samples(N_train, seed, gnorm=True)
        else:
            X_train, Y_train, X_test, Y_test, max_Y = \
                self.data_preproc.get_train_test_samples(N_train, seed, gnorm=False)
        
        model = self.model_class(best_config)
        model.build_train()
        
        
        lr_init = best_config['lr'] if isinstance(best_config['lr'], (int, float)) else best_config['lr'][0]
        lr = lr_init
        decay = lr_init / 1000
        
        
        patience = 800
        val_freq = 20
        min_epochs = 300
        
        best_train_loss = float('inf')
        wait = 0
        
        for epoch in range(1, 2001):
            _, cur_loss = model.sess.run(
                [model.train_op, model.loss],
                {model.X: X_train, model.Y: Y_train, model.lr: lr}
            )
            
            lr = lr_init * 1.0 / (1 + decay * epoch)
            
           
            if epoch % val_freq == 0:
                if cur_loss < best_train_loss * 0.995: 
                    best_train_loss = cur_loss
                    wait = 0
                else:
                    wait += val_freq
                    if wait >= patience and epoch > min_epochs:
                        if best_config.get('verbose', False):
                            print(f"      Test training stopped at epoch {epoch}")
                        break
        
        # 4. 测试集预测
        Y_pred_test = model.sess.run(model.output, {model.X: X_test})
        
        # 还原到原始尺度
        if use_gnorm:
            Y_pred_test_orig = Y_pred_test * (std_Y + 1e-6) + mean_Y
            Y_test_orig = Y_test  
            Y_pred_test_orig = Y_pred_test * max_Y
            Y_test_orig = Y_test 
        # 计算相对误差
        denom = Y_test_orig.ravel()
        rel_error = np.mean(
            np.abs(np.divide(denom - Y_pred_test_orig.ravel(), denom + 1e-6))
        ) * 100
        
        model.finalize()
        return rel_error



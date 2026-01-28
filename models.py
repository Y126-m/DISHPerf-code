#!/usr/bin/env python
# -*- coding: utf-8 -*-


import tensorflow as tf
import numpy as np


class DISHPerf:
    
    def __init__(self, config):
       
        # 统一参数提取
        self.input_dim = self._extract_param(config['input_dim'])
        self.num_neuron = self._extract_param(config['num_neuron'])
        self.num_block = self._extract_param(config['num_block'])
        self.num_layer_pb = self._extract_param(config['num_layer_pb'])
        self.lamda = float(self._extract_param(config['lamda']))
        self.use_linear = self._extract_param(config.get('linear', True))
        self.verbose = self._extract_param(config.get('verbose', False))
        
        # ：从config中获取交互对（由data_preproc预筛选）
        self.interaction_pairs = config.get('interaction_pairs', [])
        self.use_interactions = config.get('use_interactions', False)
        
        self.name = 'DISHPerf'
        
        tf.reset_default_graph()
        tf.set_random_seed(1)
    
    @staticmethod
    def _extract_param(param):
        """统一处理参数格式（支持列表或单值）"""
        if isinstance(param, (list, tuple)) and len(param) > 0:
            return param[0]
        return param
    
    def __build_neural_net(self):
        """
        构建神经网络架构（与原版保持一致）
        """
        input_layer = self.X
        output = None
        
        for block_id in range(self.num_block):
            backcast, forecast = self.__create_block(input_layer, block_id)
            
            # 拼接backcast到
            input_layer = tf.concat([self.X, backcast], 1)
            
            if block_id == 0:
                output = forecast
            else:
                output = output + forecast
        
        # 全局线性层（可选）
        if self.use_linear:
            linear_input = self.X
            linear_output = tf.layers.dense(
                linear_input, 1,
                kernel_regularizer=tf.contrib.layers.l2_regularizer(float(self.lamda))
            )
            output = output + linear_output
        
        return output
    
    def __create_block(self, x, block_id):
        # 1. 主干隐式流（与原版一致）
        layer = x
        for i in range(self.num_layer_pb):
            if i == 0:
                # 第一层：L1正则化
                layer = tf.layers.dense(
                    layer, self.num_neuron, tf.nn.relu,
                    kernel_initializer=tf.contrib.layers.xavier_initializer(seed=1),
                    kernel_regularizer=tf.contrib.layers.l1_regularizer(float(self.lamda))
                )
            else:
                # 其他层：无正则化
                layer = tf.layers.dense(
                    layer, self.num_neuron, tf.nn.relu,
                    kernel_initializer=tf.contrib.layers.xavier_initializer(seed=1)
                )
        
        # 2. 显式交互流（仅在第一个block且配置启用时）
        if block_id == 0 and self.use_interactions and len(self.interaction_pairs) > 0:
            # 2.1 构造交互特征
            interact_feats = []
            for (i, j) in self.interaction_pairs:
                
                if i < self.input_dim and j < self.input_dim:
                    interact = tf.multiply(
                        self.X[:, i:i+1], 
                        self.X[:, j:j+1]
                    )
                    interact_feats.append(interact)
            
            if interact_feats:
                # 2.2 拼接所有交互特征
                interactions = tf.concat(interact_feats, axis=1)
                
                # 2.3 交互特征的独立处理
                interact_layer = tf.layers.dense(
                    interactions, self.num_neuron, tf.nn.relu,
                    kernel_initializer=tf.contrib.layers.xavier_initializer(seed=1),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(float(self.lamda) * 0.1)
                )
                
                fused_layer = tf.layers.dense(
                    tf.concat([layer, interact_layer], axis=1),
                    self.num_neuron,  # 输出维度与layer一致
                    tf.nn.relu,
                    kernel_initializer=tf.contrib.layers.xavier_initializer(seed=1)
                )
                layer = fused_layer
        
        
        backcast = tf.layers.dense(layer, self.input_dim, tf.nn.relu)
        forecast = tf.layers.dense(layer, 1)
        
        return backcast, forecast
    
    def build_train(self):
        """构建训练图"""
        self.__add_placeholders()
        self.__add_prediction()
        self.__add_loss()
        self.__add_optimizer()
        self.init_session()
    
    def __add_placeholders(self):
        """占位符"""
        self.X = tf.placeholder(tf.float32, [None, self.input_dim], name='X')
        self.Y = tf.placeholder(tf.float32, [None, 1], name='Y')
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
    
    def __add_prediction(self):
        """预测"""
        self.output = self.__build_neural_net()
    
    def __add_loss(self):
        """损失函数"""
        mse_loss = tf.losses.mean_squared_error(self.Y, self.output)
        reg_loss = tf.losses.get_regularization_loss()
        self.loss = mse_loss + reg_loss
    
    def __add_optimizer(self):
        """优化器"""
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        
        with tf.control_dependencies(update_ops):
            grads_and_vars = optimizer.compute_gradients(self.loss)
            
            # 梯度裁剪（防止梯度爆炸）
            capped_gvs = [
                (tf.clip_by_norm(grad, 1.0), var) 
                for grad, var in grads_and_vars 
                if grad is not None
            ]
            
            self.train_op = optimizer.apply_gradients(capped_gvs)
    
    def init_session(self):
        """初始化会话"""
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
    
    def finalize(self):
        """清理资源"""
        self.sess.close()
        tf.get_default_graph().finalize()



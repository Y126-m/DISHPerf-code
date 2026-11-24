#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Paper-aligned TF1 model: full-input fixed-width connections and gated products."""

import tensorflow as tf
import numpy as np


class DISHPerf:

    def __init__(self, config):
        def _get(key, default=None):
            v = config.get(key, default)
            if isinstance(v, (list, tuple)) and len(v) > 0:
                return v[0]
            return v

        self.input_dim    = _get('input_dim')
        self.num_neuron   = _get('num_neuron')
        self.num_block    = _get('num_block')
        self.num_layer_pb = _get('num_layer_pb')
        self.lamda        = float(_get('lamda'))
        self.use_linear   = _get('linear', False)
        self.verbose      = _get('verbose', False)

        # 结构先验参数
        self.interaction_pairs = config.get('interaction_pairs', [])
        self.critical_features = config.get('critical_features', []) # [新增] 核心保留特征
        self.use_interactions  = config.get('use_interactions', False)

        self.name = 'DISHPerf'

        tf.reset_default_graph()
        tf.set_random_seed(1)

    # ------------------------------------------------------------------
    # 网络构建
    # ------------------------------------------------------------------

    def __build_neural_net(self):
        """
        Build hierarchical blocks with full-input fixed-width connections.
        """
        input_layer = self.X
        output = None

        # Eq. concat: all original configuration inputs are preserved.
        X_residual = self.X
        self.block_input_widths = []

        for block_id in range(self.num_block):
            self.block_input_widths.append(int(input_layer.shape.as_list()[1]))
            backcast, forecast = self.__create_block(input_layer, block_id)
            
            # Fixed-width full-input connection, Eq. concat.
            input_layer = tf.concat([X_residual, backcast], axis=1)
            
            output = forecast if block_id == 0 else output + forecast

        if self.use_linear:
            linear_out = tf.layers.dense(
                self.X, 1,
                kernel_regularizer=tf.contrib.layers.l2_regularizer(self.lamda))
            output = output + linear_out

        return output

    def __create_block(self, x, block_id):
        """
        构建单个 Block（含自适应门控融合层）
        """
        layer = x
        for i in range(self.num_layer_pb):
            if i == 0:
                layer = tf.layers.dense(
                    layer, self.num_neuron, tf.nn.relu,
                    kernel_initializer=tf.contrib.layers.xavier_initializer(seed=1),
                    kernel_regularizer=tf.contrib.layers.l1_regularizer(self.lamda))
            else:
                layer = tf.layers.dense(
                    layer, self.num_neuron, tf.nn.relu,
                    kernel_initializer=tf.contrib.layers.xavier_initializer(seed=1))

        # ---- 显式路径与门控融合（仅 Block 0）----
        if block_id == 0 and self.use_interactions and len(self.interaction_pairs) > 0:

            interact_feats = []
            for (i, j) in self.interaction_pairs:
                if i < self.input_dim and j < self.input_dim:
                    interact_feats.append(
                        tf.multiply(self.X[:, i:i+1], self.X[:, j:j+1]))

            if interact_feats:
                interactions = tf.concat(interact_feats, axis=1)

                bottleneck_dim = max(16, min(self.num_neuron // 4,
                                             len(interact_feats) * 2))

                interact_bottleneck = tf.layers.dense(
                    interactions, bottleneck_dim, tf.nn.relu,
                    kernel_initializer=tf.contrib.layers.xavier_initializer(seed=1),
                    kernel_regularizer=tf.contrib.layers.l1_regularizer(self.lamda))

                interact_up = tf.layers.dense(
                    interact_bottleneck, self.num_neuron, activation=None,
                    kernel_initializer=tf.contrib.layers.xavier_initializer(seed=1))

                # [核心改进 1] 自适应门控融合层 (Adaptive Gated Fusion)
                # 替代原有粗暴的 layer = tf.nn.relu(layer + interact_up)
                
                # 1. 拼接隐式特征和显式表示
                concat_features = tf.concat([layer, interact_up], axis=1)
                
                # 2. 学习门控权重 g ∈ (0, 1)
                gate = tf.layers.dense(
                    concat_features, self.num_neuron, activation=tf.nn.sigmoid,
                    kernel_initializer=tf.contrib.layers.xavier_initializer(seed=1),
                    name="gate_fusion"
                )
                
                # Elementwise gated fusion, Eq. fusion_layer.
                fused_representation = gate * layer + (1.0 - gate) * interact_up
                
                # 4. 激活输出
                layer = tf.nn.relu(fused_representation)

        # ---- 双头输出 ----
        backcast = tf.layers.dense(layer, self.input_dim, tf.nn.relu)
        forecast = tf.layers.dense(layer, 1, activation=None)

        return backcast, forecast

    def build_train(self):
        self.__add_placeholders_op()
        self.__add_pred_op()
        self.__add_loss_op()
        self.__add_train_op()
        self.init_session()

    def __add_placeholders_op(self):
        self.X  = tf.placeholder("float", [None, self.input_dim])
        self.Y  = tf.placeholder("float", [None, 1])
        self.lr = tf.placeholder("float")

    def __add_pred_op(self):
        self.output = self.__build_neural_net()

    def __add_loss_op(self):
        l2_loss = tf.losses.get_regularization_loss()
        self.loss = l2_loss + tf.losses.mean_squared_error(self.Y, self.output)

    def __add_train_op(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            grads_and_vars = optimizer.compute_gradients(self.loss)
            capped = [(tf.clip_by_norm(g, 1.0), v)
                      for g, v in grads_and_vars if g is not None]
            self.train_op = optimizer.apply_gradients(capped)

    def init_session(self):
        cfg = tf.ConfigProto()
        cfg.intra_op_parallelism_threads = 1
        cfg.inter_op_parallelism_threads = 1
        cfg.gpu_options.allow_growth = True
        self.sess = tf.Session(config=cfg)
        self.sess.run(tf.global_variables_initializer())

    def finalize(self):
        self.sess.close()
        tf.get_default_graph().finalize()

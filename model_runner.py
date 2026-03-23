#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from utils.data_preproc import seed_generator


class ModelRunner:
    """
    DISHPerf 训练与测试执行器。

    与 HINNPerf 原版 ModelRunner 的结构差异：
      HINNPerf：train() 固定跑 2000 epoch，返回训练/验证 MAE；
               run.py 中循环所有配置调用 train()，取最小验证 MAE 的配置。
      DISHPerf：search_best_config() 合并了超参搜索逻辑，内含早停机制；
               test() 新增最优权重快照与恢复。

    关于早停与否的选择：
      HINNPerf 不用早停，固定 2000 epoch，因为：
        (a) 样本量小，2000 epoch 绝对时间可接受
        (b) 不存在截断误差
      DISHPerf 加早停，理由：
        搜索空间 120 个配置，若每个都跑满 2000 epoch，
        总时间是 HINNPerf 的数倍（HINNPerf 的超参空间是 4×3×5×2=120，
        与 DISHPerf 相同，但 HINNPerf 加了 use_interactions 维度后会更大）。
        早停让坏配置自动提前终止，好配置获得充分训练，
        不需要两阶段搜索——早停本身已经实现了"差的少训练、好的多训练"的目标。

    早停参数的设置依据（参考 Prechelt 1998）：
      学习率衰减：lr(t) = lr_0 / (1 + decay*t)，decay = lr_0/1000 = 1e-6
      在 epoch=2000 时：lr(2000) = 0.001/(1+0.002) ≈ 0.000998，几乎不变。
      这意味着训练是近似常数学习率的全批量梯度下降，损失应单调下降直至收敛，
      而非像随机梯度下降那样持续震荡。
      因此早停的 patience 不需要很大，主要作用是检测真正的收敛平台期。

      patience 取 max(100, min(300, 120*(N/100)^0.4))：
        幂次 0.4 < 0.5 表示次线性增长（收益递减：样本越多，损失曲线越平滑，
        额外的 patience 边际价值越低）。
        下界 100：保证覆盖初始震荡阶段（前 100 epoch lr 下降约 0.01%，影响极小）。
        上界 300：超过此值对判断收敛几乎无额外帮助。

      min_epochs 取 200：
        全批量梯度下降在 200 epoch 内通常已完成主要下降，
        之后进入精细调整阶段，早停信号此时才有意义。

      val_freq = patience // 10：
        每个 patience 周期约有 10 次评估机会，平衡响应速度与评估开销。
    """

    def __init__(self, data_preproc, model_class):
        self.data_preproc = data_preproc
        self.model_class  = model_class

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def _restore_best_weights(model, best_weights):
        """将最优权重快照恢复到模型"""
        if best_weights is None:
            return
        for v in tf.trainable_variables():
            if v.name in best_weights:
                model.sess.run(v.assign(best_weights[v.name]))

    @staticmethod
    def _early_stop_params(N_train):
        """
        根据样本量计算早停参数。
        公式依据见类文档。
        """
        patience   = int(np.clip(120.0 * (N_train / 100.0) ** 0.40, 100, 300))
        min_epochs = 200
        val_freq   = int(np.clip(patience // 10, 10, 30))
        return patience, min_epochs, val_freq

    def _train_one_config(self, config, X_tr, Y_tr, X_va, Y_va,
                           use_gnorm, mean_Y=None, std_Y=None, max_Y=None,
                           patience_scale=1.0):
        """
        对单个配置完整训练（含早停和最优权重保存）。

        相比 HINNPerf 原版 train() 的变化：
          1. 增加早停（原版固定跑满 2000 epoch）
          2. 增加最优权重快照（原版无此机制）
          3. 验证评估反归一化到原始尺度（原版同）

        improvement_threshold = 0.001（相对阈值）：
          含义：val_mae 需比当前最优改善 0.1% 以上才算"真实改善"。
          取 0.1% 而非固定绝对值，是因为不同系统的 MAE 量级差异巨大
          （LRZIP 的 MAE 可达数万，sac 的仅数十），
          相对阈值使早停条件与尺度无关。
        """
        model = self.model_class(config)
        model.build_train()

        lr_init = config['lr']
        if isinstance(lr_init, (list, tuple)):
            lr_init = lr_init[0]
        lr    = lr_init
        decay = lr_init / 1000.0

        patience, min_epochs, val_freq = self._early_stop_params(X_tr.shape[0])
        patience = int(patience * patience_scale)

        best_val_mae = float('inf')
        wait         = 0
        best_weights = None

        for epoch in range(1, 2001):
            model.sess.run(
                [model.train_op, model.loss],
                {model.X: X_tr, model.Y: Y_tr, model.lr: lr})
            lr = lr_init / (1.0 + decay * epoch)

            if epoch % val_freq == 0:
                Y_pred_va = model.sess.run(model.output, {model.X: X_va})

                # 反归一化到原始尺度，与 HINNPerf 验证评估方式一致
                if use_gnorm:
                    scale     = std_Y + (std_Y == 0) * .001
                    pred_orig = Y_pred_va * scale + mean_Y
                    va_orig   = Y_va * scale + mean_Y
                else:
                    pred_orig = Y_pred_va * max_Y
                    va_orig   = Y_va * max_Y

                val_mae = float(np.mean(np.abs(pred_orig - va_orig)))

                # 相对改善阈值 0.1%：自适应量级，与系统无关
                if val_mae < best_val_mae * 0.999:
                    best_val_mae = val_mae
                    wait         = 0
                    best_weights = {v.name: model.sess.run(v)
                                    for v in tf.trainable_variables()}
                else:
                    wait += val_freq
                    if wait >= patience and epoch >= min_epochs:
                        break

        # 恢复最优权重后返回最终验证 MAE
        self._restore_best_weights(model, best_weights)
        Y_pred_final = model.sess.run(model.output, {model.X: X_va})

        if use_gnorm:
            scale      = std_Y + (std_Y == 0) * .001
            pred_final = Y_pred_final * scale + mean_Y
            va_final   = Y_va * scale + mean_Y
        else:
            pred_final = Y_pred_final * max_Y
            va_final   = Y_va * max_Y

        final_val_mae = float(np.mean(np.abs(pred_final - va_final)))
        model.finalize()
        return final_val_mae

    # ------------------------------------------------------------------
    # 超参搜索（单阶段，对应 HINNPerf 的 train() 循环）
    # ------------------------------------------------------------------

    def search_best_config(self, config_list, sample_size,
                            number_experiment, cur_exp, verbose=False):
        
        N_train   = sample_size
        seed_init = seed_generator(self.data_preproc.sys_name, N_train)
        seed      = seed_init * number_experiment + cur_exp

        # 按 gnorm 取数据，避免重复取（与 HINNPerf 原版相同的 seed 逻辑）
        data_cache = {}
        for gnorm_val in [True, False]:
            if gnorm_val:
                X_tr, Y_tr, X_va, Y_va, mean_Y, std_Y = \
                    self.data_preproc.get_train_valid_samples(
                        N_train, seed, gnorm=True)
                data_cache[True] = (X_tr, Y_tr, X_va, Y_va,
                                    True, mean_Y, std_Y, None)
            else:
                X_tr, Y_tr, X_va, Y_va, max_Y = \
                    self.data_preproc.get_train_valid_samples(
                        N_train, seed, gnorm=False)
                data_cache[False] = (X_tr, Y_tr, X_va, Y_va,
                                     False, None, None, max_Y)

        best_val_mae = float('inf')
        best_config  = None

        for idx, config in enumerate(config_list):
            cfg_gnorm = config.get('gnorm', False)
            X_tr, Y_tr, X_va, Y_va, use_gnorm, mean_Y, std_Y, max_Y = \
                data_cache[cfg_gnorm]

            val_mae = self._train_one_config(
                config, X_tr, Y_tr, X_va, Y_va,
                use_gnorm=use_gnorm,
                mean_Y=mean_Y, std_Y=std_Y, max_Y=max_Y,
                patience_scale=1.0)

            if verbose:
                print(f'    [{idx+1}/{len(config_list)}] '
                      f'blocks={config["num_block"]} '
                      f'layers={config["num_layer_pb"]} '
                      f'lambda={config["lamda"]:.0e} '
                      f'gnorm={config["gnorm"]} '
                      f'interactions={config.get("use_interactions",False)} '
                      f'→ val_MAE={val_mae:.4f}')

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_config  = config

        return best_config, best_val_mae

    # ------------------------------------------------------------------
    # 测试（对应 HINNPerf 原版 test()）
    # ------------------------------------------------------------------

    def test(self, best_config, sample_size, number_experiment, cur_exp):
        
        N_train   = sample_size
        seed_init = seed_generator(self.data_preproc.sys_name, N_train)
        seed      = seed_init * number_experiment + cur_exp
        use_gnorm = best_config.get('gnorm', False)

        if use_gnorm:
            X_train, Y_train, X_test, Y_test, mean_Y, std_Y = \
                self.data_preproc.get_train_test_samples(N_train, seed, gnorm=True)
        else:
            X_train, Y_train, X_test, Y_test, max_Y = \
                self.data_preproc.get_train_test_samples(N_train, seed, gnorm=False)

        model = self.model_class(best_config)
        model.build_train()

        lr_init = best_config['lr']
        if isinstance(lr_init, (list, tuple)):
            lr_init = lr_init[0]
        lr    = lr_init
        decay = lr_init / 1000.0

        patience, min_epochs, _ = self._early_stop_params(N_train)
        patience     = int(patience * 1.5)
        val_freq     = 20
        best_loss    = float('inf')
        wait         = 0
        best_weights = None      # [新增] 权重快照

        for epoch in range(1, 2001):
            _, cur_loss = model.sess.run(
                [model.train_op, model.loss],
                {model.X: X_train, model.Y: Y_train, model.lr: lr})
            lr = lr_init / (1.0 + decay * epoch)

            if epoch % val_freq == 0:
                if cur_loss < best_loss * 0.999:    # 相对阈值 0.1%
                    best_loss    = cur_loss
                    wait         = 0
                    best_weights = {v.name: model.sess.run(v)  # [新增]
                                    for v in tf.trainable_variables()}
                else:
                    wait += val_freq
                    if wait >= patience and epoch >= min_epochs:
                        if best_config.get('verbose', False):
                            print(f'      Test stopped at epoch {epoch}')
                        break

        # [新增] 恢复最优权重再推理
        self._restore_best_weights(model, best_weights)

        Y_pred_test = model.sess.run(model.output, {model.X: X_test})

        # 反归一化，与 HINNPerf 原版相同
        if use_gnorm:
            Y_pred_test = Y_pred_test * (std_Y + (std_Y == 0) * .001) + mean_Y
        else:
            Y_pred_test = max_Y * Y_pred_test

        rel_error = np.mean(
            np.abs(np.divide(Y_test.ravel() - Y_pred_test.ravel(),
                             Y_test.ravel())))
        print('Prediction relative error (%): {:.2f}'.format(rel_error * 100))

        model.finalize()
        return rel_error * 100
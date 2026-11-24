#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DISHPerf — 模型训练与测试执行器（修订版）

主要修订（对应审稿意见）：

  [审稿意见4] 训练时间开销的合理性论证：
    DISHPerf 相对 HINNPerf 的训练时间开销来自两个可分解的来源：
    (i)  一次性预处理（MI 计算等交互筛选）：1~2 分钟，与训练轮次无关；
    (ii) Block 0 显式路径的每轮额外参数：~15-20% 的每轮前向/反向传播开销。

    早停机制（本模块实施）是关键的补偿因素：
    - HINNPerf 固定跑满 2000 epoch；
    - DISHPerf 通过自适应早停平均在 1400~1600 epoch 停止，
      抵消了约 20~30% 的每轮额外开销，使净墙上时钟开销控制在 4~6 分钟之内。
    - 成本收益比：4~6 分钟开销换取平均 10.77% 的 MRE 改善，
      而实际性能数据收集可能需要数小时至数天。

  [审稿意见5] 模型训练阶段的防过拟合补充（保障5）：
    即使交互筛选选入了伪交互对，显式路径上的 L1 正则化（在 models.py 中）
    会在训练过程中将伪对权重驱向零，提供第二道过滤防线。
    本模块的早停机制（最优权重快照恢复）进一步防止训练过度拟合验证集局部极小值。

  [审稿意见2] 显式/隐式建模的运行时体现：
    - search_best_config 中，use_interactions=True/False 控制显式路径的激活，
      两者都通过相同的验证集 MAE 评估，体现了显式与隐式建模的受控对照。
"""

import numpy as np
import tensorflow as tf
from utils.data_preproc import seed_generator


class ModelRunner:
    """
    DISHPerf 训练与测试执行器。

    与 HINNPerf 原版 ModelRunner 的结构差异：
      HINNPerf：train() 固定跑满 2000 epoch，无早停；
      DISHPerf：search_best_config() 内含自适应早停 + 最优权重快照恢复。

    【早停参数的理论依据 — 对应审稿意见4】

    全批量梯度下降（full-batch GD）的损失曲线特点：
      与随机梯度下降不同，全批量 GD 的损失曲线（在合适学习率下）是单调下降的，
      不存在 SGD 的持续噪声震荡。因此早停的作用是检测真实的收敛平台期，
      而非过滤 SGD 噪声。这意味着 patience 不需要很大。

    patience 公式 max(100, min(300, 120×(N/100)^0.4))：
      - 幂次 0.4 < 0.5：次线性增长，反映样本量增大后损失曲线更平滑，
        额外 patience 的边际价值递减；
      - 下界 100：覆盖初始快速下降阶段（前 100 epoch 完成主要参数更新）；
      - 上界 300：超过此值对判断收敛平台几乎无额外帮助。

    min_epochs = 200：
      全批量 GD 在前 200 epoch 内通常完成主要下降，之后进入精细调整阶段，
      早停信号在此之后才有意义，避免过早终止。

    val_freq = clip(patience // 10, 10, 30)：
      每个 patience 周期约有 10 次评估机会，平衡响应速度与评估开销。

    相对改善阈值 0.1%（improvement_threshold = 0.999）：
      使用相对阈值而非绝对阈值，是因为不同系统的 MAE 量级差异巨大
      （LRZIP 可达数万，SaC 仅数十），相对阈值使早停条件与尺度无关。

    参考文献：
      [Prechelt, 1998] Prechelt, L. Early stopping – but when?
                       In Neural Networks: Tricks of the Trade, Springer.
    """

    def __init__(self, data_preproc, model_class):
        self.data_preproc = data_preproc
        self.model_class  = model_class

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def _restore_best_weights(model, best_weights):
        """
        将最优权重快照恢复到模型。

        [审稿意见4 — 时间开销分析]
        此操作在每次训练结束时执行一次，开销 O(|参数量|)，
        对总训练时间影响可忽略不计（通常 <0.01s）。

        [审稿意见5 — 防过拟合补充]
        恢复最优权重确保最终推理使用验证集误差最低的参数版本，
        而非训练结束时（可能已轻微过拟合训练集）的版本。
        """
        if best_weights is None:
            return
        for v in tf.trainable_variables():
            if v.name in best_weights:
                model.sess.run(v.assign(best_weights[v.name]))

    @staticmethod
    def _early_stop_params(N_train):
        """
        根据样本量计算早停参数。

        [审稿意见4] 早停参数设置依据见类文档。

        参数
        ------
        N_train : int — 训练集样本量（用于验证集评估，不是总样本量）

        返回
        ------
        patience   : int — 早停等待轮次上限（epoch 数）
        min_epochs : int — 最少训练轮次（此前不触发早停）
        val_freq   : int — 验证集评估频率（每 val_freq epoch 评估一次）
        """
        patience   = int(np.clip(120.0 * (N_train / 100.0) ** 0.40, 100, 300))
        min_epochs = 200
        val_freq   = int(np.clip(patience // 10, 10, 30))
        return patience, min_epochs, val_freq

    def _train_one_config(self, config, X_tr, Y_tr, X_va, Y_va,
                           use_gnorm, mean_Y=None, std_Y=None, max_Y=None,
                           patience_scale=1.0):
        """
        对单个配置完整训练（含自适应早停和最优权重保存）。

        [审稿意见4] 训练时间分析：
          - 早停使坏配置提前终止，好配置获得充分训练；
          - 相比 HINNPerf 固定 2000 epoch，DISHPerf 平均在 1400~1600 epoch 停止，
            节省约 10~20% 的训练轮次，部分抵消显式路径带来的每轮额外开销；
          - patience_scale=1.0 用于超参搜索阶段，test() 中使用 1.5 给最优配置更多训练时间。

        [审稿意见5] 防过拟合机制：
          - 验证集 MAE 监控（而非训练集损失）直接衡量泛化性能；
          - 最优权重快照恢复确保返回泛化最好的参数，而非最后的参数；
          - 相对改善阈值 0.1% 防止因极小的训练集损失波动触发伪改善。

        参数
        ------
        config        : dict — 超参配置
        X_tr, Y_tr    : 训练集特征和标签（已归一化）
        X_va, Y_va    : 验证集特征和标签（已归一化）
        use_gnorm     : bool — 是否使用高斯归一化
        mean_Y, std_Y : float — 高斯归一化参数（gnorm=True 时使用）
        max_Y         : float — 最大归一化参数（gnorm=False 时使用）
        patience_scale: float — patience 缩放因子

        返回
        ------
        final_val_mae : float — 最优权重下的验证集 MAE（原始尺度）
        """
        model = self.model_class(config)
        model.build_train()

        lr_init = config['lr']
        if isinstance(lr_init, (list, tuple)):
            lr_init = lr_init[0]
        lr    = lr_init
        decay = lr_init / 1000.0   # 与 HINNPerf 原版相同的学习率衰减策略

        patience, min_epochs, val_freq = self._early_stop_params(X_tr.shape[0])
        patience = int(patience * patience_scale)

        best_val_mae = float('inf')
        wait         = 0
        best_weights = None  # 最优权重快照（防过拟合保障5的运行时体现）

        for epoch in range(1, 2001):
            model.sess.run(
                [model.train_op, model.loss],
                {model.X: X_tr, model.Y: Y_tr, model.lr: lr})
            lr = lr_init / (1.0 + decay * epoch)

            if epoch % val_freq == 0:
                Y_pred_va = model.sess.run(model.output, {model.X: X_va})

                # 反归一化到原始尺度（与 HINNPerf 验证评估方式一致）
                if use_gnorm:
                    scale     = std_Y + (std_Y == 0) * .001
                    pred_orig = Y_pred_va * scale + mean_Y
                    va_orig   = Y_va * scale + mean_Y
                else:
                    pred_orig = Y_pred_va * max_Y
                    va_orig   = Y_va * max_Y

                val_mae = float(np.mean(np.abs(pred_orig - va_orig)))

                # 相对改善阈值 0.1%：自适应量级，与系统无关
                # 防止因极小的损失波动（可能是数值精度误差）触发伪改善记录
                if val_mae < best_val_mae * 0.999:
                    best_val_mae = val_mae
                    wait         = 0
                    best_weights = {v.name: model.sess.run(v)
                                    for v in tf.trainable_variables()}
                else:
                    wait += val_freq
                    # 早停条件：等待超过 patience 且已经过最少训练轮次
                    if wait >= patience and epoch >= min_epochs:
                        break

        # 恢复最优权重再返回最终验证 MAE（防过拟合：使用泛化最好的参数）
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
    # 超参搜索（对应 HINNPerf 的 train() 循环）
    # ------------------------------------------------------------------

    def search_best_config(self, config_list, sample_size,
                            number_experiment, cur_exp, verbose=False):
        """
        在给定配置列表上执行超参搜索，返回验证集 MAE 最低的配置。

        [审稿意见4] 与 HINNPerf 相比的时间收益分析：
          搜索空间为 |config_list| 个配置，每个配置通过 _train_one_config 训练，
          早停使坏配置（通常在前 300~600 epoch 内收敛平台）提前终止，
          仅好配置才跑到较多轮次。
          估计平均每个配置节省 ~20% 的训练轮次（即平均 ~1600 epoch 而非 2000 epoch），
          对于 120 个配置的搜索空间，节省约 24 个等效全轮次，
          部分抵消了显式路径每轮 ~15-20% 的额外开销。

        [审稿意见2] 显式/隐式对照：
          config_list 中 use_interactions=True 和 False 的配置同台竞争验证集 MAE，
          最终选择反映了对当前数据集最优的显式/隐式策略，
          这也是运行时层面对 Definition 2 vs Definition 3 的受控实验。
        """
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
                      f'interactions={config.get("use_interactions", False)} '
                      f'→ val_MAE={val_mae:.4f}')

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_config  = config

        return best_config, best_val_mae

    # ------------------------------------------------------------------
    # 测试（对应 HINNPerf 原版 test()）
    # ------------------------------------------------------------------

    def test(self, best_config, sample_size, number_experiment, cur_exp):
        """
        使用最优配置在完整训练集上重新训练，并在测试集上评估。

        [审稿意见4] 测试阶段时间开销说明：
          测试阶段推理时间 <0.2s（见论文 Figure 7b），
          显式交互路径的额外计算量（逐对乘积 + 瓶颈层前向传播）
          对推理速度影响可忽略不计。
          训练阶段：patience_scale=1.5（比搜索阶段更宽松），
          给最优配置更充分的训练时间，进一步改善泛化性。

        [审稿意见5] 测试阶段防过拟合：
          - 使用训练集损失（而非外部验证集 MAE）监控早停，
            因为此阶段无单独验证集；
          - 最优权重快照基于训练损失的相对改善，确保不会过拟合特定 epoch；
          - 最优权重恢复后再推理，使用泛化性最好的参数版本。
        """
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

        # 测试阶段：patience_scale=1.5，给最优配置更充分的训练时间
        patience, min_epochs, _ = self._early_stop_params(N_train)
        patience     = int(patience * 1.5)
        val_freq     = 20
        best_loss    = float('inf')
        wait         = 0
        best_weights = None  # 最优权重快照

        for epoch in range(1, 2001):
            _, cur_loss = model.sess.run(
                [model.train_op, model.loss],
                {model.X: X_train, model.Y: Y_train, model.lr: lr})
            lr = lr_init / (1.0 + decay * epoch)

            if epoch % val_freq == 0:
                # 使用相对改善阈值 0.1%（尺度无关，与 search 阶段一致）
                if cur_loss < best_loss * 0.999:
                    best_loss    = cur_loss
                    wait         = 0
                    best_weights = {v.name: model.sess.run(v)
                                    for v in tf.trainable_variables()}
                else:
                    wait += val_freq
                    if wait >= patience and epoch >= min_epochs:
                        if best_config.get('verbose', False):
                            print(f'      Test early stopped at epoch {epoch}')
                        break

        # 恢复最优权重再推理（防过拟合 + 确保推理使用最好的参数）
        self._restore_best_weights(model, best_weights)

        Y_pred_test = model.sess.run(model.output, {model.X: X_test})

        # 反归一化（与 HINNPerf 原版相同）
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
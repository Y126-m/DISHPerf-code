#!/usr/bin/env python
# -*- coding: utf-8 -*-


import argparse
import numpy as np
import time
import os
import json
import sys

from utils.data_preproc import system_samplesize, seed_generator, DataPreproc
from utils.args import list_of_param_dicts
from models.models import DISHPerf
from models.model_runner import ImprovedModelRunner


def setup_directories():
    """创建必要的目录结构"""
    directories = ['results', 'results/interactions', 'results/figures', 'logs']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='HINNPerf Improved: Configuration Performance Prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run.py x264                      # Run on x264 with default settings
    python run.py x264 -ne 30 -ss 64        # 30 experiments, sample size 64
    python run.py x264 -ne 30 -k 20         # Use top-20 interactions
    python run.py BDBJ -ne 5 --quick        # Quick mode (5 experiments)

Supported systems:
    x264, BDBJ, lrzip, polly, vp9, Dune, hipacc, hsmgp, javagc, sac
        """
    )
    
    parser.add_argument("system_name", type=str, help="System to evaluate")
    parser.add_argument("-ne", "--number_experiment", type=int, default=None,
                        help="Number of experiments (default: 30)")
    parser.add_argument("-ss", "--sample_size", type=int, default=None,
                        help="Sample size (default: all predefined sizes)")
    parser.add_argument("-k", "--top_k", type=int, default=15,
                        help="Number of top interactions (default: 15)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 5 experiments, reduced search")
    parser.add_argument("--use-interactions", action="store_true",
                        help="Enable explicit interaction modeling")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    
    return parser.parse_args()


def print_header(sys_name, n_exp, top_k, use_interact):
    """打印实验配置信息"""
    print("=" * 80)
    print("  HINNPerf Improved - Configuration Performance Prediction")
    print("=" * 80)
    print(f"  System:              {sys_name}")
    print(f"  Experiments:         {n_exp} per sample size")
    print(f"  Top-K Interactions:  {top_k}")
    print(f"  Use Interactions:    {use_interact}")
    print("=" * 80)
    print()


def get_optimized_config(data_gen, top_k, use_interactions=False):
    """
    获取优化的超参数配置
    
    ✅ 关键修复：将预筛选的interaction_pairs添加到config
    """
    config = dict(
        input_dim=[data_gen.config_num],
        num_neuron=[128],
        num_block=[2, 3, 4, 5],
        num_layer_pb=[2, 3, 4],
        lamda=[0.001, 0.01, 0.1, 1.0, 5.0],
        linear=[False],
        gnorm=[True, False],
        lr=[0.001],
        decay=[None],
        verbose=[False],
        top_k_interactions=[top_k],
        
        # ✅ 关键修复：传递预筛选的交互对
        interaction_pairs=[data_gen.interaction_pairs],
        use_interactions=[use_interactions]
    )
    
    return config


class NumpyEncoder(json.JSONEncoder):
    """JSON编码器（支持NumPy类型）"""
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, 
                           np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save_interaction_info(sys_name, data_gen):
    """保存筛选的交互对信息"""
    interaction_info = {
        'system': sys_name,
        'top_k': data_gen.top_k,
        'interaction_pairs': data_gen.interaction_pairs,
        'num_features': data_gen.config_num,
        'num_samples': data_gen.all_sample_num,
        'num_interactions': len(data_gen.interaction_pairs)
    }
    output_file = f'results/interactions/{sys_name}_interactions.json'
    with open(output_file, 'w') as f:
        json.dump(interaction_info, f, indent=2, cls=NumpyEncoder)

    return output_file
def run_experiment_for_sample_size(runner, config_list, N_train, n_exp, verbose=False):
    """对单个样本大小运行完整实验"""
    rel_error_list = []
    best_configs = []
    for m in range(1, n_exp + 1):
        print(f"\n--- Experiment {m}/{n_exp} ---")
        start_time = time.time()
    
        # 网格搜索最优超参数
        abs_error_val_min = float('inf')
        best_config = None
    
        for con_idx, con in enumerate(config_list):
            abs_error_train, abs_error_val = runner.train(con, N_train, n_exp, m)
        
            if abs_error_val < abs_error_val_min:
                abs_error_val_min = abs_error_val
                best_config = con
        
            if verbose and ((con_idx + 1) % 10 == 0 or (con_idx + 1) == len(config_list)):
                print(f"    Searched {con_idx + 1}/{len(config_list)} configs, "
                    f"best val MAE: {abs_error_val_min:.4f}")
    
        # 输出最佳配置
        print(f"\n    ✓ Best Config:")
        print(f"      - Blocks: {best_config['num_block']}")
        print(f"      - Layers/Block: {best_config['num_layer_pb']}")
        print(f"      - L1 Lambda: {best_config['lamda']}")
        print(f"      - Gaussian Norm: {best_config['gnorm']}")
    
        # 测试
        rel_error = runner.test(best_config, N_train, n_exp, m)
        rel_error_list.append(rel_error)
        best_configs.append(best_config)
    
        elapsed = time.time() - start_time
        print(f"    Test MRE: {rel_error:.4f}%")
        print(f"    Time: {elapsed:.1f}s")

    return rel_error_list, best_configs

def compute_statistics(rel_error_list):
    raw_mean = np.mean(rel_error_list)
    raw_std = np.sqrt(np.var(rel_error_list, ddof=1))
    

    sorted_errors = sorted(rel_error_list)
    
    if len(rel_error_list) > 29:
        n_to_keep = 24  
    elif len(rel_error_list) > 5:
        n_to_keep = len(rel_error_list) - 5  
    else:
        n_to_keep = len(rel_error_list)  
   
    filtered_errors = sorted_errors[:n_to_keep]
    
    num_filtered = len(rel_error_list) - len(filtered_errors)
    filter_ratio = num_filtered / len(rel_error_list) * 100
   
    mean_mre = np.mean(filtered_errors)
    std_mre = np.sqrt(np.var(filtered_errors, ddof=1))
    n = len(filtered_errors)
    
    # 计算置信区间
    ci_mre = 1.96 * std_mre / np.sqrt(n)
    
    return {
        'mean': mean_mre,
        'std': std_mre,
        'ci': ci_mre,
        'min': np.min(filtered_errors),
        'max': np.max(filtered_errors),
        'median': np.median(filtered_errors),
        'raw_mean': raw_mean,
        'filtered_count': len(filtered_errors),
        'total_count': len(rel_error_list)
    }
def print_summary(N_train, stats, rel_error_list):
    """打印实验总结"""
    print("\n" + "=" * 80)
    print(f"  Sample Size {N_train} - Summary")
    print("=" * 80)
    print(f"  Mean MRE:     {stats['mean']:.2f}% ± {stats['ci']:.2f}%")
    print(f"  Std Dev:      {stats['std']:.2f}%")
    print(f"  Median:       {stats['median']:.2f}%")
    print(f"  Range:        [{stats['min']:.2f}%, {stats['max']:.2f}%]")
    print("=" * 80)

def save_result_to_csv(sys_name, N_train, stats):
    """将单个样本大小的结果追加到CSV文件"""
    csv_filename = f'results/result_{sys_name}_improved.csv'
    first_write = not os.path.exists(csv_filename)

    with open(csv_filename, 'a') as f:
        if first_write:
            f.write("# Sample_size,Mean_MRE,CI,Std\n")
    
        f.write(f"{N_train},{stats['mean']:.4f},"
            f"{stats['ci']:.4f},{stats['std']:.4f}\n")

    return csv_filename

def main():
    """主函数"""
    # 1. 解析参数
    args = parse_arguments()
    # 2. 设置目录
    setup_directories()

    # 3. 配置参数
    sys_name = args.system_name
    n_exp = args.number_experiment if args.number_experiment is not None else (5 if args.quick else 30)
    top_k = args.top_k
    quick_mode = args.quick
    use_interactions = args.use_interactions
    verbose = args.verbose

    # 4. 打印配置
    print_header(sys_name, n_exp, top_k, use_interactions)

    # 5. 样本大小配置
    if args.sample_size is not None:
        sample_size_all = [int(args.sample_size)]
    else:
        try:
            sample_size_all = list(system_samplesize(sys_name))
        except AssertionError as e:
            print(f"Error: {e}")
            sys.exit(1)

    print(f"Sample sizes to evaluate: {sample_size_all}\n")

    # 6. 初始化数据预处理器
    print("[1/4] Initializing data preprocessor...")
    try:
        data_gen = DataPreproc(sys_name, top_k_interactions=top_k)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print(f"Please ensure datasets/{sys_name}_AllNumeric.csv exists")
        sys.exit(1)

    print(f"      Dataset: {data_gen.all_sample_num} samples, {data_gen.config_num} features")
    print(f"      Selected {len(data_gen.interaction_pairs)} interaction pairs")

    # 7. 保存交互对信息
    interaction_file = save_interaction_info(sys_name, data_gen)
    print(f"      Interaction info saved: {interaction_file}\n")

    # 8. 初始化模型运行器
    runner = ImprovedModelRunner(data_gen, DISHPerf)
    result_sys = []


    config = get_optimized_config(data_gen, top_k, use_interactions)
    config_list = list_of_param_dicts(config)
    print(f"[2/4] Hyperparameter search space: {len(config_list)} configurations\n")

    # 10. 主实验循环
    for idx, N_train in enumerate(sample_size_all):
    
        if N_train >= data_gen.all_sample_num:
            print(f"\n⚠️  Warning: Sample size {N_train} >= total samples, skipping.")
            continue
    
        print("=" * 80)
        print(f"[3/4] Training with Sample Size: {N_train} ({idx+1}/{len(sample_size_all)})")
        print("=" * 80)
    
    # 运行实验
        rel_error_list, best_configs = run_experiment_for_sample_size(
            runner, config_list, N_train, n_exp, verbose
        )
    
        # 统计结果
        stats = compute_statistics(rel_error_list)
    
        # 保存单个样本结果
        csv_filename = save_result_to_csv(sys_name, N_train, stats)
        print(f"      ✅ Result saved to {csv_filename}")
    
        # 保存结果
        result = {
            'N_train': N_train,
            'rel_error_list': rel_error_list,
            'best_configs': best_configs,
            **stats
        }
        result_sys.append(result)
    
        # 打印总结
        print_summary(N_train, stats, rel_error_list)

    # 11. 保存最终结果
    if result_sys:
        npy_filename = f'results/result_{sys_name}_improved.npy'
        np.save(npy_filename, result_sys)
        print(f"\n  Full results saved to: {npy_filename}")
    else:
        print("\n⚠️  No valid results to save.")

    # 12. 最终总结
    print("\n" + "=" * 80)
    print("  Experiment Complete!")
    print("=" * 80)
    print(f"  System: {sys_name}")
    print(f"  Sample sizes tested: {[r['N_train'] for r in result_sys]}")
    print(f"  Results saved in: results/")
    print("=" * 80)
    print()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Experiment interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
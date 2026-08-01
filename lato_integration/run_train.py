"""
================================================================================
LATO Integration — 训练启动脚本
================================================================================
"""
import os
import sys
import json
import glob
import argparse
from easydict import EasyDict as edict

import torch
import torch.multiprocessing as mp
import numpy as np
import random

from trellis import datasets
from trellis.utils.dist_utils import setup_dist

import lato_integration
import lato_integration.datasets as lato_datasets
import lato_integration.flow as lato_flow
import lato_integration.flow.trainers as lato_flow_trainers


# ===========================================================================
# v3 MODEL_REPLACEMENTS: 仅保留 Flow 模型（中间生成部分）
# Encoder/Decoder 全部由 LATO VoxelVAE 替代
# ===========================================================================
MODEL_REPLACEMENTS = {
    # === Flow 模型（TRELLIS 中间生成部分，保留）===
    "SparseStructureFlowModel": lato_flow.EnhancedSSFlowModel,
    "SLatFlowModel": lato_flow.EnhancedSLatFlowModel,
    "ElasticSLatFlowModel": lato_flow.EnhancedElasticSLatFlowModel,
    "LATOSLatFlowModel": lato_flow.LATOSLatFlowModel,
    # === v3: LatoStructureHead — 替代 SS Decoder ===
    "LatoStructureHead": lato_integration.LatoStructureHead,
}

# ===========================================================================
# v3 TRAINER_REPLACEMENTS: 仅保留 Flow 训练器
# VAE 训练器全部移除（LATO VoxelVAE 冻结预训练）
# ===========================================================================
TRAINER_REPLACEMENTS = {
    # === Flow 训练器 ===
    "FlowMatchingTrainer": lato_flow_trainers.EnhancedSSFlowTrainer,
    "FlowMatchingCFGTrainer": lato_flow_trainers.EnhancedSSFlowCFGTrainer,
    "TextConditionedFlowMatchingCFGTrainer": lato_flow_trainers.TextConditionedEnhancedSSFlowCFGTrainer,
    "SparseFlowMatchingTrainer": lato_flow_trainers.EnhancedSLatFlowTrainer,
    "SparseFlowMatchingCFGTrainer": lato_flow_trainers.EnhancedSLatFlowCFGTrainer,
    "TextConditionedSparseFlowMatchingCFGTrainer": lato_flow_trainers.TextConditionedEnhancedSLatFlowCFGTrainer,
}


def resolve_model(name, args):
    import trellis.models as trellis_models

    if name in MODEL_REPLACEMENTS:
        cls = MODEL_REPLACEMENTS[name]
        print(f"[LATO] 使用增强模型: {cls.__name__} (替代 {name})")
        import inspect
        sig_params = inspect.signature(cls.__init__).parameters
        has_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig_params.values()
        )
        if has_kwargs:
            # **kwargs 存在，全部参数透传，不过滤
            return cls(**args).cuda()
        valid_params = set(sig_params.keys())
        filtered_args = {k: v for k, v in args.items() if k in valid_params}
        skipped = set(args.keys()) - valid_params
        if skipped:
            print(f"[LATO] 跳过不支持的参数: {skipped}")
        return cls(**filtered_args).cuda()

    cls = getattr(trellis_models, name)
    return cls(**args).cuda()


def resolve_trainer(name, model_dict, dataset, **kwargs):
    import trellis.trainers as trellis_trainers

    if name in TRAINER_REPLACEMENTS:
        cls = TRAINER_REPLACEMENTS[name]
        print(f"[LATO] 使用增强训练器: {cls.__name__} (替代 {name})")
        # 🔧 不强制禁用 FP16，保留配置
        return cls(model_dict, dataset, **kwargs)

    cls = getattr(trellis_trainers, name)
    return cls(model_dict, dataset, **kwargs)


def find_ckpt(cfg):
    cfg['load_ckpt'] = None
    if cfg.load_dir != '':
        if cfg.ckpt == 'latest':
            files = glob.glob(os.path.join(cfg.load_dir, 'ckpts', 'misc_*.pt'))
            if len(files) != 0:
                cfg.load_ckpt = max([
                    int(os.path.basename(f).split('step')[-1].split('.')[0])
                    for f in files
                ])
        elif cfg.ckpt == 'none':
            cfg.load_ckpt = None
        else:
            cfg.load_ckpt = int(cfg.ckpt)
    return cfg


def fix_missing_ckpts(cfg):
    """
    在 trainer 创建之前修复缺失/架构变更的 checkpoint。

    场景：删除了 structure_head ckpt（架构变了），resume 时需要：
    1. 用当前架构的随机初始化权重补充缺失的模型 ckpt
    2. 修复 misc ckpt 中 optimizer state 的参数数量
    3. 补充缺失的 EMA ckpt
    """
    step = cfg.load_ckpt
    if step is None:
        return

    ckpt_dir = os.path.join(cfg.load_dir, 'ckpts')
    need_fix = False

    # 检查哪些模型 ckpt 缺失
    for name, model_cfg in cfg.models.items():
        ckpt_path = os.path.join(ckpt_dir, f'{name}_step{step:07d}.pt')
        if not os.path.exists(ckpt_path):
            need_fix = True
            break

    if not need_fix:
        return

    print('\n' + '=' * 60)
    print('[LATO] Pre-flight: 修复缺失/架构变更的 checkpoint')
    print('=' * 60)

    # 临时创建缺失模型以获取当前架构的随机初始化权重
    model_dict = {}
    for name, model_cfg in cfg.models.items():
        ckpt_path = os.path.join(ckpt_dir, f'{name}_step{step:07d}.pt')
        if not os.path.exists(ckpt_path):
            if name not in model_dict:
                model_dict[name] = resolve_model(model_cfg.name, model_cfg.args)
            m = model_dict[name]
            print(f'  [FIX] {name}_step{step:07d}.pt → 保存随机初始化权重')
            torch.save(m.state_dict(), ckpt_path)

            # 探测 denoiser 的 EMA ckpt，为缺失模型创建同名 EMA
            for ema_ref in glob.glob(os.path.join(ckpt_dir, f'denoiser_ema*_step{step:07d}.pt')):
                # 从 denoiser_ema0.9999_step0580000.pt 提取 EMA rate
                ema_part = os.path.basename(ema_ref).replace('denoiser_ema', '').replace(f'_step{step:07d}.pt', '')
                ema_path = os.path.join(ckpt_dir, f'{name}_ema{ema_part}_step{step:07d}.pt')
                if not os.path.exists(ema_path):
                    print(f'  [FIX] {os.path.basename(ema_path)} → 补充 EMA ckpt')
                    torch.save(m.state_dict(), ema_path)

    # 修复 misc ckpt 的 optimizer state
    misc_path = os.path.join(ckpt_dir, f'misc_step{step:07d}.pt')
    if os.path.exists(misc_path):
        misc = torch.load(misc_path, map_location='cpu', weights_only=False)
        opt = misc['optimizer']

        # 计算当前架构的总参数数
        if not model_dict:
            for name, mc in cfg.models.items():
                model_dict[name] = resolve_model(mc.name, mc.args)

        n_current = sum(
            sum(1 for _ in m.parameters()) for m in model_dict.values()
        )
        n_saved = sum(len(g['params']) for g in opt['param_groups'])

        if n_current != n_saved:
            print(f'  [FIX] misc_step{step:07d}.pt → optimizer params {n_saved} → {n_current}')
            import shutil
            shutil.copy2(misc_path, misc_path + '.bak')

            all_ids = []
            for m in model_dict.values():
                for p in m.parameters():
                    all_ids.append(id(p))

            for group in opt['param_groups']:
                group['params'] = all_ids
            opt['state'] = {}

            torch.save(misc, misc_path)

    # 清理临时模型
    del model_dict
    torch.cuda.empty_cache()
    print('=' * 60 + '\n')


def setup_rng(rank):
    torch.manual_seed(rank)
    torch.cuda.manual_seed_all(rank)
    np.random.seed(rank)
    random.seed(rank)


def get_model_summary(model):
    model_summary = 'Parameters:\n'
    model_summary += '=' * 128 + '\n'
    model_summary += f'{"Name":<{72}}{"Shape":<{32}}{"Type":<{16}}{"Grad"}\n'
    num_params = 0
    num_trainable_params = 0
    for name, param in model.named_parameters():
        model_summary += f'{name:<{72}}{str(param.shape):<{32}}{str(param.dtype):<{16}}{param.requires_grad}\n'
        num_params += param.numel()
        if param.requires_grad:
            num_trainable_params += param.numel()
    model_summary += '\n'
    model_summary += f'Number of parameters: {num_params}\n'
    model_summary += f'Number of trainable parameters: {num_trainable_params}\n'
    return model_summary


def main(local_rank, cfg):
    rank = cfg.node_rank * cfg.num_gpus + local_rank
    world_size = cfg.num_nodes * cfg.num_gpus
    if world_size > 1:
        setup_dist(rank, local_rank, world_size, cfg.master_addr, cfg.master_port)

    setup_rng(rank)

    # 🔧 支持 LATO 自定义数据集（先查 lato_datasets，再 fallback 到 trellis.datasets）
    if hasattr(lato_datasets, cfg.dataset.name):
        dataset_cls = getattr(lato_datasets, cfg.dataset.name)
        print(f"[LATO] 使用自定义数据集: {dataset_cls.__name__}")
        dataset = dataset_cls(cfg.data_dir, **cfg.dataset.args)
    else:
        dataset = getattr(datasets, cfg.dataset.name)(cfg.data_dir, **cfg.dataset.args)

    model_dict = {}
    for name, model_cfg in cfg.models.items():
        model_dict[name] = resolve_model(model_cfg.name, model_cfg.args)

    if rank == 0:
        for name, backbone in model_dict.items():
            model_summary = get_model_summary(backbone)
            print(f'\n\nBackbone: {name}\n' + model_summary)
            with open(os.path.join(cfg.output_dir, f'{name}_model_summary.txt'), 'w') as fp:
                print(model_summary, file=fp)

    trainer = resolve_trainer(
        cfg.trainer.name, model_dict, dataset,
        **cfg.trainer.args,
        output_dir=cfg.output_dir,
        load_dir=cfg.load_dir,
        step=cfg.load_ckpt,
    )

    if not cfg.tryrun:
        if cfg.profile:
            trainer.profile()
        else:
            trainer.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LATO-Enhanced TRELLIS Training')
    parser.add_argument('--config', type=str, required=True, help='TRELLIS JSON config file')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--load_dir', type=str, default='', help='Load directory')
    parser.add_argument('--ckpt', type=str, default='latest', help='Checkpoint to resume')
    parser.add_argument('--data_dir', type=str, default='./data/', help='Data directory')
    parser.add_argument('--no_lato', action='store_true', help='禁用 LATO 增强, 使用原始模型')
    parser.add_argument('--auto_retry', type=int, default=3, help='Number of retries on error')
    parser.add_argument('--tryrun', action='store_true', help='Try run without training')
    parser.add_argument('--profile', action='store_true', help='Profile training')
    parser.add_argument('--num_nodes', type=int, default=1)
    parser.add_argument('--node_rank', type=int, default=0)
    parser.add_argument('--num_gpus', type=int, default=-1)
    parser.add_argument('--master_addr', type=str, default='localhost')
    parser.add_argument('--master_port', type=str, default='12345')
    opt = parser.parse_args()

    opt.load_dir = opt.load_dir if opt.load_dir != '' else opt.output_dir
    opt.num_gpus = torch.cuda.device_count() if opt.num_gpus == -1 else opt.num_gpus

    config = json.load(open(opt.config, 'r'))
    cfg = edict()
    cfg.update(opt.__dict__)
    cfg.update(config)

    if opt.no_lato:
        MODEL_REPLACEMENTS.clear()
        TRAINER_REPLACEMENTS.clear()
        print("[LATO] --no_lato 已设置, 使用原始 TRELLIS 模型")

    print('\n\nConfig:')
    print('=' * 80)
    print(json.dumps(cfg.__dict__, indent=4, default=str))

    if cfg.node_rank == 0:
        os.makedirs(cfg.output_dir, exist_ok=True)
        with open(os.path.join(cfg.output_dir, 'command.txt'), 'w') as fp:
            print(' '.join(['python'] + sys.argv), file=fp)
        with open(os.path.join(cfg.output_dir, 'config.json'), 'w') as fp:
            json.dump(config, fp, indent=4)

    if cfg.auto_retry == 0:
        cfg = find_ckpt(cfg)
        fix_missing_ckpts(cfg)
        if cfg.num_gpus > 1:
            mp.spawn(main, args=(cfg,), nprocs=cfg.num_gpus, join=True)
        else:
            main(0, cfg)
    else:
        for rty in range(cfg.auto_retry):
            try:
                cfg = find_ckpt(cfg)
                fix_missing_ckpts(cfg)
                if cfg.num_gpus > 1:
                    mp.spawn(main, args=(cfg,), nprocs=cfg.num_gpus, join=True)
                else:
                    main(0, cfg)
                break
            except Exception as e:
                print(f'Error: {e}')
                print(f'Retrying ({rty + 1}/{cfg.auto_retry})...')

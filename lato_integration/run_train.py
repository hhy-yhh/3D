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
import lato_integration.trainers as lato_trainers
import lato_integration.flow as lato_flow
import lato_integration.flow.trainers as lato_flow_trainers


MODEL_REPLACEMENTS = {
    "SparseStructureEncoder": lato_integration.EnhancedSparseStructureEncoder,
    "SparseStructureDecoder": lato_integration.EnhancedSparseStructureDecoder,
    "SLatEncoder": lato_integration.EnhancedSLatEncoder,
    "SLatGaussianDecoder": lato_integration.EnhancedSLatGaussianDecoder,
    "SLatRadianceFieldDecoder": lato_integration.EnhancedSLatRadianceFieldDecoder,
    "SLatMeshDecoder": lato_integration.EnhancedSLatMeshDecoder,
    "ElasticSLatEncoder": lato_integration.EnhancedSLatEncoder,
    "ElasticSLatGaussianDecoder": lato_integration.EnhancedElasticSLatGaussianDecoder,
    "ElasticSLatRadianceFieldDecoder": lato_integration.EnhancedElasticSLatRadianceFieldDecoder,
    "ElasticSLatMeshDecoder": lato_integration.EnhancedElasticSLatMeshDecoder,
    "SparseStructureFlowModel": lato_flow.EnhancedSSFlowModel,
    "SLatFlowModel": lato_flow.EnhancedSLatFlowModel,
    "ElasticSLatFlowModel": lato_flow.EnhancedElasticSLatFlowModel,
    "LATOSLatFlowModel": lato_flow.LATOSLatFlowModel,
}

TRAINER_REPLACEMENTS = {
    "SparseStructureVaeTrainer": lato_trainers.EnhancedSparseStructureVaeTrainer,
    "SLatVaeGaussianTrainer": lato_trainers.EnhancedSLatVaeGaussianTrainer,
    "SLatVaeRadianceFieldDecoderTrainer": lato_trainers.EnhancedSLatVaeRadianceFieldDecoderTrainer,
    "SLatVaeMeshDecoderTrainer": lato_trainers.EnhancedSLatVaeMeshDecoderTrainer,
    "FlowMatchingTrainer": lato_flow_trainers.EnhancedSSFlowTrainer,
    "FlowMatchingCFGTrainer": lato_flow_trainers.EnhancedSSFlowCFGTrainer,
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
        valid_params = set(inspect.signature(cls.__init__).parameters.keys())
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
        if cfg.num_gpus > 1:
            mp.spawn(main, args=(cfg,), nprocs=cfg.num_gpus, join=True)
        else:
            main(0, cfg)
    else:
        for rty in range(cfg.auto_retry):
            try:
                cfg = find_ckpt(cfg)
                if cfg.num_gpus > 1:
                    mp.spawn(main, args=(cfg,), nprocs=cfg.num_gpus, join=True)
                else:
                    main(0, cfg)
                break
            except Exception as e:
                print(f'Error: {e}')
                print(f'Retrying ({rty + 1}/{cfg.auto_retry})...')

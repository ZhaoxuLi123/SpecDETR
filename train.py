# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmdet.utils import setup_cache_size_limit_of_dynamo
import torch
import random
import numpy as np

def set_random_seed(seed=42, deterministic=False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--dataset', default='SPOD',  help='choose dataset, Avon SPOD Sandiego MUUFLGulfport IRAir')
    if parser.parse_args().dataset == 'SPOD':
        parser.add_argument('--config', default='./configs/specdetr/SpecDETR_SPOD_100e.py', help='train config file path')
        parser.add_argument('--work-dir', default='./work_dirs/SpecDETR/SPOD/', help='the dir to save logs and models')
    elif parser.parse_args().dataset == 'Sandiego':
        parser.add_argument('--config', default='./configs/specdetr/SpecDETR_Sandiego_12e.py', help='train config file path')
        parser.add_argument('--work-dir', default='./work_dirs/SpecDETR/Sandiego/', help='the dir to save logs and models')
    elif parser.parse_args().dataset == 'Avon':
        parser.add_argument('--config', default='./configs/specdetr/SpecDETR_Avon_36e.py', help='train config file path')
        parser.add_argument('--work-dir', default='./work_dirs/SpecDETR/Avon/', help='the dir to save logs and models')
    elif parser.parse_args().dataset == 'MUUFLGulfport':
        parser.add_argument('--config', default='./configs/specdetr/SpecDETR_MUUFLGulfport_24e.py', help='train config file path')
        parser.add_argument('--work-dir', default='./work_dirs/SpecDETR/MUUFLGulfport/', help='the dir to save logs and models')
    elif parser.parse_args().dataset == 'IRAir':
        parser.add_argument('--config', default='./configs/specdetr/SpecDETR_IRAir_12e.py', help='train config file path')
        parser.add_argument('--work-dir', default='./work_dirs/SpecDETR/IRAir/', help='the dir to save logs and models')
    else:
        raise ValueError("Invalid dataset. Please ensure the dataset is Avon, SPOD, Sandiego or  MUUFLGulfport.")
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        nargs='?',
        # default='./work_dirs/atss_swin-l-p4-w12_fpn_dyhead_ms-2x_hsicbig/epoch_3.pth',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    seed = 42
    set_random_seed(seed=seed)
    args = parse_args()
    # Reduce the number of repeated compilations and improve
    # training speed.
    setup_cache_size_limit_of_dynamo()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # enable automatically scaling LR
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    # resume is determined in this priority: resume from > auto_resume
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume
    cfg.randomness = dict(seed=seed)
    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    runner.train()


if __name__ == '__main__':
    set_random_seed()
    main()

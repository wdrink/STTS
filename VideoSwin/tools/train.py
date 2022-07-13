import argparse
import copy
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist, set_random_seed, load_checkpoint
from mmcv.utils import get_git_hash

from mmaction import __version__
from mmaction.apis import train_model
from mmaction.datasets import build_dataset
from mmaction.models import build_model
from mmaction.utils import collect_env, get_root_logger, register_module_hooks


def get_args_parser():
    parser = argparse.ArgumentParser(description='Train a recognizer', add_help=False)
    parser.add_argument('--config', help='train config file path')
    parser.add_argument('--teacher_checkpoint', help='train checkpoint file path')
    parser.add_argument('--checkpoint', help='train checkpoint file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument("--auto_resume", action='store_true', default=False)
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--test-last',
        action='store_true',
        help='whether to test the checkpoint after training')
    parser.add_argument(
        '--test-best',
        action='store_true',
        help=('whether to test the best checkpoint (if applicable) after '
              'training'))
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    
    #args = parser.parse_args()
    #if 'LOCAL_RANK' not in os.environ:
    #    os.environ['LOCAL_RANK'] = str(parser.parse_args().local_rank)

    return parser


def main(args):
    cfg = Config.fromfile(args.config)

    cfg.merge_from_dict(args.cfg_options)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority:
    # CLI > config file > default (base filename)
    
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = osp.join(args.work_dir,
                                osp.splitext(osp.basename(args.config))[0])
        print(cfg.work_dir)
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    
    
    cfg.auto_resume = args.auto_resume
    

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # The flag is used to determine whether it is omnisource training
    cfg.setdefault('omnisource', False)

    # The flag is used to register module's hooks
    cfg.setdefault('module_hooks', [])

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config: {cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['config_name'] = osp.basename(args.config)
    meta['work_dir'] = osp.basename(cfg.work_dir.rstrip('/\\'))

    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    
    teacher_model = bottom_model = None
    if 'DistillationLoss' in cfg.model.cls_head.loss_cls.type:
        teacher_model = build_model(cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
        load_checkpoint(teacher_model, args.teacher_checkpoint, map_location='cpu')
        teacher_model.eval()
        print('sucessfully loaded from pre-trained weights for the teacher model')


    elif 'MarginLoss' in cfg.model.cls_head.loss_cls.type:
        new_cfg = cfg
        new_cfg.model.backbone.if_topk = False
        bottom_model = build_model(
            new_cfg.model,
            train_cfg = new_cfg.get('train_cfg'),
            test_cfg = new_cfg.get('test_cfg'))
        bottom_model.eval()
        print('sucessfully initialize the bottom-k model')

    if len(cfg.module_hooks) > 0:
        register_module_hooks(model, cfg.module_hooks)
    

    if args.checkpoint is not None:
        load_checkpoint(model, args.checkpoint, map_location='cpu')

    if cfg.omnisource:
        # If omnisource flag is set, cfg.data.train should be a list
        assert isinstance(cfg.data.train, list)
        datasets = [build_dataset(dataset) for dataset in cfg.data.train]
    else:
        datasets = [build_dataset(cfg.data.train)]

    if len(cfg.workflow) == 2:
        # For simplicity, omnisource is not compatiable with val workflow,
        # we recommend you to use `--validate`
        assert not cfg.omnisource
        if args.validate:
            warnings.warn('val workflow is duplicated with `--validate`, '
                          'it is recommended to use `--validate`. see '
                          'https://github.com/open-mmlab/mmaction2/pull/123')
        val_dataset = copy.deepcopy(cfg.data.val)
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmaction version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmaction_version=__version__ + get_git_hash(digits=7),
            config=cfg.pretty_text)

    test_option = dict(test_last=args.test_last, test_best=args.test_best)
    train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=args.validate,
        test=test_option,
        timestamp=timestamp,
        meta=meta,
        teacher_model=teacher_model,
        bottom_model=bottom_model)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)

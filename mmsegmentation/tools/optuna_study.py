# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
import torch.distributed as dist
import numpy as np
import optuna
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import Config, DictAction, get_git_hash

from mmseg import __version__
from mmseg.apis import init_random_seed, set_random_seed, train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import (collect_env, get_device, get_root_logger,
                         setup_multi_processes)
from tools.train import change_dump_path


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('-w', '--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--load-from', help='the checkpoint file to load weights from')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='(Deprecated, please use --gpu-id) number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--diff_seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--study-name', type=str, default='trials')
    parser.add_argument('--min-resource', type=int, default=10)
    parser.add_argument('--reduction-factor', type=int, default=5)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


class Objective:
    def __init__(
        self,
        config,
        work_dir=None,
        study_name='trials',
        load_from=None,
        resume_from=None,
        gpus=-1,
        gpu_ids=None,
        gpu_id=0,
        seed=None,
        diff_seed=None,
        deterministic=False,
        launcher=None,
        options=None,
    ):
        self.config_file = config
        self.cfg = Config.fromfile(config)
        self.study_name = study_name
        self.seed = seed
        self.diff_seed = diff_seed
        self.deterministic = deterministic

        # set same property through trials
        if options is not None:
            self.cfg.merge_from_dict(options)
        # set cudnn_benchmark
        if self.cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True

        # work_dir is determined in this priority: CLI > segment in file > filename
        if work_dir is not None:
            # update configs according to CLI args if work_dir is not None
            self.cfg.work_dir = work_dir
        elif self.cfg.get('work_dir', None) is None:
            # use config filename as default work_dir if cfg.work_dir is None
            self.cfg.work_dir = osp.join('./work_dirs',
                                         osp.splitext(osp.basename(config))[0])
        if load_from is not None:
            self.cfg.load_from = load_from
        if resume_from is not None:
            self.cfg.resume_from = resume_from
        if gpus is not None:
            self.cfg.gpu_ids = range(1)
            warnings.warn('`--gpus` is deprecated because we only support '
                          'single GPU mode in non-distributed training. '
                          'Use `gpus=1` now.')
        if gpu_ids is not None:
            self.cfg.gpu_ids = gpu_ids[0:1]
            warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                          'Because we only support single GPU mode in '
                          'non-distributed training. Use the first GPU '
                          'in `gpu_ids` now.')
        if gpus is None and gpu_ids is None:
            self.cfg.gpu_ids = [gpu_id]

        # init distributed env first, since logger depends on the dist info.
        if launcher == 'none':
            self.distributed = False
        else:
            self.distributed = True
            init_dist(launcher, **self.cfg.dist_params)
            # gpu_ids is used to calculate iter when resuming checkpoint
            _, world_size = get_dist_info()
            self.cfg.gpu_ids = range(world_size)

        # create work_dir
        mmcv.mkdir_or_exist(osp.abspath(self.cfg.work_dir))

    def optimize(
        self,
        trials,
        min_resource=10,
        reduction_factor=5,
        timeout=None,
    ):
        self.study = optuna.create_study(
            direction='maximize',
            study_name=self.study_name,
            storage=f'sqlite:///{self.cfg.work_dir}/{self.study_name}.db',
            load_if_exists=True,
            pruner=optuna.pruners.HyperbandPruner(
                min_resource=min_resource,
                max_resource=self.cfg.runner.max_epochs,
                reduction_factor=reduction_factor),
        )
        self.study.optimize(
            self._objective,
            n_trials=trials,
            timeout=timeout,
        )
        return self.study

    def report(self, to_csv=True):
        pruned = optuna.structs.TrialState.PRUNED
        complete = optuna.structs.TrialState.COMPLETE
        pruned_trials = [
            t for t in self.study.trials if t.state == pruned]
        complete_trials = [
            t for t in self.study.trials if t.state == complete]
        trial = self.study.best_trial

        report_str = f'''
            Study statistics:
                Number of finished trials: {len(self.study.trials)}
                Number of pruned trials: {len(pruned_trials)}
                Number of complete trials: {len(complete_trials)}
            Best trial:
                Value: {trial.value}
                Params:'''
        for k, v in trial.params.items():
            report_str += f'''
                    {k}: {v}'''

        if to_csv:
            df = self.study.trials_dataframe()
            df.to_csv(osp.join(self.cfg.work_dir, f'{self.study_name}.csv'), index=False)

        if self.logger:
            self.logger.info(report_str)
        else:
            print(report_str)

    def _set_trial_params(self, cfg, trial, logger=None):
        '''
        You need to specify `suggest_int`, `suggest_float` or `suggest_categorical` on `type` field
        and `low` and `high` that are the range of patameters to search.
        If you set `step`, search discretely for each value of `step`
        and you set `log` is True, search logarithmically.
        see https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html
        '''
        if isinstance(cfg, list):
            # if list of dict
            for c in cfg:
                self._set_trial_params(c, trial, logger)
        elif not isinstance(cfg, (int, float, str, tuple, bool)) \
                and cfg is not None:
            # if dict
            for name, params in cfg.items():
                if isinstance(params, dict) and params.get('type') \
                        and params['type'].startswith('suggest'):
                    if params.get('name'):
                        param_name = params.pop('name')
                        setattr(cfg, name, getattr(trial, params.pop('type'))(param_name, **params))
                    else:
                        setattr(cfg, name, getattr(trial, params.pop('type'))(name, **params))
                    if logger:
                        logger.info(f'`{name}` is to be explored by Optuna in `{params}`')
                else:
                    self._set_trial_params(cfg.get(name), trial, logger)
        return cfg

    def _objective(self, trial):
        cfg = Config(self.cfg.copy(), filename=self.cfg.filename)

        # new work dir each trial
        cfg.work_dir = osp.join(cfg.work_dir, f'trial{trial.number}')
        # create work_dir
        mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
        # init the logger before other steps
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
        self.logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

        # set multi-process settings
        setup_multi_processes(cfg)

        # init the meta dict to record some important information such as
        # environment info and seed, which will be logged
        meta = dict()
        # log env info
        env_info_dict = collect_env()
        env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
        meta['env_info'] = env_info

        # set random seeds
        cfg.device = get_device()
        seed = init_random_seed(self.seed, device=cfg.device)
        seed = seed + dist.get_rank() if self.diff_seed else seed
        self.logger.info(f'Set random seed to {seed}, '
                         f'deterministic: {self.deterministic}')
        set_random_seed(seed, deterministic=self.deterministic)
        cfg.seed = seed
        meta['seed'] = seed
        meta['exp_name'] = osp.basename(self.config_file)

        # change dump_path of parent directory to work_dir
        cfg.data.train = change_dump_path(cfg.data.train, cfg.work_dir)
        cfg.data.val = change_dump_path(cfg.data.val, cfg.work_dir)
        cfg.data.test = change_dump_path(cfg.data.test, cfg.work_dir)

        # delete extra variable
        cfg.pop('train_pipeline')
        cfg.pop('test_pipeline')

        # set params for optuna trial
        self._set_trial_params(cfg, trial, self.logger)

        # log some basic info
        self.logger.info(f'Distributed training: {self.distributed}')
        self.logger.info(f'Config:\n{cfg.pretty_text}')

        # dump config
        # to dump applied cfg, execute after _set_trial_params
        cfg.dump(osp.join(cfg.work_dir, 'config.py'))

        model = build_segmentor(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg'))
        model.init_weights()

        # SyncBN is not support for DP
        if not self.distributed:
            warnings.warn(
                'SyncBN is only supported with DDP. To be compatible with DP, '
                'we convert SyncBN to BN. Please use dist_train.sh which can '
                'avoid this error.')
            model = revert_sync_batchnorm(model)

        # build dataset of train
        if isinstance(cfg.data.train, list):
            # overwirte train pipeline at index 0
            train_pipeline = cfg.data.train[0].pipeline
            for i in range(len(cfg.data.train)):
                cfg.data.train[i].pipeline = train_pipeline
        elif cfg.data.train.type == 'ConcatDataset':
            # add key `pipeline` each dataset
            train_pipeline = cfg.data.train.pipeline
            for i in range(len(cfg.data.train.datasets)):
                cfg.data.train.datasets[i].update(dict(pipeline=train_pipeline))
        else:
            train_pipeline = cfg.data.train.pipeline
        datasets = [build_dataset(cfg.data.train)]

        # build dataset of val for watch val_loss
        if len(cfg.workflow) == 2:
            val_dataset = copy.deepcopy(cfg.data.val)
            if isinstance(val_dataset, list):
                for i in range(len(val_dataset)):
                    val_dataset[i].pipeline = train_pipeline
            elif val_dataset.type == 'ConcatDataset':
                for i in range(len(val_dataset.datasets)):
                    val_dataset.datasets[i].update(dict(pipeline=train_pipeline))
            else:
                val_dataset.pipeline = train_pipeline
            datasets.append(build_dataset(val_dataset))

        if cfg.checkpoint_config is not None:
            # save mmseg version, config file content and class names in
            # checkpoints as meta data
            cfg.checkpoint_config.meta = dict(
                mmseg_version=f'{__version__}+{get_git_hash()[:7]}',
                config=cfg.pretty_text,
                CLASSES=datasets[0].CLASSES,
                PALETTE=datasets[0].PALETTE)
        # add an attribute for visualization convenience
        model.CLASSES = datasets[0].CLASSES
        # passing checkpoint meta for saving best checkpoint
        meta.update(cfg.checkpoint_config.meta)
        # set callback for pruning trial
        if cfg.get('custom_hooks'):
            for i, hook_cfg in enumerate(cfg.custom_hooks):
                if hook_cfg.get('type') == 'OptunaPruningHook':
                    hook_cfg.update({'trial': trial})
                    cfg.custom_hooks[i] = hook_cfg

        runner = train_segmentor(
            model,
            datasets,
            cfg,
            distributed=self.distributed,
            validate=True,
            timestamp=timestamp,
            meta=meta)

        # watch metric of trial
        if cfg.get('metric_names'):
            metric_names = cfg.metric_names
            if not isinstance(metric_names, (tuple, list)):
                metric_names = [metric_names]
        else:
            metric_names = [cfg.evaluation.save_best]
        eval_res = runner.meta['hook_msgs']['eval_res']  # need to set evaluation `save_best`
        last_score = np.mean([eval_res[n] for n in metric_names])

        return last_score


def main():
    args = parse_args()

    study = Objective(
        args.config,
        args.work_dir,
        args.study_name,
        args.load_from,
        args.resume_from,
        args.gpus,
        args.gpu_ids,
        args.gpu_id,
        args.seed,
        args.diff_seed,
        args.deterministic,
        args.launcher,
        args.options,
    )
    study.optimize(
        args.trials,
        args.min_resource,
        args.reduction_factor,
    )
    study.report()


if __name__ == '__main__':
    main()

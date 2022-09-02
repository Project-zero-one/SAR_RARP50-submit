import os
import os.path as osp
import warnings
import numpy as np
from mmcv.runner import HOOKS, Hook
from mmcv.runner.dist_utils import master_only


@HOOKS.register_module()
class ModelCheckpoint(Hook):
    def __init__(self,
                 monitor='loss',
                 min_delta=0,
                 mode='min',
                 by_epoch=True):
        super().__init__()

        self.monitor = monitor
        self.by_epoch = by_epoch
        self.min_delta = abs(min_delta)

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            NotImplementedError('ModelCheckpoint mode is available of `min` or `max`.'
                                'Not implement auto mode like Keras.')
        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def before_run(self, runner):
        if runner.meta is None:
            warnings.warn('runner.meta is None. Creating an empty one.')
            runner.meta = dict()
        runner.meta.setdefault('hook_msgs', dict())
        self.best_ckpt_path = runner.meta['hook_msgs'].get(
            'model_ckpt', None)
        # Allow instances to be re-used
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def after_val_epoch(self, runner):
        # TODO: val_lossがTextLoggerと一致しない(train_lossはこれで一致するものが取れるので、おそらくval_lossを取れている)
        key_score = self._get_monitor_value(runner)
        if key_score is None:
            return

        runner._epoch -= 1  # runner.epoch += 1 has been done before val workflow, so back one epochs
        if self._is_improvement(key_score, self.best):
            self.best = key_score
            self._save_checkpoint(runner)
        runner._epoch += 1

    def _get_monitor_value(self, runner):
        # only get value on val mode
        if runner.mode == 'val':
            if self.monitor_op == np.less:
                # mode=='min'
                return runner.outputs['log_vars'].get(self.monitor)
            else:
                # mode=='max'
                return runner.meta['hook_msgs']['eval_res'].get(self.monitor)

    def _is_improvement(self, monitor_value, reference_value):
        return self.monitor_op(monitor_value - self.min_delta, reference_value)

    @master_only
    def _save_checkpoint(self, runner):
        """Save the current checkpoint and delete unwanted checkpoint."""
        out_dir = runner.work_dir

        if self.by_epoch:
            current = f'epoch_{runner.epoch + 1}'
            cur_type, cur_time = 'epoch', runner.epoch + 1
        else:
            current = f'iter_{runner.iter + 1}'
            cur_type, cur_time = 'iter', runner.iter + 1

        if self.best_ckpt_path and osp.isfile(self.best_ckpt_path):
            os.remove(self.best_ckpt_path)
            runner.logger.info(
                (f'The previous best checkpoint {self.best_ckpt_path} was '
                    'removed'))

        best_ckpt_name = f'best_{self.monitor}_{current}.pth'
        self.best_ckpt_path = osp.join(out_dir, best_ckpt_name)
        runner.meta['hook_msgs']['model_ckpt'] = self.best_ckpt_path

        runner.save_checkpoint(
            out_dir,
            filename_tmpl=best_ckpt_name,
            save_optimizer=True)
        runner.logger.info(
            f'Now best checkpoint is saved as {best_ckpt_name}.')
        runner.logger.info(
            f'Best {self.monitor} is {self.best:0.4f} '
            f'at {cur_time} {cur_type}.')

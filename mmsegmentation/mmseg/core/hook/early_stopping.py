from mmcv.runner import HOOKS, Hook
from mmcv.runner.dist_utils import master_only
import os.path as osp
import warnings
import numpy as np


@HOOKS.register_module()
class EarlyStopping(Hook):
    def __init__(self,
                 monitor='loss',
                 min_delta=0,
                 patience=0,
                 mode='min',
                 baseline=None,
                 restore_last_weights=True):
        super().__init__()

        self.monitor = monitor
        self.patience = patience
        self.baseline = baseline
        self.min_delta = abs(min_delta)
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_last_weights = restore_last_weights

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            NotImplementedError('EarlyStopping mode is available of `min` or `max`.'
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
            'last_ckpt', None)
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def after_val_epoch(self, runner):
        epoch = runner.epoch  # runner.epoch += 1 has been done before val workflow
        # TODO: val_lossがTextLoggerと一致しない(train_lossはこれで一致するものが取れるので、おそらくval_lossを取れている)
        current = self._get_monitor_value(runner)
        if current is None:
            return

        self.wait += 1
        if self._is_improvement(current, self.best):
            self.best = current
            # Only restart wait if we beat both the baseline and our previous best.
            if self.baseline is None or self._is_improvement(current, self.baseline):
                self.wait = 0

        # Only check after the first epoch.
        if self.wait >= self.patience and epoch > 0:
            runner.logger.info(f'Epoch {epoch}: early stopping')
            if self.restore_last_weights:
                runner.logger.info(f'Saving last checkpoint at {epoch} epochs')
                self._save_checkpoint(runner)
            # change condition of while loop(`while self.epoch < self._max_epochs:`) to stop runner
            runner._max_epochs = -1

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
        runner._epoch -= 1  # runner.epoch += 1 has been done before val workflow, so back one epochs
        runner.save_checkpoint(out_dir, save_optimizer=True)
        if runner.meta is not None:
            runner.meta['hook_msgs']['last_ckpt'] = osp.join(
                out_dir, f'epoch_{runner.epoch + 1}.pth')
        runner._epoch += 1

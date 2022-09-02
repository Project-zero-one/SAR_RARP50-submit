from typing import Union
from mmcv.runner import HOOKS, Hook
import warnings
import optuna
import numpy as np


@HOOKS.register_module()
class OptunaPruningHook(Hook):

    def __init__(self, trial: optuna.trial.Trial, monitor: Union[str, list]) -> None:
        self._trial = trial
        if isinstance(monitor, str):
            monitor = [monitor]
        self.monitor = monitor

    def after_val_epoch(self, runner):
        epoch = runner.epoch
        current_score = np.mean([
            runner.meta['hook_msgs']['eval_res'].get(key, np.nan)
            for key in self.monitor])

        if np.isnan(current_score):
            message = (
                "The metric '{}' is not in the evaluation logs for pruning. "
                "Please make sure you set the correct metric name.".format(self.monitor)
            )
            warnings.warn(message)
            return

        self._trial.report(current_score, step=epoch)
        if self._trial.should_prune():
            message = "Trial was pruned at epoch {}.".format(epoch)
            raise optuna.TrialPruned(message)

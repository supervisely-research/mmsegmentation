from mmcv.transforms import Compose
from mmengine.hooks import Hook

from mmseg.registry import HOOKS


@HOOKS.register_module()
class PipelineSwitchHook(Hook):

    def __init__(self, switch_epoch, switch_pipeline):
        self.switch_epoch = switch_epoch
        self.switch_pipeline = switch_pipeline
        self._restart_dataloader = False
        self._has_switched = False

    def before_train_epoch(self, runner):
        epoch = runner.epoch
        train_loader = runner.train_dataloader
        if epoch >= self.switch_epoch and not self._has_switched:
            runner.logger.info('Switch pipeline now!')
            train_loader.dataset.pipeline = Compose(self.switch_pipeline)
            if hasattr(train_loader, 'persistent_workers'
                       ) and train_loader.persistent_workers is True:
                train_loader._DataLoader__initialized = False
                train_loader._iterator = None
                self._restart_dataloader = True
            self._has_switched = True
        else:
            if self._restart_dataloader:
                train_loader._DataLoader__initialized = True
from mmengine.hooks import Hook
from mmengine.runner import Runner

from mmseg.datasets.online_training_dataset import OnlineTrainingDataset
from mmseg.registry import HOOKS


class BaseOnlinePolicy(Hook):

    def __init__(self):
        self._runner: 'Runner' = None
        self._train_dataset: OnlineTrainingDataset = None

    def before_run(self, runner: 'Runner'):
        self._runner = runner
        self._train_dataset: OnlineTrainingDataset = runner.train_loop.dataloader.dataset
        assert isinstance(self._train_dataset, OnlineTrainingDataset), \
            'The training dataset must be an instance of OnlineTrainingDataset.'

    def add_sample(self, img_info: dict, seg_map_path: str) -> int:
        return self._train_dataset.add_sample(img_info, seg_map_path)

    @property
    def iter(self):
        return self._runner.iter
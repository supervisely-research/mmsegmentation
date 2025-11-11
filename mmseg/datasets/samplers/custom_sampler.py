from typing import Iterator, Optional, Sized
from mmengine.dist import get_dist_info, sync_random_seed
from mmengine.dataset.sampler import DefaultSampler
from mmseg.registry import DATA_SAMPLERS
import math

@DATA_SAMPLERS.register_module()
class CustomSampler(DefaultSampler):
    
    def __init__(self,
                 dataset: Sized,
                 shuffle: bool = True,
                 seed: Optional[int] = None,
                 round_up: bool = True) -> None:
        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size

        self.dataset = dataset
        self.shuffle = shuffle
        if seed is None:
            seed = sync_random_seed()
        self.seed = seed
        self.epoch = 0
        self.round_up = round_up

    @property
    def num_samples(self) -> int:
        if self.round_up:
            return math.ceil(len(self.dataset) / self.world_size)
        else:
            return math.ceil((len(self.dataset) - self.rank) / self.world_size)

    @property
    def total_size(self) -> int:
        if self.round_up:
            return self.num_samples * self.world_size
        else:
            return len(self.dataset)
import copy
from mmseg.registry import DATASETS
from .tomato import TomatoDataset


@DATASETS.register_module()
class AdaptiveTomatoDataset(TomatoDataset):
    """Adaptive learning dataset for tomato segmentation.
    
    Gradually increases training samples during training.
    
    Args:
        max_samples (int): Maximum number of samples to use
        initial_samples (int): Number of samples to start with
        samples_per_stage (int): Number of samples to add per stage
    """
    
    def __init__(self,
                 max_samples=5,
                 initial_samples=2,
                 samples_per_stage=1,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.max_samples = max_samples
        self.initial_samples = initial_samples
        self.samples_per_stage = samples_per_stage
        self.current_samples = initial_samples
        
        # Store full dataset
        self._full_data_list = copy.deepcopy(self.data_list)
        
        # Initialize with subset
        self._update_dataset()
    
    def _update_dataset(self):
        """Update active dataset based on current_samples."""
        num_samples = min(self.current_samples, len(self._full_data_list))
        self.data_list = self._full_data_list[:num_samples]
    
    def add_samples(self):
        """Add more samples to the dataset."""
        if self.current_samples < self.max_samples:
            self.current_samples = min(
                self.current_samples + self.samples_per_stage,
                self.max_samples
            )
            self._update_dataset()
            return True
        return False
    
    def get_current_stage(self):
        """Get current stage number."""
        return (self.current_samples - self.initial_samples) // self.samples_per_stage + 1

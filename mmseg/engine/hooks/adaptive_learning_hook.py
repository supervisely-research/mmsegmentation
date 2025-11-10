from mmengine.hooks import Hook
from mmseg.registry import HOOKS


@HOOKS.register_module()
class AdaptiveLearningHook(Hook):
    """Hook for adaptive learning - adds samples during training.
    
    Args:
        iters_per_stage (int): Number of iterations before adding samples
        save_checkpoint (bool): Save checkpoint when adding samples
    """
    
    def __init__(self, iters_per_stage=100, save_checkpoint=True):
        self.iters_per_stage = iters_per_stage
        self.save_checkpoint = save_checkpoint
    
    def before_train(self, runner):
        """Log initial dataset size."""
        dataset = runner.train_dataloader.dataset
        if hasattr(dataset, 'current_samples'):
            runner.logger.info(
                f'AdaptiveLearning: Starting with {dataset.current_samples} samples'
            )
    
    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        """Check if it's time to add samples."""
        dataset = runner.train_dataloader.dataset
        
        # Check if dataset supports adaptive learning
        if not hasattr(dataset, 'add_samples'):
            return
        
        # Check if it's time to add samples
        if (runner.iter + 1) % self.iters_per_stage == 0:
            added = dataset.add_samples()
            
            if added:
                stage = dataset.get_current_stage()
                runner.logger.info(
                    f'\n{"="*60}\n'
                    f'AdaptiveLearning: Stage {stage} - '
                    f'Added samples (now using {dataset.current_samples}/{dataset.max_samples})\n'
                    f'{"="*60}\n'
                )
                
                # Save checkpoint if requested
                if self.save_checkpoint:
                    runner.logger.info('Saving checkpoint after adding samples...')
                    # Checkpoint will be saved by CheckpointHook

from collections import OrderedDict

from mmengine.dist import get_dist_info
from mmengine.hooks import Hook
from torch import nn

from mmseg.registry import HOOKS
from mmseg.utils import all_reduce_dict


def get_norm_states(module: nn.Module) -> OrderedDict:
    async_norm_states = OrderedDict()
    for name, child in module.named_modules():
        if isinstance(child, nn.modules.batchnorm._NormBase):
            for k, v in child.state_dict().items():
                async_norm_states['.'.join([name, k])] = v
    return async_norm_states


@HOOKS.register_module()
class SyncNormHook(Hook):

    def before_val_epoch(self, runner):
        module = runner.model
        _, world_size = get_dist_info()
        if world_size == 1:
            return
        norm_states = get_norm_states(module)
        if len(norm_states) == 0:
            return
        norm_states = all_reduce_dict(norm_states, op='mean')
        module.load_state_dict(norm_states, strict=False)
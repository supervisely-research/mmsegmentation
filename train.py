import os.path as osp
import torch
from mmengine.runner.checkpoint import CheckpointLoader
from mmengine.config import Config
from mmengine.runner import Runner


BASE_CONFIG_FILE = "configs/mask2former/custom/swin-l-online-app.py"


@CheckpointLoader.register_scheme(prefixes='', force=True)
def load_from_local(filename, map_location):
    filename = osp.expanduser(filename)
    if not osp.isfile(filename):
        raise FileNotFoundError(f'{filename} can not be found.')
    checkpoint = torch.load(filename, map_location=map_location, weights_only=False)
    return checkpoint


if __name__ == "__main__":
    cfg = Config.fromfile(BASE_CONFIG_FILE)
    runner = Runner.from_cfg(cfg)
    runner.train()
import os.path as osp
import torch
import os
from mmengine.runner.checkpoint import CheckpointLoader
from mmengine.config import Config
from mmengine.runner import Runner
from mmseg.engine.hooks.online_training_api import OnlineTrainingAPI


os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'


BASE_CONFIG_FILE = "configs/mask2former/custom/swin-l-online-app.py"
RESUME_FROM_CHECKPOINT = None
API_HOST = "0.0.0.0"
API_PORT = 8000
INITIAL_SAMPLES = 2


@CheckpointLoader.register_scheme(prefixes='', force=True)
def load_from_local(filename, map_location):
    filename = osp.expanduser(filename)
    if not osp.isfile(filename):
        raise FileNotFoundError(f'{filename} can not be found.')
    checkpoint = torch.load(filename, map_location=map_location, weights_only=False)
    return checkpoint


if __name__ == "__main__":
    cfg = Config.fromfile(BASE_CONFIG_FILE)

    data_root = 'app_data/test_project/'
    work_dir = 'app_data/test_project/experiments/'
    
    os.makedirs(work_dir, exist_ok=True)

    cfg.work_dir = work_dir
    cfg.data_root = data_root
    cfg.train_cfg.max_iters = 1000
    cfg.default_hooks.checkpoint.interval = 100
    cfg.log_level = 'INFO'
    cfg.env_cfg = dict(dist_cfg=dict(backend='nccl'))

    if RESUME_FROM_CHECKPOINT is not None:
        cfg.resume = True
        cfg.load_from = RESUME_FROM_CHECKPOINT
        print(f"🔄 Resuming training from checkpoint: {RESUME_FROM_CHECKPOINT}")

    print("⚙️ Initializing MMEngine Runner...")
    runner = Runner.from_cfg(cfg)
    
    custom_hook = OnlineTrainingAPI(
        start_samples=INITIAL_SAMPLES,
        data_root=data_root,
        host=API_HOST,
        port=API_PORT
    )
    runner.register_custom_hooks([custom_hook])

    print("🚀 Starting training loop...")
    print(f"   Send samples to: POST http://{API_HOST}:{API_PORT}/add_sample")
    print(f"   Format: {{\"image\": \"<base64>\", \"mask\": \"<base64>\"}}")
    
    runner.train()
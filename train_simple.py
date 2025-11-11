import os
import os.path as osp
import torch
from pathlib import Path
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


def add_initial_samples(dataset, data_root, num_samples=2):
    img_dir = Path(data_root) / 'images/train/images/training'
    mask_dir = Path(data_root) / 'images/train/annotations/training'
    
    img_files = sorted(list(img_dir.glob('*.jpeg')) + list(img_dir.glob('*.JPG')))[:num_samples]
    
    print(f"Adding {len(img_files)} samples to dataset...")
    
    for img_path in img_files:
        mask_filename = img_path.stem + '.png'
        mask_path = mask_dir / mask_filename
        
        if not mask_path.exists():
            print(f"Warning: mask not found for {img_path.name}, skipping")
            continue
        
        from PIL import Image
        img = Image.open(img_path)
        width, height = img.size
        
        img_info = {
            'file_name': str(img_path),
            'width': width,
            'height': height,
        }
        
        sample_idx = dataset.add_sample(img_info, str(mask_path))
        print(f"  Added sample {sample_idx}: {img_path.name}")
    
    print(f"Dataset size: {len(dataset)}")


if __name__ == "__main__":
    cfg = Config.fromfile(BASE_CONFIG_FILE)

    work_dir = 'work_dirs/test_simple/'
    cfg.work_dir = work_dir
    
    cfg.train_cfg.max_iters = 100
    cfg.default_hooks.checkpoint.interval = 20
    cfg.default_hooks.logger.interval = 10

    print("⚙️ Initializing MMEngine Runner...")
    runner = Runner.from_cfg(cfg)
    
    dataset = runner.train_dataloader.dataset
    add_initial_samples(dataset, cfg.data_root, num_samples=3)
    
    train_loop = runner.train_loop
    dataloader_cfg = runner.cfg.train_dataloader.copy()
    dataloader_cfg['dataset'] = dataset
    new_dataloader = runner.build_dataloader(dataloader_cfg, seed=runner.seed)
    train_loop.dataloader = new_dataloader

    print("🚀 Starting training loop...")
    runner.train()
import os
import os.path as osp
import torch
import time
from pathlib import Path
from PIL import Image
from mmengine.runner.checkpoint import CheckpointLoader
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.model.base_module import BaseModule


os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'


original_dump = BaseModule._dump_init_info
def safe_dump(self, logger_name=None):
    pass
BaseModule._dump_init_info = safe_dump


BASE_CONFIG_FILE = "configs/mask2former/custom/swin-l-online-app.py"


@CheckpointLoader.register_scheme(prefixes='', force=True)
def load_from_local(filename, map_location):
    filename = osp.expanduser(filename)
    if not osp.isfile(filename):
        raise FileNotFoundError(f'{filename} can not be found.')
    checkpoint = torch.load(filename, map_location=map_location, weights_only=False)
    return checkpoint


def add_samples_from_disk(dataset, data_root, sample_indices):
    img_dir = Path(data_root) / 'images/train/images/training'
    mask_dir = Path(data_root) / 'images/train/annotations/training'
    
    img_files = sorted(list(img_dir.glob('*.jpeg')) + list(img_dir.glob('*.JPG')))
    
    for idx in sample_indices:
        if idx >= len(img_files):
            print(f"⚠️  Index {idx} out of range, skipping")
            continue
        
        img_path = img_files[idx]
        mask_path = mask_dir / (img_path.stem + '.png')
        
        if not mask_path.exists():
            print(f"⚠️  Mask not found for {img_path.name}, skipping")
            continue
        
        img = Image.open(img_path)
        width, height = img.size
        
        img_info = {
            'file_name': str(img_path),
            'width': width,
            'height': height,
        }
        
        sample_idx = dataset.add_sample(img_info, str(mask_path))
        print(f"  ✅ Added sample {sample_idx}: {img_path.name}")
    
    return len(dataset)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧪 Simple Incremental Training Test (No FastAPI)")
    print("="*70)
    
    cfg = Config.fromfile(BASE_CONFIG_FILE)

    data_root = Path('/root/data')
    work_dir = 'work_dirs/test_simple_incremental/'
    os.makedirs(work_dir, exist_ok=True)

    classes = ['background', 'Core', 'Locule', 'Navel', 'Pericarp', 'Placenta', 'Septum', 'Tomato']
    num_classes = len(classes)
    palette = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], 
               [255, 255, 0], [255, 0, 255], [0, 255, 255], [255, 128, 0]]
    metainfo = dict(classes=classes, palette=palette)

    cfg.work_dir = work_dir
    cfg.data_root = str(data_root)
    cfg.classes = classes
    cfg.num_classes = num_classes
    cfg.metainfo = metainfo
    cfg.train_cfg.max_iters = 150
    cfg.default_hooks.checkpoint.interval = 50
    cfg.default_hooks.logger.interval = 10
    cfg.log_level = 'INFO'
    
    cfg.model.decode_head.num_classes = num_classes
    cfg.model.decode_head.loss_cls = dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        class_weight=[1.0] * num_classes + [0.1]
    )
    
    cfg.train_dataloader.dataset.metainfo = metainfo
    cfg.train_dataloader.dataset.data_root = str(data_root)

    print(f"📋 Classes: {num_classes} classes")
    print(f"📁 Data root: {data_root}")
    print(f"💾 Work dir: {work_dir}")
    
    print("\n⚙️  Initializing MMEngine Runner...")
    runner = Runner.from_cfg(cfg)
    
    dataset = runner.train_dataloader.dataset
    
    print("\n📥 Adding initial 3 samples...")
    add_samples_from_disk(dataset, data_root, [0, 1, 2])
    print(f"   Dataset size: {len(dataset)}")
    
    train_loop = runner.train_loop
    dataloader_cfg = runner.cfg.train_dataloader.copy()
    dataloader_cfg['dataset'] = dataset
    new_dataloader = runner.build_dataloader(dataloader_cfg, seed=runner.seed)
    train_loop.dataloader = new_dataloader
    
    print("\n🚀 Starting training...")
    print("="*70 + "\n")
    
    runner.train()
    
    print("\n" + "="*70)
    print("✅ Test completed successfully!")
    print(f"   Final dataset size: {len(dataset)} samples")
    print("="*70)
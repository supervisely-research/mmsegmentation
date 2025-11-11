import os
import os.path as osp
import torch
import time
import threading
from pathlib import Path
from mmengine.runner.checkpoint import CheckpointLoader
from mmengine.config import Config
from mmengine.runner import Runner
from mmseg.engine.hooks.online_training_api import OnlineTrainingAPI


os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'

BASE_CONFIG_FILE = "configs/mask2former/custom/swin-l-online-app.py"
RESUME_FROM_CHECKPOINT = None
API_HOST = "0.0.0.0"
API_PORT = 8000
INITIAL_SAMPLES = 2
MAX_ITERS = 200


@CheckpointLoader.register_scheme(prefixes='', force=True)
def load_from_local(filename, map_location):
    filename = osp.expanduser(filename)
    if not osp.isfile(filename):
        raise FileNotFoundError(f'{filename} can not be found.')
    checkpoint = torch.load(filename, map_location=map_location, weights_only=False)
    return checkpoint


def add_samples_incrementally():
    time.sleep(15)
    
    print("\n" + "="*70)
    print("📥 Starting incremental sample addition...")
    print("="*70)
    
    import requests
    import base64
    from PIL import Image
    
    data_root = Path('/root/data')
    img_dir = data_root / 'images/train/images/training'
    mask_dir = data_root / 'images/train/annotations/training'
    
    img_files = sorted(list(img_dir.glob('*.jpeg')) + list(img_dir.glob('*.JPG')))[3:6]
    
    for i, img_path in enumerate(img_files, 1):
        mask_path = mask_dir / (img_path.stem + '.png')
        
        if not mask_path.exists():
            print(f"⚠️  Mask not found: {mask_path.name}, skipping")
            continue
        
        with open(img_path, 'rb') as f:
            image_b64 = base64.b64encode(f.read()).decode('utf-8')
        
        with open(mask_path, 'rb') as f:
            mask_b64 = base64.b64encode(f.read()).decode('utf-8')
        
        payload = {
            "image": image_b64,
            "mask": mask_b64,
            "filename": img_path.name
        }
        
        try:
            print(f"\n📤 Sending sample {i}/3: {img_path.name}")
            response = requests.post(f"http://{API_HOST}:{API_PORT}/add_sample", json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Sample added! Dataset size: {result.get('dataset_size')}")
            else:
                print(f"❌ Failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        time.sleep(20)
    
    print("\n" + "="*70)
    print("✅ Incremental addition completed!")
    print("="*70)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 Online Training FastAPI Test")
    print("="*70)
    
    cfg = Config.fromfile(BASE_CONFIG_FILE)

    data_root = Path('/root/data')
    work_dir = 'work_dirs/test_api/'
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
    cfg.train_cfg.max_iters = MAX_ITERS
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

    print(f"📋 Classes: {classes}")
    print(f"📁 Data root: {data_root}")
    print(f"💾 Work dir: {work_dir}")
    print(f"🔢 Max iterations: {MAX_ITERS}")
    
    if RESUME_FROM_CHECKPOINT is not None:
        cfg.resume = True
        cfg.load_from = RESUME_FROM_CHECKPOINT
        print(f"🔄 Resuming from: {RESUME_FROM_CHECKPOINT}")

    print("\n⚙️  Initializing MMEngine Runner...")
    runner = Runner.from_cfg(cfg)
    
    print("🔧 Registering OnlineTrainingAPI hook...")
    custom_hook = OnlineTrainingAPI(
        start_samples=INITIAL_SAMPLES,
        data_root=str(data_root),
        host=API_HOST,
        port=API_PORT
    )
    runner.register_custom_hooks([custom_hook])

    print("\n🧵 Starting incremental sample addition thread...")
    adder_thread = threading.Thread(target=add_samples_incrementally, daemon=True)
    adder_thread.start()
        
    print("\n🏁 Starting training loop...")
    print(f"   API Server: http://{API_HOST}:{API_PORT}")
    print(f"   Initial samples needed: {INITIAL_SAMPLES}")
    print(f"   Max iterations: {MAX_ITERS}")
    print("="*70 + "\n")
    
    runner.train()
    
    print("\n" + "="*70)
    print("✅ Test completed successfully!")
    print("="*70)
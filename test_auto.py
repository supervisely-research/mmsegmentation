import os
import os.path as osp
import torch
import time
import requests
import base64
from pathlib import Path
from PIL import Image
from mmengine.runner.checkpoint import CheckpointLoader
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.model.base_module import BaseModule
from mmseg.engine.hooks.online_training_api import OnlineTrainingAPI


os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'

def safe_dump(self, logger_name=None):
    pass
BaseModule._dump_init_info = safe_dump


BASE_CONFIG_FILE = "configs/mask2former/custom/swin-l-online-app.py"
API_HOST = "127.0.0.1"
API_PORT = 8001
API_URL = f"http://{API_HOST}:{API_PORT}"


@CheckpointLoader.register_scheme(prefixes='', force=True)
def load_from_local(filename, map_location):
    filename = osp.expanduser(filename)
    if not osp.isfile(filename):
        raise FileNotFoundError(f'{filename} can not be found.')
    checkpoint = torch.load(filename, map_location=map_location, weights_only=False)
    return checkpoint


def send_sample(img_path, mask_path, delay=5):
    time.sleep(delay)
    
    print(f"\n📤 Preparing {img_path.name}...")
    
    if not img_path.exists():
        print(f"❌ Image not found: {img_path}")
        return False
    
    if not mask_path.exists():
        print(f"❌ Mask not found: {mask_path}")
        return False
    
    print(f"   Reading image: {img_path.stat().st_size} bytes")
    with open(img_path, 'rb') as f:
        img_bytes = f.read()
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
    
    print(f"   Reading mask: {mask_path.stat().st_size} bytes")
    with open(mask_path, 'rb') as f:
        mask_bytes = f.read()
        mask_b64 = base64.b64encode(mask_bytes).decode('utf-8')
    
    print(f"   Image base64: {len(img_b64)} chars")
    print(f"   Mask base64: {len(mask_b64)} chars")
    
    payload = {
        "image": img_b64,
        "mask": mask_b64,
        "filename": img_path.name
    }
    
    print(f"   Payload keys: {list(payload.keys())}")
    print(f"   Sending to {API_URL}/add_sample...")
    
    try:
        response = requests.post(f"{API_URL}/add_sample", json=payload, timeout=30)
        print(f"   Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Added! Dataset size: {result.get('dataset_size')}")
            return True
        else:
            print(f"❌ Failed {response.status_code}")
            print(f"   Response: {response.text[:500]}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 Automatic FastAPI Test (All-in-One)")
    print("="*70)
    
    cfg = Config.fromfile(BASE_CONFIG_FILE)

    data_root = Path('/root/data')
    work_dir = 'work_dirs/test_api_auto/'
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
    cfg.train_cfg.max_iters = 100
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

    print(f"📋 Classes: {num_classes}")
    print(f"📁 Data root: {data_root}")
    print(f"💾 Work dir: {work_dir}")
    
    print("\n⚙️  Initializing Runner...")
    runner = Runner.from_cfg(cfg)
    
    print("🔧 Registering OnlineTrainingAPI hook...")
    custom_hook = OnlineTrainingAPI(
        start_samples=2,
        data_root=str(data_root),
        host=API_HOST,
        port=API_PORT
    )
    runner.register_custom_hooks([custom_hook])

    print(f"\n🌐 API will start on {API_URL}")
    print("="*70)
    
    img_dir = data_root / 'images/train/images/training'
    mask_dir = data_root / 'images/train/annotations/training'
    img_files = sorted(list(img_dir.glob('*.jpeg')) + list(img_dir.glob('*.JPG')))[:3]
    
    import threading
    
    def auto_add_samples():
        print("\n🤖 Auto-sender: waiting 10 seconds for server startup...")
        time.sleep(10)
        
        for i, img_path in enumerate(img_files, 1):
            mask_path = mask_dir / (img_path.stem + '.png')
            if not mask_path.exists():
                print(f"⚠️  Mask not found: {mask_path.name}")
                continue
            
            print(f"\n🤖 Auto-sender: sending sample {i}/{len(img_files)}")
            send_sample(img_path, mask_path, delay=0)
            
            if i < len(img_files):
                print(f"🤖 Auto-sender: waiting 15 seconds before next sample...")
                time.sleep(15)
    
    sender_thread = threading.Thread(target=auto_add_samples, daemon=True)
    sender_thread.start()
    
    print("\n🏁 Starting training (this will block until samples arrive)...")
    print("="*70 + "\n")
    
    runner.train()
    
    print("\n" + "="*70)
    print("✅ Test completed!")
    print("="*70)
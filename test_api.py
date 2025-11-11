import requests
import base64
import time
from pathlib import Path
from PIL import Image
import numpy as np
import cv2


API_URL = "http://localhost:8000"


def encode_image(image_path):
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def encode_mask(mask_path):
    with open(mask_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def test_health():
    print("\n" + "="*70)
    print("Testing /health endpoint...")
    print("="*70)
    
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200


def test_status():
    print("\n" + "="*70)
    print("Testing /status endpoint...")
    print("="*70)
    
    response = requests.get(f"{API_URL}/status")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200


def test_add_sample(image_path, mask_path, filename=None):
    print("\n" + "="*70)
    print(f"Testing /add_sample with {image_path.name}...")
    print("="*70)
    
    image_b64 = encode_image(image_path)
    mask_b64 = encode_mask(mask_path)
    
    payload = {
        "image": image_b64,
        "mask": mask_b64,
        "filename": filename or image_path.name
    }
    
    response = requests.post(f"{API_URL}/add_sample", json=payload)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Sample added successfully!")
        print(f"   Sample index: {result.get('sample_index')}")
        print(f"   Dataset size: {result.get('dataset_size')}")
        print(f"   Iteration: {result.get('metadata', {}).get('iteration')}")
        return True
    else:
        print(f"❌ Failed: {response.text}")
        return False


def test_predict(image_path):
    print("\n" + "="*70)
    print(f"Testing /predict with {image_path.name}...")
    print("="*70)
    
    image_b64 = encode_image(image_path)
    
    payload = {
        "image": image_b64
    }
    
    response = requests.post(f"{API_URL}/predict", json=payload)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Prediction successful!")
        print(f"   Status: {result.get('status')}")
        print(f"   Unique labels: {result.get('unique_labels')}")
        print(f"   Iteration: {result.get('metadata', {}).get('iteration')}")
        return True
    else:
        print(f"❌ Failed: {response.text}")
        return False


def run_full_test():
    print("\n" + "="*70)
    print("🚀 Starting Full FastAPI Test")
    print("="*70)
    
    data_root = Path('/root/data')
    img_dir = data_root / 'images/train/images/training'
    mask_dir = data_root / 'images/train/annotations/training'
    
    img_files = sorted(list(img_dir.glob('*.jpeg')) + list(img_dir.glob('*.JPG')))[:5]
    
    if not img_files:
        print("❌ No images found in", img_dir)
        return
    
    print(f"\n📁 Found {len(img_files)} images to test")
    
    time.sleep(3)
    
    if not test_health():
        print("❌ Health check failed!")
        return
    
    if not test_status():
        print("❌ Status check failed!")
        return
    
    print("\n" + "="*70)
    print("📥 Testing incremental sample addition...")
    print("="*70)
    
    for i, img_path in enumerate(img_files[:3], 1):
        mask_path = mask_dir / (img_path.stem + '.png')
        
        if not mask_path.exists():
            print(f"⚠️  Mask not found: {mask_path.name}, skipping")
            continue
        
        print(f"\n--- Adding sample {i}/3 ---")
        success = test_add_sample(img_path, mask_path)
        
        if not success:
            print(f"❌ Failed to add sample {i}")
            continue
        
        print(f"⏳ Waiting 10 seconds for training to process...")
        time.sleep(10)
    
    print("\n" + "="*70)
    print("🔮 Testing prediction...")
    print("="*70)
    
    test_img = img_files[3] if len(img_files) > 3 else img_files[0]
    test_predict(test_img)
    
    print("\n" + "="*70)
    print("✅ Full test completed!")
    print("="*70)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "health":
            test_health()
        elif sys.argv[1] == "status":
            test_status()
        elif sys.argv[1] == "add" and len(sys.argv) > 3:
            test_add_sample(Path(sys.argv[2]), Path(sys.argv[3]))
        elif sys.argv[1] == "predict" and len(sys.argv) > 2:
            test_predict(Path(sys.argv[2]))
        else:
            print("Usage:")
            print("  python test_api_client.py              # Run full test")
            print("  python test_api_client.py health       # Test health")
            print("  python test_api_client.py status       # Test status")
            print("  python test_api_client.py add <img> <mask>")
            print("  python test_api_client.py predict <img>")
    else:
        run_full_test()
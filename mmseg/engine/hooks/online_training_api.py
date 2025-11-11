import torch
import cv2
import logging
from pathlib import Path
from typing import Optional
import base64
import io
from PIL import Image
import numpy as np
import time

from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.logging import print_log
from mmseg.apis import inference_model
from mmseg.datasets.online_training_dataset import OnlineTrainingDataset
from mmseg.registry import HOOKS

from mmseg.online_training.request_queue import RequestQueue, RequestType
from mmseg.online_training.api_server import start_api_server


@HOOKS.register_module()
class OnlineTrainingAPI(Hook):
    
    priority = 'VERY_LOW'
    
    def __init__(
        self,
        start_samples: int = 3,
        data_root: Optional[str] = None,
        host='0.0.0.0',
        port=8000
    ):
        super().__init__()
        self.request_queue = RequestQueue()
        self.api_thread = start_api_server(
            request_queue=self.request_queue,
            host=host,
            port=port
        )
        self.data_root = Path(data_root) if data_root else None
        self.image_counter = 0
        self.start_samples = start_samples
    
    @property
    def _dataset(self) -> OnlineTrainingDataset:
        return self.train_loop.dataloader.dataset
    
    def before_train(self, runner: Runner):
        self.train_loop = runner.train_loop
        
        if len(self._dataset) < self.start_samples:
            print_log(
                f"\n{'='*70}\n"
                f"⏳ Dataset is empty! Waiting for samples to be added via API...\n"
                f"{'='*70}\n"
                f"   Please send samples to: POST /add_sample\n"
                f"{'='*70}",
                logger='current',
                level=logging.WARNING
            )
            
            max_wait_time = 3600
            check_interval = 1
            elapsed_time = 0

            while len(self._dataset) < self.start_samples and elapsed_time < max_wait_time:
                if not self.request_queue.is_empty():
                    print_log(
                        "📨 Processing add_sample request...",
                        logger='current',
                        level=logging.INFO
                    )
                    self._process_pending_requests(runner)
                
                if len(self._dataset) < self.start_samples:
                    time.sleep(check_interval)
                    elapsed_time += check_interval
                    print(f"Still waiting... ({len(self._dataset)}/{self.start_samples} samples)")
                    
            if len(self._dataset) < self.start_samples:
                raise RuntimeError(
                    f"Training cannot start: Dataset is still empty after "
                    f"waiting {max_wait_time} seconds. Please add samples via "
                    f"POST /add_sample endpoint."
                )
            
            print_log(
                f"\n{'='*70}\n"
                f"✅ Dataset initialized with {len(self._dataset)} sample(s)!\n"
                f"{'='*70}\n",
                logger='current',
                level=logging.INFO
            )
            
            self._rebuild_dataloader(runner)
        
        else:
            print_log(
                f"✅ Dataset ready with {len(self._dataset)} samples",
                logger='current',
                level=logging.INFO
            )
    
    def before_train_iter(
        self,
        runner: Runner,
        batch_idx: int,
        data_batch=None
    ):
        self._process_pending_requests(runner)
    
    def before_val_iter(
        self,
        runner: Runner,
        batch_idx: int,
        data_batch=None
    ):
        self._process_pending_requests(runner)
    
    def _process_pending_requests(self, runner: Runner):
        if self.request_queue.is_empty():
            return
        
        requests = self.request_queue.get_all()
        if not requests:
            return
        
        print_log(
            f"\n{'='*70}\n"
            f"📨 Processing {len(requests)} API request(s) at iteration {runner.iter}\n"
            f"{'='*70}",
            logger='current',
            level=logging.INFO
        )
        
        needs_dataloader_rebuild = False
        
        for request_type, request_data, future in requests:
            try:
                if request_type == RequestType.PREDICT:
                    result = self._handle_inference(runner, request_data)
                    future.set_result(result)
                
                elif request_type == RequestType.ADD_SAMPLE:
                    result = self._handle_add_sample(runner, request_data)
                    future.set_result(result)
                    needs_dataloader_rebuild = True
                
            except Exception as e:
                print_log(
                    f"❌ Request failed: {e}",
                    logger='current',
                    level=logging.ERROR
                )
                future.set_exception(e)
                import traceback
                traceback.print_exc()

        if needs_dataloader_rebuild:
            self._rebuild_dataloader(runner)
        
        print_log(
            f"{'='*70}\n"
            f"✅ Requests processed, resuming training\n"
            f"{'='*70}\n",
            logger='current',
            level=logging.INFO
        )
    
    @torch.no_grad()
    def _handle_inference(self, runner: Runner, request_data: dict) -> dict:
        image_bytes = base64.b64decode(request_data['image'])
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_np = np.array(image)
        
        model = runner.model
        was_training = model.training
        model.eval()
        
        try:
            result = inference_model(model, image_np)

            pred_sem_seg = result.pred_sem_seg.data[0].cpu().numpy()
            
            unique_labels = np.unique(pred_sem_seg).tolist()
            
            label_counts = {}
            for label in unique_labels:
                if label > 0:
                    label_counts[int(label)] = int(np.sum(pred_sem_seg == label))
            
            return {
                'status': 'success',
                'segmentation_map': pred_sem_seg.tolist(),
                'unique_labels': unique_labels,
                'label_pixel_counts': label_counts,
                'metadata': {
                    'iteration': runner.iter,
                    'epoch': runner.epoch,
                    'image_shape': pred_sem_seg.shape
                }
            }
        
        finally:
            if was_training:
                model.train()
    
    def _handle_add_sample(self, runner: Runner, request_data: dict) -> dict:
        train_loop = runner.train_loop
        dataset = train_loop.dataloader.dataset
        
        if not hasattr(dataset, 'add_sample'):
            raise RuntimeError(
                "Dataset must have 'add_sample' method. "
                "Please use OnlineTrainingDataset."
            )
        
        image_bytes = base64.b64decode(request_data['image'])
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        if self.data_root is None:
            self.data_root = Path(runner.work_dir) / "data"
        
        image_dir = self.data_root / "online_images"
        mask_dir = self.data_root / "online_masks"
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        
        self.image_counter += 1
        filename = request_data.get('filename') or f'online_{self.image_counter:06d}.jpg'
        image_path = image_dir / filename
        
        image.save(image_path)
        
        width, height = image.size
        img_info = {
            'id': self.image_counter,
            'file_name': str(image_path),
            'width': width,
            'height': height,
        }
        
        mask_data = request_data['mask']
        mask = self._decode_mask(mask_data, (width, height))
        mask = self._validate_mask(mask, img_info)
        
        mask_filename = Path(filename).stem + '.png'
        mask_path = mask_dir / mask_filename
        cv2.imwrite(str(mask_path), mask)
        
        dataset: 'OnlineTrainingDataset'
        sample_idx = dataset.add_sample(img_info, str(mask_path))
        
        print_log(
            f"✅ Added sample {sample_idx}: {filename}, "
            f"dataset size: {len(dataset)}",
            logger='current',
            level=logging.INFO
        )
        
        return {
            'status': 'success',
            'sample_index': sample_idx,
            'image_info': img_info,
            'dataset_size': len(dataset),
            'metadata': {
                'iteration': runner.iter,
                'epoch': runner.epoch
            }
        }
    
    def _decode_mask(self, mask_data: str, img_size: tuple) -> np.ndarray:
        mask_bytes = base64.b64decode(mask_data)
        mask_image = Image.open(io.BytesIO(mask_bytes))
        mask = np.array(mask_image)
        
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        
        if mask.shape != (img_size[1], img_size[0]):
            mask = cv2.resize(mask, img_size, interpolation=cv2.INTER_NEAREST)
        
        return mask.astype(np.uint8)
    
    def _validate_mask(self, mask: np.ndarray, img_info: dict) -> np.ndarray:
        if mask.shape[0] != img_info['height'] or mask.shape[1] != img_info['width']:
            raise ValueError(
                f"Mask shape {mask.shape} does not match image size "
                f"({img_info['width']}, {img_info['height']})"
            )
        
        return mask
    
    def _rebuild_dataloader(self, runner: Runner):
        print_log(
            "🔄 Rebuilding dataloader to propagate dataset changes...",
            logger='current',
            level=logging.INFO
        )
        
        train_loop = runner.train_loop
        
        current_dataset = train_loop.dataloader.dataset
        
        dataloader_cfg = runner.cfg.train_dataloader.copy()
        dataloader_cfg['dataset'] = current_dataset
        
        new_dataloader = runner.build_dataloader(
            dataloader_cfg,
            seed=runner.seed,
        )
        
        old_dataloader = train_loop.dataloader
        train_loop.dataloader = new_dataloader
        
        train_loop._max_iters = train_loop._max_epochs * len(new_dataloader)
        
        del old_dataloader
        
        print_log(
            f"✅ Dataloader rebuilt: {len(new_dataloader)} batches per epoch",
            logger='current',
            level=logging.INFO
        )
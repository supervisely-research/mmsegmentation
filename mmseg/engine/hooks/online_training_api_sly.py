import torch
import cv2
from pathlib import Path
from PIL import Image
import numpy as np
import time

from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmseg.apis import inference_model
from mmseg.datasets.online_training_dataset import OnlineTrainingDataset
from mmseg.engine.hooks.loss_plateau_detector import LossPlateauDetector
from mmseg.registry import HOOKS
from mmseg.online_training.request_queue import RequestQueue, RequestType
from mmseg.online_training.dataset_utils import (
    sly_to_mask,
    validate_mask
)
from mmseg.online_training.inference_utils import get_test_pipeline, predictions_to_sly_figures
import supervisely as sly
from supervisely import logger

@HOOKS.register_module()
class OnlineTrainingAPISly(Hook):
    
    priority = 'VERY_LOW'
    
    def __init__(
        self,
        request_queue: RequestQueue,
        obj_classes: sly.ObjClassCollection,
        images_dir: str,
        start_samples: int = 2,
        warmup_iters: int = 50,
        score_thr: float = 0.3,
    ):
        super().__init__()
        self.request_queue = request_queue
        self.obj_classes = obj_classes
        self.classes = [cls.name for cls in obj_classes]
        self.class2idx = {cls_name: idx + 1 for idx, cls_name in enumerate(self.classes)}
        self.idx2class = {idx + 1: cls_name for idx, cls_name in enumerate(self.classes)}

        self.images_dir = Path(images_dir)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir = self.images_dir.parent / 'masks'
        self.masks_dir.mkdir(parents=True, exist_ok=True)
        
        self.start_samples = start_samples
        self.train_loop = None
        self.runner: Runner = None
        self._ready_to_predict = False
        self._iters_to_warmup = warmup_iters
        self._model_loaded = False
        self.img_ids_set = set()
        self.score_thr = score_thr
        self.plateau_detector = LossPlateauDetector(
            window_size=25,
            threshold=0.005,
            patience=3,
            check_interval=5,
            log_tensorboard=False,
        )
        self._samples_to_adapt = 12
        self._is_adapted = False
        self._loss = None
        self._is_paused = False
        self.test_pipeline = None

    def _init_with_runner(self, runner: Runner):
        self.train_loop = runner.train_loop
        self.runner = runner
        self._model_loaded = True
        self.test_pipeline = get_test_pipeline(runner)
        if not hasattr(runner.model, 'cfg'):
            runner.model.cfg = runner.cfg

    @property
    def _dataset(self) -> OnlineTrainingDataset:
        return self.train_loop.dataloader.dataset
    
    @property
    def iter(self) -> int:
        return self.runner.iter if self.runner else None
    
    def before_train(self, runner: Runner):
        self._init_with_runner(runner)
        self._wait_for_initial_samples(runner)

    def _wait_for_initial_samples(self, runner: Runner):
        if len(self._dataset) < self.start_samples:
            print(f"⏳ Dataset is empty! Waiting for samples to be added via API...")
            samples_needed = self.start_samples - len(self._dataset)
            self._is_paused = True
            self._wait_for_new_samples(runner, samples_needed=samples_needed, max_wait_time=3600)

            print(f"✅ Dataset initialized with {len(self._dataset)} sample(s)!")
            self._is_paused = False
        else:
            print(f"✅ Dataset ready with {len(self._dataset)} samples")
            
    def before_train_iter(
        self,
        runner: Runner,
        batch_idx: int,
        data_batch=None
    ):
        if runner.iter > self._iters_to_warmup:
            self._ready_to_predict = True
        self.process_pending_requests(runner)
    
    def before_val_iter(
        self,
        runner: Runner,
        batch_idx: int,
        data_batch=None
    ):
        self.process_pending_requests(runner)
    
    def after_train_iter(
        self,
        runner: Runner,
        batch_idx: int,
        data_batch: dict = None,
        outputs: dict = None
    ) -> None:
        loss = outputs['loss'].item()
        is_plateau = self.plateau_detector.step(loss, runner.iter)
        self._loss = loss
        if is_plateau:
            self._pause_on_plateau(runner)
            self.plateau_detector.reset()
    
    def _wait_for_new_samples(self, runner: Runner, samples_needed: int = 1, max_wait_time: int = None):
        sleep_interval = 0.5
        elapsed_time = 0
        samples_before = len(self._dataset)

        while len(self._dataset) - samples_before < samples_needed:
            if max_wait_time is not None and elapsed_time >= max_wait_time:
                raise RuntimeError(
                    f"Training cannot proceed: No new samples added after waiting {max_wait_time} seconds."
                )
            if not self.request_queue.is_empty():
                self.process_pending_requests(runner)
            
            if len(self._dataset) < self.start_samples:
                time.sleep(sleep_interval)
                elapsed_time += sleep_interval

    def _pause_on_plateau(self, runner: Runner):
        print(f"⚠️  Loss plateau detected at iteration {runner.iter}. Training will pause until new samples are added.")
        self._is_paused = True
        self._wait_for_new_samples(runner)
        print(f"✅ New samples added, resuming training.")
        self._is_paused = False

    def process_pending_requests(self, runner: Runner):
        if self.request_queue.is_empty():
            return
        
        requests = self.request_queue.get_all()
        if not requests:
            return

        print(f"📨 Processing {len(requests)} API request(s) at iteration {runner.iter}")
        
        new_samples_added = False
        
        for request_type, request_data, future in requests:
            try:
                if request_type == RequestType.PREDICT:
                    result = self.handle_predict(runner, request_data)
                    future.set_result(result)
                
                elif request_type == RequestType.ADD_SAMPLE:
                    result = self.handle_add_sample(runner, request_data)
                    future.set_result(result)
                    new_samples_added = True

                elif request_type == RequestType.STATUS:
                    result = self.status()
                    future.set_result(result)

                print(f"✅ Requests processed, resuming training")

            except Exception as e:
                import traceback
                logger.error(f"❌ Error processing request {request_type}: {e}", exc_info=False)
                traceback.print_exc()
                future.set_exception(e)

        if new_samples_added:
            self._rebuild_dataloader(runner)
            self._adapt_loss_plateau_detector()
            self.plateau_detector.reset()

    @torch.no_grad()
    def handle_predict(self, runner: Runner, request_data: dict) -> dict:
        if not self._ready_to_predict:
            logger.warning(
                f"⚠️ Model not ready for predictions yet. "
                f"Current iteration: {self.iter}, warmup required: {self._iters_to_warmup}. "
                f"Predictions may be unreliable."
            )
        image_np = np.array(request_data['image'])              
        orig_h, orig_w = image_np.shape[:2]
        
        
        image_np = np.array(request_data['image'])
        original_shape = image_np.shape[:2] 
        
        model = runner.model
        model.cfg = runner.cfg
        was_training = model.training
        model.eval()
        
        try:
            result = inference_model(model, image_np)
            
            objects = predictions_to_sly_figures(
                result,
                self.score_thr,
                self.idx2class,
                self.obj_classes,
                original_shape=original_shape 
            )
            
            return {
                'objects': objects,
                'image_id': request_data['image_id'],
                'status': self.status(),
            }
        finally:
            if was_training:
                model.train()
                
    def handle_add_sample(self, runner: Runner, request_data: dict) -> dict:
        dataset: 'OnlineTrainingDataset' = self._dataset
        img_id = request_data['image_id']
        image_np = request_data['image']
        ann = request_data['annotation']
        filename = request_data['image_name']
        if img_id in self.img_ids_set:
            logger.warning(f"⚠️ Image ID {img_id} already exists in dataset, skipping duplicate.")
            raise ValueError(f"Image ID {img_id} already exists in dataset.")
        self.img_ids_set.add(img_id)

        orig_h, orig_w = image_np.shape[:2]
    
        image = Image.fromarray(image_np).convert('RGB')

        image_path = self.images_dir / filename
        image.save(image_path)
        
        width, height = image.size
        img_info = {
            'file_name': str(image_path.resolve()),
            'width': width,
            'height': height,
        }

        mask = sly_to_mask(ann, (width, height), self.class2idx)
        mask = cv2.resize(mask, (target_w, TARGET_HEIGHT), interpolation=cv2.INTER_NEAREST)
        mask = validate_mask(mask, img_info, len(self.classes) + 1)
        
        mask_filename = Path(filename).stem + '.png'
        mask_path = self.masks_dir / mask_filename

        mask = mask.astype(np.uint8)
        cv2.imwrite(str(mask_path), mask)
                
        sample_idx = dataset.add_sample(img_info, str(mask_path))
        
        print(
            f"➕ Added sample {filename} with mask. "
            f"{len(dataset)} total samples in dataset."
        )
        
        return {
            'status': self.status(),
        }

    def status(self) -> dict:
        status = {
            'iteration': self.iter,
            'loss': self._loss,
            'training_paused': self._is_paused,
            'dataset_size': len(self._dataset),
            'ready_to_predict': self._ready_to_predict,
        }
        return status
    
    def _rebuild_dataloader(self, runner: Runner):
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
        del old_dataloader
        logger.debug(
            f"🔄 Dataloader rebuilt: {len(new_dataloader)} batches per epoch"
        )
    
    def _adapt_loss_plateau_detector(self):
        if len(self._dataset) >= self._samples_to_adapt and not self._is_adapted:
            print(f"🔧 Adapting LossPlateauDetector parameters for dataset size {len(self._dataset)}")
            self.plateau_detector.patience = 5
            self.plateau_detector.window_size = 40
            self.plateau_detector.threshold = 0.003
            self.plateau_detector.check_interval = 10
            self.plateau_detector.reset()
            self._is_adapted = True
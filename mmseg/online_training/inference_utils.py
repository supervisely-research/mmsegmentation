import base64
import zlib
import cv2
from copy import deepcopy
from io import BytesIO
import numpy as np
from PIL import Image
from mmcv.transforms import Compose
from mmengine.runner import Runner
import supervisely as sly
from supervisely.annotation.label import LabelingStatus
from mmseg.structures import SegDataSample


def get_test_pipeline(runner: Runner) -> Compose:
    test_pipeline = deepcopy(runner.cfg.test_pipeline)
    test_pipeline[0].type = 'mmseg.LoadImageFromNDArray'
    return Compose(test_pipeline)


def mask_to_sly_bitmap(mask: np.ndarray, origin: list) -> str:
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    pil_img = Image.fromarray(mask_uint8, mode='L')
    img_stream = BytesIO()
    pil_img.save(img_stream, format='PNG')
    png_bytes = img_stream.getvalue()
    compressed_data = zlib.compress(png_bytes)
    bitmap_data = base64.b64encode(compressed_data).decode('utf-8')
    return bitmap_data


def predictions_to_sly_figures(
        result: SegDataSample,
        score_thr: float,
        idx2class: dict,
        obj_classes: sly.ObjClassCollection
    ) -> list:
    pred_sem_seg = result.pred_sem_seg.data[0].cpu().numpy()
    
    objects = []
    unique_labels = np.unique(pred_sem_seg)
    
    for label_idx in unique_labels:
        if label_idx == 0:
            continue
        
        class_name = idx2class.get(label_idx)
        if class_name is None:
            continue
            
        obj_class = obj_classes.get(class_name)
        if obj_class is None:
            continue
        
        class_mask = (pred_sem_seg == label_idx).astype(np.uint8)
        
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) < 10:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            mask_crop = class_mask[y:y+h, x:x+w]
            
            bitmap_data = mask_to_sly_bitmap(mask_crop, [x, y])
            
            bitmap = sly.Bitmap(
                data=bitmap_data,
                origin=sly.PointLocation(y, x)
            )
            
            label = sly.Label(
                bitmap, obj_class, status=LabelingStatus.AUTO
            )
            objects.append(label.to_json())
    
    return objects
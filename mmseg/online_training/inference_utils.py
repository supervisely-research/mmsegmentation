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


# def mask_to_sly_bitmap(mask: np.ndarray, origin: list) -> str:
#     mask_uint8 = (mask > 0).astype(np.uint8) * 255
#     pil_img = Image.fromarray(mask_uint8, mode='L')
#     img_stream = BytesIO()
#     pil_img.save(img_stream, format='PNG')
#     png_bytes = img_stream.getvalue()
#     compressed_data = zlib.compress(png_bytes)
#     bitmap_data = base64.b64encode(compressed_data).decode('utf-8')
#     return bitmap_data


def predictions_to_sly_figures(
        result: SegDataSample,
        score_thr: float,
        idx2class: dict,
        obj_classes: sly.ObjClassCollection,
        original_shape: tuple = None
    ) -> list:
    pred_sem_seg = result.pred_sem_seg.data[0].cpu().numpy()
    pred_height, pred_width = pred_sem_seg.shape
    
    # Calculate scale factors if original shape provided
    if original_shape is not None:
        orig_height, orig_width = original_shape
        scale_y = orig_height / pred_height
        scale_x = orig_width / pred_width
    else:
        scale_y = scale_x = 1.0
    
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
            
            # Scale coordinates back to original size
            x_orig = int(x * scale_x)
            y_orig = int(y * scale_y)
            w_orig = int(w * scale_x)
            h_orig = int(h * scale_y)
            
            # Scale mask as well
            mask_crop = class_mask[y:y+h, x:x+w]
            if scale_x != 1.0 or scale_y != 1.0:
                mask_crop = cv2.resize(
                    mask_crop, 
                    (w_orig, h_orig), 
                    interpolation=cv2.INTER_NEAREST
                )
            
            bitmap = sly.Bitmap(
                data=mask_crop,
                origin=sly.PointLocation(y_orig, x_orig)
            )
            
            label = sly.Label(
                bitmap, obj_class, status=LabelingStatus.AUTO
            )
            objects.append(label.to_json())
    

    return objects
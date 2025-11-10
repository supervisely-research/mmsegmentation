import base64
import zlib
import numpy as np
from io import BytesIO
from PIL import Image


def decode_sly_bitmap_png(bitmap_data):
    compressed_data = base64.b64decode(bitmap_data)
    png_bytes = zlib.decompress(compressed_data)
    img_stream = BytesIO(png_bytes)
    pil_img = Image.open(img_stream)
    bitmap_mask = np.array(pil_img)
    
    if len(bitmap_mask.shape) == 3:
        bitmap_mask = bitmap_mask[:, :, 0]
    
    return (bitmap_mask > 0).astype(np.uint8)


def sly_to_mask(sly_ann: dict, img_size: tuple, class_to_id: dict) -> np.ndarray:
    mask = np.zeros((img_size[1], img_size[0]), dtype=np.uint8)
    
    for obj in sly_ann.get('objects', []):
        class_title = obj.get('classTitle', '')
        geom_type = obj.get('geometryType', '')
        
        if geom_type == 'bitmap' and 'bitmap' in obj:
            bitmap_info = obj['bitmap']
            if 'data' in bitmap_info:
                origin = bitmap_info.get('origin', [0, 0])
                bitmap_mask = decode_sly_bitmap_png(bitmap_info['data'])
                
                if bitmap_mask is not None:
                    class_id = class_to_id.get(class_title, 0)
                    if class_id > 0:
                        offset_x, offset_y = origin
                        h, w = bitmap_mask.shape
                        
                        y_start = max(0, offset_y)
                        x_start = max(0, offset_x)
                        y_end = min(img_size[1], offset_y + h)
                        x_end = min(img_size[0], offset_x + w)
                        
                        src_y_start = max(0, -offset_y)
                        src_x_start = max(0, -offset_x)
                        src_y_end = src_y_start + (y_end - y_start)
                        src_x_end = src_x_start + (x_end - x_start)
                        
                        if y_end > y_start and x_end > x_start:
                            final_mask = np.zeros((img_size[1], img_size[0]), dtype=np.uint8)
                            final_mask[y_start:y_end, x_start:x_end] = \
                                bitmap_mask[src_y_start:src_y_end, src_x_start:src_x_end]
                            
                            object_mask = (final_mask > 0).astype(np.uint8) * class_id
                            mask = np.maximum(mask, object_mask)
    
    return mask


def validate_mask(mask: np.ndarray, img_info: dict, num_classes: int) -> np.ndarray:
    if mask.shape[0] != img_info['height'] or mask.shape[1] != img_info['width']:
        raise ValueError(f"Mask shape {mask.shape} does not match image size ({img_info['width']}, {img_info['height']})")
    
    if mask.max() >= num_classes:
        raise ValueError(f"Mask contains invalid class ID {mask.max()}, but num_classes={num_classes}")
    
    return mask
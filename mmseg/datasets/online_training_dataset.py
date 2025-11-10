import copy
import os.path as osp
from typing import List, Union

from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset


@DATASETS.register_module()
class OnlineTrainingDataset(BaseSegDataset):
    
    METAINFO: dict = dict()

    def __init__(self, *args, **kwargs):
        kwargs['serialize_data'] = False
        super().__init__(*args, **kwargs)
        self._img_counter = 0

    def load_data_list(self) -> List[dict]:
        return []

    def add_sample(self, img_info: dict, seg_map_path: str) -> int:
        img_info['img_id'] = self._img_counter
        self._img_counter += 1
        
        data_info = self.parse_data_info({
            'raw_img_info': img_info,
            'raw_seg_map_path': seg_map_path
        })
        self.data_list.append(data_info)
        return img_info['img_id']

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        img_info = raw_data_info['raw_img_info']
        seg_map_path = raw_data_info['raw_seg_map_path']

        data_info = {}

        img_path = osp.join(self.data_prefix['img_path'], img_info['file_name'])
        data_info['img_path'] = img_path
        data_info['img_id'] = img_info['img_id']
        data_info['seg_map_path'] = seg_map_path
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']
        data_info['label_map'] = None
        data_info['reduce_zero_label'] = self.reduce_zero_label
        data_info['seg_fields'] = []

        return data_info
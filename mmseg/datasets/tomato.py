from pathlib import Path
from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS


@DATASETS.register_module()
class TomatoDataset(BaseSegDataset):
    """Tomato segmentation dataset with 8 classes.
    
    Classes:
        0: background
        1: Core
        2: Locule
        3: Navel
        4: Pericarp
        5: Placenta
        6: Septum
        7: Tomato
    """
    
    METAINFO = {
        'classes': (
            'background', 'Core', 'Locule', 'Navel',
            'Pericarp', 'Placenta', 'Septum', 'Tomato'
        ),
        'palette': [
            [0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255],
            [255, 255, 0], [255, 0, 255], [0, 255, 255], [255, 128, 0]
        ]
    }

    def __init__(self, **kwargs):
        super().__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs
        )
    
    def load_data_list(self):
        """Load annotation from directory.
        
        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []
        img_dir = Path(self.data_prefix['img_path'])
        ann_dir = Path(self.data_prefix['seg_map_path'])
        
        # Search for image files with different extensions
        for img_path in sorted(img_dir.glob('*')):
            if img_path.suffix.lower() in ['.jpg', '.jpeg']:
                # Find corresponding annotation
                seg_path = ann_dir / f'{img_path.stem}.png'
                
                if seg_path.exists():
                    data_info = dict(
                        img_path=str(img_path),
                        seg_map_path=str(seg_path),
                        label_map=None,
                        reduce_zero_label=self.reduce_zero_label,
                        seg_fields=[]
                    )
                    data_list.append(data_info)
        
        return data_list
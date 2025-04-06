"""
coco_utils: A collection of utilities for working with COCO datasets.
"""

__version__ = "0.1.0"

from .coco_file_utils import (
    merge_coco_files,
    split_coco_dataset
)

from .coco_labels_utils import (
    shrink_bbox,
    shrink_coco_bboxes
)

from .coco_viz_utils import (
    find_image_info,
    find_annotations,
    visualize_bbox,
    visualize_keypoints
)

__all__ = [
    'merge_coco_files',
    'split_coco_dataset',
    'shrink_bbox',
    'shrink_coco_bboxes',
    'find_image_info',
    'find_annotations',
    'visualize_bbox',
    'visualize_keypoints',
    '__version__'
]

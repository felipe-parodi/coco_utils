"""COCO Refinement tools for annotation editing and visualization."""

from .refine_boxes import COCORefinementGUI
from .refine_utils import (
    KEYPOINT_COLOR_CENTER,
    KEYPOINT_COLOR_LEFT,
    KEYPOINT_COLOR_RIGHT,
    SKELETON_COLOR_CENTER,
    SKELETON_COLOR_LEFT,
    SKELETON_COLOR_MIXED,
    SKELETON_COLOR_RIGHT,
    get_keypoint_color,
    get_skeleton_color,
    load_and_organize_coco_json,
)

__all__ = [
    "COCORefinementGUI",
    "get_keypoint_color",
    "get_skeleton_color",
    "load_and_organize_coco_json",
    "KEYPOINT_COLOR_LEFT",
    "KEYPOINT_COLOR_RIGHT",
    "KEYPOINT_COLOR_CENTER",
    "SKELETON_COLOR_LEFT",
    "SKELETON_COLOR_RIGHT",
    "SKELETON_COLOR_CENTER",
    "SKELETON_COLOR_MIXED",
]

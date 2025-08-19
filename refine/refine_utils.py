"""Utility functions for COCO refinement GUI.

This module contains helper functions for color mapping, data loading,
and organization used by the COCO refinement GUI.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Color constants for keypoint and skeleton visualization
KEYPOINT_COLOR_LEFT = (0, 255, 0)  # Green for left keypoints
KEYPOINT_COLOR_RIGHT = (255, 0, 0)  # Blue for right keypoints
KEYPOINT_COLOR_CENTER = (0, 255, 255)  # Yellow for center keypoints
SKELETON_COLOR_LEFT = (0, 255, 0)  # Green for left-side skeleton links
SKELETON_COLOR_RIGHT = (255, 0, 0)  # Blue for right-side skeleton links
SKELETON_COLOR_CENTER = (0, 255, 255)  # Yellow for center skeleton links
SKELETON_COLOR_MIXED = (255, 0, 255)  # Magenta for mixed left/right links


def get_keypoint_color(keypoint_name: str) -> tuple:
    """Get color for keypoint based on name.

    Args:
        keypoint_name: Name of the keypoint.

    Returns:
        Tuple representing BGR color for the keypoint.
    """
    keypoint_name_lower = keypoint_name.lower()

    # Center keypoints (nose, eyes, ears)
    if any(center_kpt in keypoint_name_lower for center_kpt in ["nose", "eye", "ear"]):
        return KEYPOINT_COLOR_CENTER
    # Left keypoints
    elif "left" in keypoint_name_lower or "l_" in keypoint_name_lower:
        return KEYPOINT_COLOR_LEFT
    # Right keypoints
    elif "right" in keypoint_name_lower or "r_" in keypoint_name_lower:
        return KEYPOINT_COLOR_RIGHT
    else:
        return KEYPOINT_COLOR_CENTER  # Default fallback


def get_skeleton_color(keypoint1_name: str, keypoint2_name: str) -> tuple:
    """Get color for skeleton link based on connected keypoint names.

    Args:
        keypoint1_name: Name of the first keypoint.
        keypoint2_name: Name of the second keypoint.

    Returns:
        Tuple representing BGR color for the skeleton link.
    """
    color1 = get_keypoint_color(keypoint1_name)
    color2 = get_keypoint_color(keypoint2_name)

    # If both keypoints are the same type (left/right/center), use that color
    if color1 == color2:
        return color1
    # Otherwise use mixed color for connections between different types
    else:
        return SKELETON_COLOR_MIXED


def load_and_organize_coco_json(
    coco_json_path: str, valid_ids: Optional[Set[int]] = None, is_video_mode: bool = False
) -> Tuple[Optional[Dict[int, List[Dict]]], Optional[List[Dict]], int]:
    """Load and organize annotations from a COCO JSON file.

    Args:
        coco_json_path: Path to the COCO JSON file.
        valid_ids: Set of valid image/frame IDs to include.
        is_video_mode: Flag to indicate if IDs are frame numbers.

    Returns:
        Tuple containing:
            - annotations_by_image: Dictionary mapping image IDs to annotations.
            - categories: List of category dictionaries.
            - total_num_annotations: Total number of annotations loaded.
    """
    print(f"Loading COCO annotations from: {coco_json_path}")
    cleaned_coco_path = coco_json_path.strip("\"'")

    try:
        with open(cleaned_coco_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find the file: {cleaned_coco_path}")
        return None, None, 0
    except json.JSONDecodeError as e:
        print(f"Error: Could not parse JSON from {cleaned_coco_path}: {e}")
        return None, None, 0

    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    if not annotations:
        print("Warning: No annotations found in the JSON file.")
        return {}, categories, 0

    annotations_by_image = {}
    total_annotations = 0
    skipped_annotations = 0

    for ann in annotations:
        image_id = ann.get("image_id")

        # Skip if valid_ids is specified and image_id is not in it
        if valid_ids is not None and image_id not in valid_ids:
            skipped_annotations += 1
            continue

        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []

        annotations_by_image[image_id].append(ann)
        total_annotations += 1

    if valid_ids and skipped_annotations > 0:
        mode_str = "frame" if is_video_mode else "image"
        print(f"Filtered {skipped_annotations} annotations not matching valid {mode_str} IDs.")

    print(
        f"Loaded {total_annotations} annotations across {len(annotations_by_image)} images/frames."
    )

    if categories:
        print(f"Found {len(categories)} categories.")
        for cat in categories[:3]:  # Show first 3 categories
            cat_name = cat.get("name", "unknown")
            cat_id = cat.get("id", "unknown")
            print(f"  - Category {cat_id}: {cat_name}")
        if len(categories) > 3:
            print(f"  ... and {len(categories) - 3} more")

    return annotations_by_image, categories, total_annotations


def load_coco_data(
    json_path: str, images_dir: Optional[str] = None, mode: str = "directory"
) -> Tuple[
    Optional[List[np.ndarray]],
    Optional[Dict[int, List[Dict]]],
    Optional[List[str]],
    Optional[List[int]],
    Optional[List[Dict]],
]:
    """Load COCO data from directory or video.

    Args:
        json_path: Path to COCO JSON file.
        images_dir: Directory containing images (for directory mode).
        mode: Either "directory" or "video".

    Returns:
        Tuple containing:
            - frames: List of loaded images/frames.
            - annotations_by_image: Dictionary mapping image IDs to annotations.
            - frame_paths: List of frame file paths.
            - frame_ids: List of frame IDs.
            - categories: List of category dictionaries.
    """
    # Load annotations
    if mode == "directory":
        annotations_by_image, categories, _ = load_and_organize_coco_json(json_path)

        if annotations_by_image is None:
            return None, None, None, None, None

        if not images_dir:
            print("Error: images_dir is required for directory mode")
            return None, None, None, None, None

        # Load images from directory
        frame_paths = sorted(
            [
                os.path.join(images_dir, f)
                for f in os.listdir(images_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )

        if not frame_paths:
            print(f"No images found in {images_dir}")
            return None, None, None, None, None

        frames = []
        frame_ids = []

        for i, path in enumerate(frame_paths):
            img = cv2.imread(path)
            if img is not None:
                frames.append(img)
                # Use index as frame ID for directory mode
                frame_ids.append(i)
            else:
                print(f"Warning: Could not load image {path}")

        return frames, annotations_by_image, frame_paths, frame_ids, categories

    else:  # video mode
        print("Video mode not implemented in this utility function")
        return None, None, None, None, None


def load_video_data(json_path: str, video_path: str, sample_rate: int = 1) -> Tuple[
    Optional[List[np.ndarray]],
    Optional[Dict[int, List[Dict]]],
    Optional[List[str]],
    Optional[List[int]],
    Optional[List[Dict]],
]:
    """Load COCO data from video file.

    Args:
        json_path: Path to COCO JSON file.
        video_path: Path to video file.
        sample_rate: Sample every nth frame.

    Returns:
        Tuple containing:
            - frames: List of loaded video frames.
            - annotations_by_frame: Dictionary mapping frame IDs to annotations.
            - frame_paths: List of frame identifiers.
            - frame_numbers: List of frame numbers.
            - categories: List of category dictionaries.
    """
    print(f"Loading video from: {video_path}")
    print(f"Sample rate: {sample_rate} (keeping every {sample_rate} frame(s))")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return None, None, None, None, None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Video info: {total_frames} frames at {fps:.2f} FPS")
    print(f"Duration: {total_frames/fps:.2f} seconds")

    # Collect frame numbers to load
    frame_numbers_to_load = list(range(0, total_frames, sample_rate))
    valid_frame_ids = set(frame_numbers_to_load)

    # Load annotations (filtering by valid frame IDs)
    annotations_by_frame, categories, _ = load_and_organize_coco_json(
        json_path, valid_ids=valid_frame_ids, is_video_mode=True
    )

    if annotations_by_frame is None:
        cap.release()
        return None, None, None, None, None

    # Load frames
    frames = []
    frame_paths = []
    frame_numbers = []

    for frame_num in frame_numbers_to_load:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()

        if ret:
            frames.append(frame)
            frame_paths.append(f"{video_path}:frame_{frame_num}")
            frame_numbers.append(frame_num)
        else:
            print(f"Warning: Could not read frame {frame_num}")

    cap.release()

    print(f"Loaded {len(frames)} frames from video")
    print(f"Annotations available for {len(annotations_by_frame)} frames")

    return frames, annotations_by_frame, frame_paths, frame_numbers, categories


def load_results_video_data(
    original_json_path: str,
    shrunk_json_path: str,
    video_path: str,
    sample_rate: int = 1,
    shrink_percentage: Optional[float] = None,
) -> Tuple[
    Optional[List[np.ndarray]],
    Optional[Dict[int, List[Dict]]],
    Optional[Dict[int, List[Dict]]],
    Optional[List[str]],
    Optional[List[int]],
    Optional[List[Dict]],
]:
    """Load original and shrunk COCO data from video for comparison.

    Args:
        original_json_path: Path to original COCO JSON file.
        shrunk_json_path: Path to shrunk COCO JSON file.
        video_path: Path to video file.
        sample_rate: Sample every nth frame.
        shrink_percentage: Shrink percentage to apply (if not loading pre-shrunk).

    Returns:
        Tuple containing:
            - frames: List of loaded video frames.
            - original_annotations: Dictionary of original annotations.
            - shrunk_annotations: Dictionary of shrunk annotations.
            - frame_paths: List of frame identifiers.
            - frame_numbers: List of frame numbers.
            - categories: List of category dictionaries.
    """
    print(f"Loading comparison data from video: {video_path}")

    # Load video frames
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return None, None, None, None, None, None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Video info: {total_frames} frames at {fps:.2f} FPS")

    # Collect frame numbers to load
    frame_numbers_to_load = list(range(0, total_frames, sample_rate))
    valid_frame_ids = set(frame_numbers_to_load)

    # Load original annotations
    original_annotations, categories, _ = load_and_organize_coco_json(
        original_json_path, valid_ids=valid_frame_ids, is_video_mode=True
    )

    if original_annotations is None:
        cap.release()
        return None, None, None, None, None, None

    # Load or generate shrunk annotations
    if shrunk_json_path and os.path.exists(shrunk_json_path):
        print(f"Loading pre-shrunk annotations from: {shrunk_json_path}")
        shrunk_annotations, _, _ = load_and_organize_coco_json(
            shrunk_json_path, valid_ids=valid_frame_ids, is_video_mode=True
        )
    else:
        # Generate shrunk annotations if needed
        if shrink_percentage:
            print(f"Generating shrunk annotations with {shrink_percentage}% shrinkage")
            # Import here to avoid circular dependency
            from coco_utils.coco_labels_utils import calculate_shrunk_bboxes

            # Create a temporary COCO data structure
            temp_coco = {"annotations": [], "categories": categories, "images": []}

            # Flatten annotations
            for frame_id, anns in original_annotations.items():
                temp_coco["annotations"].extend(anns)

            # Calculate shrunk bboxes
            shrunk_coco, _ = calculate_shrunk_bboxes(temp_coco, shrink_percentage)

            # Reorganize shrunk annotations
            shrunk_annotations = {}
            for ann in shrunk_coco.get("annotations", []):
                frame_id = ann.get("image_id")
                if frame_id not in shrunk_annotations:
                    shrunk_annotations[frame_id] = []
                shrunk_annotations[frame_id].append(ann)
        else:
            print("No shrunk data available")
            shrunk_annotations = {}

    # Load frames
    frames = []
    frame_paths = []
    frame_numbers = []

    for frame_num in frame_numbers_to_load:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()

        if ret:
            frames.append(frame)
            frame_paths.append(f"{video_path}:frame_{frame_num}")
            frame_numbers.append(frame_num)

    cap.release()

    print(f"Loaded {len(frames)} frames")
    print(f"Original annotations: {len(original_annotations)} frames")
    print(f"Shrunk annotations: {len(shrunk_annotations)} frames")

    return frames, original_annotations, shrunk_annotations, frame_paths, frame_numbers, categories

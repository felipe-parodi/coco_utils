"""COCO Visualization Utilities.

This module provides functions to visualize COCO dataset annotations,
including bounding boxes and keypoints, on their corresponding images.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from .exceptions import ImageNotFoundError, VisualizationError
from .logger import get_logger

logger = get_logger(__name__)


def find_image_info(coco_data: Dict[str, Any], image_id: int) -> Optional[Dict[str, Any]]:
    """Find the image dictionary for a given image ID.

    Args:
        coco_data: The loaded COCO dataset dictionary containing 'images' key.
        image_id: The ID of the image to find.

    Returns:
        The dictionary containing image information, or None if not found.

    Examples:
        >>> coco_data = {"images": [{"id": 1, "file_name": "img.jpg"}]}
        >>> find_image_info(coco_data, 1)
        {"id": 1, "file_name": "img.jpg"}
    """
    for img_info in coco_data.get("images", []):
        if img_info.get("id") == image_id:
            return img_info
    return None


def find_annotations(coco_data: Dict[str, Any], image_id: int) -> List[Dict[str, Any]]:
    """Find all annotations for a given image ID.

    Args:
        coco_data: The loaded COCO dataset dictionary containing 'annotations' key.
        image_id: The ID of the image whose annotations are sought.

    Returns:
        A list of annotation dictionaries for the given image_id.
        Returns empty list if no annotations found.

    Examples:
        >>> coco_data = {"annotations": [{"id": 1, "image_id": 1, "bbox": [10, 10, 50, 50]}]}
        >>> find_annotations(coco_data, 1)
        [{"id": 1, "image_id": 1, "bbox": [10, 10, 50, 50]}]
    """
    return [ann for ann in coco_data.get("annotations", []) if ann.get("image_id") == image_id]


def visualize_bbox(
    coco_data: Dict[str, Any],
    image_id: int,
    image_dir: Union[str, Path],
    box_color: str = "red",
    box_width: int = 3,
    figsize: Tuple[int, int] = (10, 10),
    show_plot: bool = True,
) -> Optional[Image.Image]:
    """Visualize all bounding boxes for a given image ID.

    Draws all bounding boxes found in annotations for the specified image
    and displays the result using matplotlib.

    Args:
        coco_data: Loaded COCO data as a dictionary containing 'images' and 'annotations'.
        image_id: The ID of the image to visualize.
        image_dir: Directory containing the image files referenced in coco_data.
        box_color: Color for the bounding box outlines. Default is "red".
        box_width: Width of the bounding box outlines in pixels. Default is 3.
        figsize: Figure size for matplotlib display as (width, height). Default is (10, 10).
        show_plot: If True, displays the plot. If False, returns the image. Default is True.

    Returns:
        PIL Image object with drawn bounding boxes if show_plot is False, None otherwise.

    Raises:
        ImageNotFoundError: If the image file cannot be found.
        VisualizationError: If visualization fails due to invalid data.

    Examples:
        >>> coco_data = load_coco_data("annotations.json")
        >>> visualize_bbox(coco_data, image_id=1, image_dir="./images")
    """
    image_dir = Path(image_dir)

    # Find image information
    img_info = find_image_info(coco_data, image_id)
    if not img_info:
        logger.warning(f"Image with ID {image_id} not found in coco_data['images']")
        raise VisualizationError(f"Image with ID {image_id} not found")

    # Get annotations
    annotations = find_annotations(coco_data, image_id)
    if not annotations:
        logger.info(f"No annotations found for image ID {image_id}")

    # Get image file path
    image_filename = img_info.get("file_name")
    if not image_filename:
        logger.error(f"'file_name' missing for image ID {image_id}")
        raise VisualizationError(f"'file_name' missing for image ID {image_id}")

    image_path = image_dir / image_filename

    # Load image
    try:
        img = Image.open(image_path).convert("RGB")
        logger.debug(f"Successfully loaded image from {image_path}")
    except FileNotFoundError:
        logger.error(f"Image file not found at '{image_path}'")
        raise ImageNotFoundError(f"Image file not found at '{image_path}'")
    except Exception as e:
        logger.error(f"Error loading image '{image_path}': {e}")
        raise VisualizationError(f"Error loading image: {e}")

    # Draw bounding boxes
    draw = ImageDraw.Draw(img)
    num_boxes_drawn = 0

    for annotation in annotations:
        ann_id = annotation.get("id", "N/A")
        bbox = annotation.get("bbox")  # COCO format: [x_min, y_min, width, height]

        if bbox and len(bbox) == 4:
            x_min, y_min, width, height = bbox

            # Convert to PIL format [x0, y0, x1, y1]
            x_max = x_min + width
            y_max = y_min + height

            draw.rectangle([x_min, y_min, x_max, y_max], outline=box_color, width=box_width)
            logger.debug(f"Drew bbox {bbox} for annotation ID {ann_id}")
            num_boxes_drawn += 1
        elif "bbox" in annotation:
            logger.warning(f"Invalid bounding box for annotation ID {ann_id}")

    logger.info(f"Drew {num_boxes_drawn} bounding box(es) for image ID {image_id}")

    # Display or return image
    if show_plot:
        plt.figure(figsize=figsize)
        plt.imshow(img)
        title = f"Image: {image_filename} (ID: {image_id}) | {num_boxes_drawn} Bounding Box(es)"
        plt.title(title)
        plt.axis("off")
        plt.show()
        return None
    else:
        return img


def visualize_keypoints(
    coco_data: Dict[str, Any],
    image_id: int,
    image_dir: Optional[Union[str, Path]] = None,
    image_root_dir: Optional[Union[str, Path]] = None,
    keypoint_color: Union[str, List[str]] = "blue",
    keypoint_radius: int = 5,
    skeleton_color: str = "yellow",
    skeleton_width: int = 2,
    figsize: Tuple[int, int] = (12, 12),
    show_plot: bool = True,
) -> Optional[Image.Image]:
    """Visualize keypoints and skeleton for a given image ID.

    Draws keypoints and skeleton connections (if defined) for all individuals
    detected in the image annotations.

    Args:
        coco_data: Loaded COCO data dictionary containing 'images', 'annotations', and 'categories'.
        image_id: The ID of the image to visualize.
        image_dir: Directory containing the image files. Either this or image_root_dir must be provided.
        image_root_dir: Root directory to search for images recursively. Either this or image_dir must be provided.
        keypoint_color: Color(s) for keypoints. Can be a single color or list of colors for multiple individuals.
        keypoint_radius: Radius of the drawn keypoints in pixels. Default is 5.
        skeleton_color: Color for the skeleton connections. Default is "yellow".
        skeleton_width: Width of the skeleton lines in pixels. Default is 2.
        figsize: Figure size for matplotlib display as (width, height). Default is (12, 12).
        show_plot: If True, displays the plot. If False, returns the image. Default is True.

    Returns:
        PIL Image object with drawn keypoints if show_plot is False, None otherwise.

    Raises:
        ValueError: If neither image_dir nor image_root_dir is provided.
        ImageNotFoundError: If the image file cannot be found.
        VisualizationError: If visualization fails due to invalid data.

    Examples:
        >>> coco_data = load_coco_data("keypoints.json")
        >>> visualize_keypoints(coco_data, image_id=1, image_dir="./images")
    """
    # Validate input
    if image_dir is None and image_root_dir is None:
        raise ValueError("Either 'image_dir' or 'image_root_dir' must be provided")

    # Convert to Path objects
    if image_dir is not None:
        image_dir = Path(image_dir)
    if image_root_dir is not None:
        image_root_dir = Path(image_root_dir)

    # Ensure keypoint_color is a list
    if isinstance(keypoint_color, str):
        keypoint_colors = [keypoint_color]
    else:
        keypoint_colors = keypoint_color

    # Find image information
    img_info = find_image_info(coco_data, image_id)
    if not img_info:
        logger.warning(f"Image with ID {image_id} not found")
        raise VisualizationError(f"Image with ID {image_id} not found")

    # Get annotations with keypoints
    annotations = find_annotations(coco_data, image_id)
    keypoint_annotations = [
        ann for ann in annotations if "keypoints" in ann and ann.get("num_keypoints", 0) > 0
    ]

    if not keypoint_annotations:
        logger.warning(f"No annotations with keypoints found for image ID {image_id}")
        # Still visualize the image without keypoints
        annotations = []

    # Get image file path
    image_filename = img_info.get("file_name")
    if not image_filename:
        logger.error(f"'file_name' missing for image ID {image_id}")
        raise VisualizationError(f"'file_name' missing for image ID {image_id}")

    # Resolve image path
    if image_root_dir:
        image_path = _find_image_recursively(image_root_dir, image_filename)
        if not image_path:
            logger.error(f"Image '{image_filename}' not found under '{image_root_dir}'")
            raise ImageNotFoundError(f"Image '{image_filename}' not found")
    else:
        image_path = image_dir / image_filename

    # Load image
    try:
        img = Image.open(image_path).convert("RGB")
        logger.debug(f"Successfully loaded image from {image_path}")
    except FileNotFoundError:
        logger.error(f"Image file not found at '{image_path}'")
        raise ImageNotFoundError(f"Image file not found at '{image_path}'")
    except Exception as e:
        logger.error(f"Error loading image '{image_path}': {e}")
        raise VisualizationError(f"Error loading image: {e}")

    draw = ImageDraw.Draw(img)

    # Get skeleton definition from categories
    skeleton = None
    keypoint_names = None
    if keypoint_annotations:
        category_id = keypoint_annotations[0].get("category_id")
        if category_id is not None:
            for cat in coco_data.get("categories", []):
                if cat.get("id") == category_id:
                    skeleton = cat.get("skeleton")  # List of [idx1, idx2] (1-based)
                    keypoint_names = cat.get("keypoints")  # List of keypoint names
                    logger.debug(
                        f"Found category '{cat.get('name', 'N/A')}' with "
                        f"{len(keypoint_names) if keypoint_names else 0} keypoints"
                    )
                    break

    # Draw keypoints for each individual
    for i, ann in enumerate(keypoint_annotations):
        keypoints = ann.get("keypoints")  # Format: [x1, y1, v1, x2, y2, v2, ...]
        if not keypoints:
            continue

        color = keypoint_colors[i % len(keypoint_colors)]
        ann_id = ann.get("id", "N/A")
        logger.debug(f"Drawing keypoints for annotation {ann_id} with color {color}")

        # Parse and draw keypoints
        kp_coords = []  # Store (x, y, visibility) for skeleton drawing
        for k in range(0, len(keypoints), 3):
            x, y, v = keypoints[k], keypoints[k + 1], keypoints[k + 2]
            kp_coords.append((x, y, v))

            # v=0: not labeled, v=1: labeled but not visible, v=2: labeled and visible
            if v > 0:  # Draw if labeled
                draw.ellipse(
                    (
                        x - keypoint_radius,
                        y - keypoint_radius,
                        x + keypoint_radius,
                        y + keypoint_radius,
                    ),
                    fill=color,
                    outline=color,
                )

        # Draw skeleton connections
        if skeleton and kp_coords:
            for connection in skeleton:
                # COCO uses 1-based indexing for skeleton
                idx1 = connection[0] - 1
                idx2 = connection[1] - 1

                if 0 <= idx1 < len(kp_coords) and 0 <= idx2 < len(kp_coords):
                    x1, y1, v1 = kp_coords[idx1]
                    x2, y2, v2 = kp_coords[idx2]

                    # Draw connection if both keypoints are labeled
                    if v1 > 0 and v2 > 0:
                        draw.line([(x1, y1), (x2, y2)], fill=skeleton_color, width=skeleton_width)

    num_individuals = len(keypoint_annotations)
    logger.info(f"Visualized keypoints for {num_individuals} individual(s)")

    # Display or return image
    if show_plot:
        plt.figure(figsize=figsize)
        plt.imshow(img)
        title = f"Image: {image_filename} (ID: {image_id}) | {num_individuals} Individual(s)"
        plt.title(title)
        plt.axis("off")
        plt.show()
        return None
    else:
        return img


def _find_image_recursively(root_dir: Path, filename: str) -> Optional[Path]:
    """Recursively search for a file within a directory tree.

    Args:
        root_dir: Root directory to start the search.
        filename: Name of the file to find.

    Returns:
        Path to the found file, or None if not found.
    """
    for path in root_dir.rglob(filename):
        if path.is_file():
            return path
    return None

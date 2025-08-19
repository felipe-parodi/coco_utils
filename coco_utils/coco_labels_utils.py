"""COCO Label Manipulation Utilities.

This module provides functions for modifying the content within COCO
annotation files, such as shrinking bounding boxes interactively.
It often works in conjunction with coco_viz_utils for visual feedback.
"""

import copy
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from .coco_viz_utils import find_annotations, find_image_info
from .exceptions import (
    FileOperationError,
    InvalidCOCOFormatError,
    VisualizationError,
)
from .logger import get_logger

logger = get_logger(__name__)

# Default values for annotation validation
DEFAULT_CORE_KEYS = {"id", "image_id", "category_id", "bbox", "area", "iscrowd", "center", "scale"}
DEFAULT_FIX_VALUES = {"iscrowd": 0}
DEFAULT_CONSISTENCY_IGNORE_KEYS = {"keypoints", "num_keypoints", "segmentation"}


def load_coco_data(coco_path: Union[str, Path]) -> Dict[str, Any]:
    """Load COCO data from a JSON file.

    Args:
        coco_path: Path to the COCO JSON file.

    Returns:
        Dictionary containing the loaded COCO data.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        InvalidCOCOFormatError: If the file contains invalid JSON.
        FileOperationError: If there's an error reading the file.

    Example:
        >>> coco_data = load_coco_data("annotations.json")
        >>> print(f"Loaded {len(coco_data['images'])} images")
    """
    coco_path = Path(coco_path)

    if not coco_path.exists():
        raise FileNotFoundError(f"COCO file not found: {coco_path}")

    try:
        logger.info(f"Loading COCO data from {coco_path}")
        with open(coco_path, "r") as f:
            coco_data = json.load(f)
        logger.info(f"Successfully loaded COCO data from {coco_path}")
        return coco_data
    except json.JSONDecodeError as e:
        raise InvalidCOCOFormatError(f"Invalid JSON in file {coco_path}: {e}")
    except Exception as e:
        raise FileOperationError(f"Error loading COCO file {coco_path}: {e}")


def shrink_bbox(
    bbox: List[float],
    shrink_percent: float,
    img_width: Optional[int] = None,
    img_height: Optional[int] = None,
) -> List[float]:
    """Shrink a single bounding box towards its center by a percentage.

    The box is shrunk uniformly from all sides towards its center point.
    If image dimensions are provided, the box is clamped to stay within bounds.

    Args:
        bbox: Bounding box as [x_min, y_min, width, height].
        shrink_percent: Percentage (0-100) to shrink width/height.
        img_width: Optional image width for boundary clamping.
        img_height: Optional image height for boundary clamping.

    Returns:
        Shrunk bounding box as [x_min, y_min, width, height].
        Returns original bbox if shrink would result in invalid dimensions.

    Example:
        >>> bbox = [10, 10, 100, 100]
        >>> shrunk = shrink_bbox(bbox, 10.0)  # Shrink by 10%
        >>> print(shrunk)  # [15, 15, 90, 90]
    """
    if not (0 < shrink_percent < 100):
        logger.warning(f"Invalid shrink_percent ({shrink_percent}). Returning original bbox.")
        return bbox

    x_min, y_min, width, height = bbox

    # Calculate shrink amounts (half on each side)
    shrink_w = (width * shrink_percent / 100.0) / 2.0
    shrink_h = (height * shrink_percent / 100.0) / 2.0

    # Calculate new dimensions
    new_x_min = x_min + shrink_w
    new_y_min = y_min + shrink_h
    new_width = width - 2 * shrink_w
    new_height = height - 2 * shrink_h

    # Ensure dimensions are positive
    if new_width <= 0 or new_height <= 0:
        logger.debug(
            f"Shrinking would result in non-positive dimensions for bbox {bbox}. "
            f"Returning original."
        )
        return bbox

    # Clamp coordinates to image boundaries if provided
    if img_width is not None:
        new_x_min = max(0, new_x_min)
        new_width = min(new_width, img_width - new_x_min)
    if img_height is not None:
        new_y_min = max(0, new_y_min)
        new_height = min(new_height, img_height - new_y_min)

    # Ensure dimensions didn't become negative due to clamping
    new_width = max(0, new_width)
    new_height = max(0, new_height)

    return [new_x_min, new_y_min, new_width, new_height]


def calculate_shrunk_bboxes(
    coco_data: Dict[str, Any], shrink_percent: float
) -> Tuple[Dict[str, Any], Set[int]]:
    """Calculate shrunk bounding boxes for all annotations in COCO data.

    Creates a deep copy of the data and modifies all bounding boxes.
    Does not perform visualization or file I/O.

    Args:
        coco_data: Loaded COCO data dictionary.
        shrink_percent: Percentage to shrink width/height (0 < shrink_percent < 100).

    Returns:
        Tuple containing:
            - modified_coco_data: Deep copy with updated bbox fields.
            - modified_image_ids: Set of image IDs with at least one modified bbox.

    Raises:
        ValueError: If shrink_percent is not between 0 and 100.

    Example:
        >>> modified_data, modified_ids = calculate_shrunk_bboxes(coco_data, 10.0)
        >>> print(f"Modified {len(modified_ids)} images")
    """
    if not (0 < shrink_percent < 100):
        raise ValueError("shrink_percent must be between 0 and 100 (exclusive).")

    modified_coco_data = copy.deepcopy(coco_data)
    modified_image_ids = set()
    num_boxes_shrunk = 0

    # Store image dimensions for quick lookup
    image_dims = {
        img["id"]: (img.get("width"), img.get("height"))
        for img in modified_coco_data.get("images", [])
    }

    for ann in modified_coco_data.get("annotations", []):
        if "bbox" in ann and ann["bbox"] is not None and len(ann["bbox"]) == 4:
            original_bbox = list(ann["bbox"])
            image_id = ann["image_id"]
            img_width, img_height = image_dims.get(image_id, (None, None))

            new_bbox = shrink_bbox(original_bbox, shrink_percent, img_width, img_height)

            if new_bbox != original_bbox:
                ann["bbox"] = new_bbox
                # Update area if it exists
                if "area" in ann:
                    ann["area"] = new_bbox[2] * new_bbox[3]
                modified_image_ids.add(image_id)
                num_boxes_shrunk += 1

    logger.info(
        f"Shrinking calculation complete. {num_boxes_shrunk} boxes modified "
        f"across {len(modified_image_ids)} images."
    )
    return modified_coco_data, modified_image_ids


def shrink_coco_bboxes(
    coco_data: Dict[str, Any],
    output_path: Union[str, Path],
    image_dir: Optional[Union[str, Path]] = None,
    shrink_percent: float = 5.0,
    original_box_color: str = "blue",
    shrunk_box_color: str = "red",
    box_width: int = 2,
    interactive: bool = True,
) -> Optional[str]:
    """Shrink all bounding boxes in COCO data towards their center by a percentage.

    Uses calculate_shrunk_bboxes for the core logic. If interactive=True, visualizes
    changes for the first modified image and saves upon user confirmation.
    If interactive=False, performs the shrinkage and saves directly.

    Args:
        coco_data: Loaded COCO data dictionary.
        output_path: Path to save the modified JSON file.
        image_dir: Directory containing images (required if interactive=True).
        shrink_percent: Percentage to shrink width/height (0 < shrink_percent < 100).
        original_box_color: Color for original boxes in interactive mode.
        shrunk_box_color: Color for shrunk boxes in interactive mode.
        box_width: Width of box outlines in interactive mode.
        interactive: If True, show visualization and ask for confirmation.

    Returns:
        Path to the saved JSON file if saved, otherwise None.

    Raises:
        ValueError: If required arguments are missing or invalid.
        FileNotFoundError: If image directory or images not found in interactive mode.
        VisualizationError: If there's an error during visualization.
        FileOperationError: If there's an error saving the file.

    Example:
        >>> output = shrink_coco_bboxes(
        ...     coco_data,
        ...     "shrunk_annotations.json",
        ...     image_dir="images",
        ...     shrink_percent=10.0,
        ...     interactive=False
        ... )
        >>> print(f"Saved to {output}")
    """
    output_path = Path(output_path)

    if interactive:
        if image_dir is None:
            raise ValueError("image_dir must be provided when interactive=True.")
        image_dir_path = Path(image_dir)
        if not image_dir_path.is_dir():
            raise FileNotFoundError(f"Image directory not found for interactive mode: {image_dir}")

    if not (0 < shrink_percent < 100):
        raise ValueError("shrink_percent must be between 0 and 100 (exclusive).")

    mode_str = "interactive" if interactive else "batch"
    logger.info(f"Starting {mode_str} bounding box shrinking process ({shrink_percent}%)...")

    # Calculate shrunk boxes
    try:
        shrunk_coco_data, modified_image_ids = calculate_shrunk_bboxes(coco_data, shrink_percent)
    except ValueError as e:
        raise e

    if not modified_image_ids:
        logger.info("No bounding boxes were modified. No file saved.")
        return None

    # Interactive section
    accept_changes = False
    if interactive:
        logger.info("Visualizing changes on the first modified image...")
        first_modified_id = sorted(list(modified_image_ids))[0]

        # Find image info
        img_info = find_image_info(coco_data, first_modified_id)
        if not img_info:
            raise ValueError(
                f"Could not find image info for modified image ID {first_modified_id}."
            )

        image_filename = img_info.get("file_name")
        if not image_filename:
            raise ValueError(f"'file_name' missing for example image ID {first_modified_id}.")

        image_path = Path(image_dir) / image_filename

        try:
            img = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file not found at '{image_path}' for visualization.")
        except Exception as e:
            raise VisualizationError(f"Error loading image '{image_path}' for visualization: {e}")

        draw = ImageDraw.Draw(img)

        # Get annotations for the specific image
        original_annotations = find_annotations(coco_data, first_modified_id)
        shrunk_annotations = find_annotations(shrunk_coco_data, first_modified_id)

        logger.info(f"Example Image (ID: {first_modified_id}, Filename: {image_filename})")
        logger.info(
            f"Drawing {len(original_annotations)} original boxes and "
            f"{len(shrunk_annotations)} shrunk boxes."
        )

        # Draw original boxes
        num_orig_drawn = 0
        for ann_orig in original_annotations:
            bbox = ann_orig.get("bbox")
            if bbox and len(bbox) == 4:
                x_min, y_min, width, height = map(round, bbox)
                draw.rectangle(
                    [x_min, y_min, x_min + width, y_min + height],
                    outline=original_box_color,
                    width=box_width,
                )
                num_orig_drawn += 1

        # Draw modified boxes
        num_shrunk_drawn = 0
        original_bboxes_by_id = {
            ann.get("id"): ann.get("bbox")
            for ann in original_annotations
            if ann.get("id") is not None
        }

        for ann_shrunk in shrunk_annotations:
            bbox_shrunk = ann_shrunk.get("bbox")
            ann_id = ann_shrunk.get("id")
            if bbox_shrunk and len(bbox_shrunk) == 4 and ann_id is not None:
                bbox_original = original_bboxes_by_id.get(ann_id)
                if bbox_original is None or bbox_shrunk != bbox_original:
                    x_min, y_min, width, height = map(round, bbox_shrunk)
                    draw.rectangle(
                        [x_min, y_min, x_min + width, y_min + height],
                        outline=shrunk_box_color,
                        width=box_width,
                    )
                    num_shrunk_drawn += 1

        logger.info(
            f"Displayed: {num_orig_drawn} original boxes, " f"{num_shrunk_drawn} shrunk boxes."
        )

        plt.figure(figsize=(12, 12))
        plt.imshow(img)
        title = (
            f"Image ID: {first_modified_id} - Shrinkage Example ({shrink_percent}%)\n"
            f"Original ({original_box_color.title()}) vs. Shrunk ({shrunk_box_color.title()})"
        )
        plt.title(title)
        plt.axis("off")
        plt.show(block=True)

        logger.info("Waiting for user confirmation...")
        time.sleep(1)

        # Confirmation prompt
        while True:
            user_input = (
                input(
                    f"\nAccept shrinkage for ALL {len(modified_image_ids)} "
                    f"modified images? (y/n): "
                )
                .lower()
                .strip()
            )
            if user_input == "y":
                accept_changes = True
                logger.info("User accepted changes for all modified images.")
                break
            elif user_input == "n":
                logger.info("User rejected changes. No file will be saved.")
                plt.close("all")
                return None
            else:
                print("Invalid input. Please enter y or n.")
        plt.close("all")

    else:
        logger.info("Running in non-interactive mode. All changes will be applied.")
        accept_changes = True

    # Save section
    if not accept_changes:
        logger.info("Changes were not accepted. No file saved.")
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Summary before saving:")
    logger.info(f"  Output path: {output_path}")
    logger.info(f"  Original images: {len(coco_data.get('images', []))}")
    logger.info(f"  Original annotations: {len(coco_data.get('annotations', []))}")
    logger.info(f"  Final images: {len(shrunk_coco_data.get('images', []))}")
    logger.info(f"  Final annotations: {len(shrunk_coco_data.get('annotations', []))}")
    logger.info(f"  Bboxes shrunk in {len(modified_image_ids)} images.")

    logger.info(f"Saving modified data to '{output_path}'...")
    try:
        with open(output_path, "w") as f:
            json.dump(shrunk_coco_data, f, indent=4)
        logger.info("Save successful.")
        return str(output_path)
    except Exception as e:
        raise FileOperationError(f"Error saving file: {e}")


def _calculate_geo_props_from_bbox(bbox: List[Union[float, int]]) -> Dict[str, Any]:
    """Calculate derived geometric properties from bbox.

    Args:
        bbox: Bounding box as [x, y, width, height].

    Returns:
        Dictionary with calculated 'area', 'center', and 'scale' if valid.
    """
    props = {}
    try:
        if isinstance(bbox, list) and len(bbox) == 4:
            x, y, w, h = bbox
            # Ensure w, h are positive for valid calculations
            if w > 0 and h > 0:
                props["area"] = float(w * h)
                props["center"] = [float(x + w / 2), float(y + h / 2)]
                props["scale"] = [float(w / 200.0), float(h / 200.0)]
    except (TypeError, IndexError):
        # Ignore errors if bbox has invalid content
        pass
    return props


def validate_annotations(
    coco_data: Dict[str, Any],
    core_keys: Set[str] = DEFAULT_CORE_KEYS,
    fix: bool = False,
    default_values: Dict[str, Any] = DEFAULT_FIX_VALUES,
    check_consistency: bool = True,
    consistency_ignore_keys: Set[str] = DEFAULT_CONSISTENCY_IGNORE_KEYS,
    raise_error: bool = False,
    output_path: Optional[Union[str, Path]] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """Validate annotations in COCO data for missing keys and inconsistencies.

    Checks if all annotations contain a set of core keys. Optionally checks
    if all annotations have the exact same set of keys as the first annotation.
    Can optionally attempt to fix missing core keys.

    Args:
        coco_data: The loaded COCO data dictionary.
        core_keys: Set of keys expected in every annotation.
        fix: If True, attempt to add missing core keys.
        default_values: Default values for keys that cannot be calculated.
        check_consistency: If True, check if all annotations have same keys.
        consistency_ignore_keys: Keys to ignore during consistency check.
        raise_error: If True, raise ValueError if inconsistencies found.
        output_path: If fix=True and provided, save modified data here.

    Returns:
        Tuple containing:
            - is_valid: True if all checks pass, False otherwise.
            - coco_data: Original or potentially modified coco_data dict.

    Raises:
        ValueError: If raise_error is True and inconsistencies are found.

    Example:
        >>> is_valid, fixed_data = validate_annotations(
        ...     coco_data,
        ...     fix=True,
        ...     output_path="fixed_annotations.json"
        ... )
        >>> if is_valid:
        ...     print("Annotations are valid!")
    """
    logger.info("Running annotation validation...")
    annotations = coco_data.get("annotations")
    if not annotations:
        logger.info("No annotations found to validate.")
        return True, coco_data

    is_valid = True
    issues_found = defaultdict(list)
    num_fixed = 0
    fixed_keys_summary = defaultdict(int)

    # Use deep copy if fixing to avoid modifying original
    output_coco_data = copy.deepcopy(coco_data) if fix else coco_data
    output_annotations = output_coco_data["annotations"]

    reference_keys = set(output_annotations[0].keys()) if check_consistency else set()
    logger.info(f"Using core keys for check: {sorted(list(core_keys))}")
    if check_consistency:
        logger.info(
            f"Using keys from first annotation (ID: {output_annotations[0].get('id')}) "
            f"for consistency check: {sorted(list(reference_keys))}"
        )

    for i, ann in enumerate(output_annotations):
        ann_id = ann.get("id", f"(index {i})")
        current_keys = set(ann.keys())

        # Calculate geometric properties if bbox exists
        geo_props = {}
        if "bbox" in ann:
            geo_props = _calculate_geo_props_from_bbox(ann["bbox"])

        # Check for missing core keys
        missing_core = core_keys - current_keys
        if missing_core:
            is_valid = False
            msg = f"Missing core keys: {sorted(list(missing_core))}"
            issues_found[ann_id].append(msg)
            if fix:
                fixed_in_ann = False
                for key in missing_core:
                    if key in geo_props:
                        ann[key] = geo_props[key]
                    elif key in default_values:
                        ann[key] = default_values[key]
                    else:
                        logger.warning(
                            f"Cannot fix missing core key '{key}' for ann {ann_id} "
                            f"(no default/calculation possible)."
                        )
                        continue
                    fixed_keys_summary[key] += 1
                    fixed_in_ann = True
                if fixed_in_ann:
                    num_fixed += 1

        # Check for key set consistency
        if check_consistency:
            current_keys_after_fix = set(ann.keys())
            if current_keys_after_fix != reference_keys:
                # Calculate differences ignoring specified keys
                missing_relative = (
                    reference_keys - current_keys_after_fix
                ) - consistency_ignore_keys
                extra_relative = (current_keys_after_fix - reference_keys) - consistency_ignore_keys

                if missing_relative:
                    msg = f"Keys missing compared to first ann: {sorted(list(missing_relative))}"
                    issues_found[ann_id].append(msg)
                    is_valid = False

                    if fix:
                        fixed_in_ann_relative = False
                        for key in missing_relative:
                            if key not in core_keys or key not in ann:
                                if key in geo_props:
                                    ann[key] = geo_props[key]
                                elif key in default_values:
                                    ann[key] = default_values[key]
                                else:
                                    logger.warning(
                                        f"No default/calculation for key '{key}'. "
                                        f"Skipping fix for this key in ann {ann_id}."
                                    )
                                    continue
                                fixed_keys_summary[key] += 1
                                fixed_in_ann_relative = True
                        if fixed_in_ann_relative and ann_id not in issues_found:
                            num_fixed += 1

                if extra_relative:
                    msg = f"Extra keys compared to first ann: {sorted(list(extra_relative))}"
                    issues_found[ann_id].append(msg)

    # Reporting
    if not issues_found:
        logger.info("Validation successful: All annotations appear consistent.")
    else:
        logger.info(f"Validation finished. Issues found in {len(issues_found)} annotations:")
        # Log details for first few problematic annotations
        limit = 5
        for idx, (ann_id, msgs) in enumerate(issues_found.items()):
            if idx < limit:
                logger.info(f"  Ann ID {ann_id}:")
                for msg in msgs:
                    logger.info(f"    - {msg}")
            elif idx == limit:
                logger.info(f"  ... and {len(issues_found) - limit} more annotations with issues.")
                break

        if fix and num_fixed > 0:
            logger.info(f"Attempted to fix issues in {num_fixed} annotations.")
            logger.info(f"Summary of fixed/added keys: {dict(fixed_keys_summary)}")
            is_valid = True  # Assume fixed state is valid
            logger.info("Note: Consistency check might still fail if extra keys remain.")

    if not is_valid and raise_error:
        raise ValueError("Annotation validation failed. Found inconsistencies.")

    # Optional saving
    if fix and output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Attempting to save fixed data to: {output_path}")
        try:
            with open(output_path, "w") as f:
                json.dump(output_coco_data, f, indent=4)
            logger.info("Save successful.")
        except Exception as e:
            logger.error(f"Error saving fixed file to {output_path}: {e}")

    logger.info("Annotation validation complete.")
    return is_valid, output_coco_data

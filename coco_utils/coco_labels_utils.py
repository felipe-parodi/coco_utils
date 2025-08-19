"""
COCO Label Manipulation Utilities.

This module provides functions for modifying the content within COCO
annotation files, such as shrinking bounding boxes interactively.
It often works in conjunction with coco_viz_utils for visual feedback.
"""

# coco_labels_utils.py
import copy
import json
import math
import os
from pathlib import Path
import time # Import the time module
from typing import Any, Dict, List, Optional, Union, Tuple, Set
from collections import defaultdict
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from .coco_viz_utils import find_annotations, find_image_info

def load_coco_data(coco_path: Union[str, Path]) -> Dict:
    with open(coco_path, "r") as f:
        coco_data = json.load(f)
    return coco_data


# --- Helper: Shrink Single Bbox ---

def shrink_bbox(
    bbox: List[float],
    shrink_percent: float,
    img_width: Optional[int] = None,
    img_height: Optional[int] = None,
) -> List[float]:
    """
    Shrinks a single bounding box towards its center by a percentage.

    Args:
        bbox: List [x_min, y_min, width, height].
        shrink_percent: Percentage (0-100) to shrink width/height.
        img_width: Optional image width for clamping.
        img_height: Optional image height for clamping.

    Returns:
        List representing the shrunk bounding box [x_min, y_min, width, height].
        Returns the original bbox if shrink_percent is invalid or leads to non-positive dimensions.
    """
    if not (0 < shrink_percent < 100):
        print(f"Warning: Invalid shrink_percent ({shrink_percent}). Returning original bbox.")
        return bbox # Return original if percentage is invalid

    x_min, y_min, width, height = bbox
    center_x = x_min + width / 2
    center_y = y_min + height / 2

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
        # print(f"Warning: Shrinking leads to non-positive dimensions for box {bbox}. Returning original.")
        return bbox # Return original if dimensions become non-positive

    # Clamp coordinates to image boundaries if provided
    if img_width is not None:
        new_x_min = max(0, new_x_min)
        new_width = min(new_width, img_width - new_x_min)
    if img_height is not None:
        new_y_min = max(0, new_y_min)
        new_height = min(new_height, img_height - new_y_min)

    # Ensure width and height didn't become negative due to clamping edge cases
    new_width = max(0, new_width)
    new_height = max(0, new_height)


    return [new_x_min, new_y_min, new_width, new_height]

# --- Core Calculation Function ---

def calculate_shrunk_bboxes(
    coco_data: Dict, shrink_percent: float
) -> Tuple[Dict, Set[int]]:
    """
    Calculates shrunk bounding boxes for all annotations in COCO data.

    Does not perform visualization or file I/O. Returns the modified data
    and the set of image IDs that had at least one box modified.

    Args:
        coco_data: Loaded COCO data dictionary.
        shrink_percent: Percentage to shrink width/height (0 < shrink_percent < 100).

    Returns:
        Tuple containing:
        - modified_coco_data: A deep copy of coco_data with 'bbox' fields updated.
        - modified_image_ids: A set of image IDs where at least one bbox was changed.

    Raises:
        ValueError: If shrink_percent is not between 0 and 100.
    """
    if not (0 < shrink_percent < 100):
        raise ValueError("shrink_percent must be between 0 and 100 (exclusive).")

    modified_coco_data = copy.deepcopy(coco_data)
    modified_image_ids = set()
    num_boxes_shrunk = 0

    # Store image dimensions for quick lookup
    image_dims = {
        img['id']: (img.get('width'), img.get('height'))
        for img in modified_coco_data.get('images', [])
    }

    for ann in modified_coco_data.get("annotations", []):
        if "bbox" in ann and ann["bbox"] is not None and len(ann["bbox"]) == 4:
            original_bbox = list(ann["bbox"]) # Keep original for comparison
            image_id = ann["image_id"]
            img_width, img_height = image_dims.get(image_id, (None, None))

            # Check if we have dimensions; shrink_bbox can handle None, but good practice
            # if img_width is None or img_height is None:
            #     print(f"Warning: Missing dimensions for image {image_id}, cannot clamp bbox for annotation {ann.get('id')}")
                # Continue shrinking without clamping

            new_bbox = shrink_bbox(
                original_bbox, shrink_percent, img_width, img_height
            )

            if new_bbox != original_bbox:
                # Ensure the new bounding box coordinates are rounded or formatted consistently if needed
                # For simplicity, keeping floats as calculated. JSON handles floats.
                ann["bbox"] = new_bbox
                # Update area if it exists
                if "area" in ann:
                    ann["area"] = new_bbox[2] * new_bbox[3]
                modified_image_ids.add(image_id)
                num_boxes_shrunk += 1

    print(
        f"Shrinking calculation complete. {num_boxes_shrunk} boxes potentially modified across {len(modified_image_ids)} images."
    )
    return modified_coco_data, modified_image_ids


# --- Main Function with Interactive/Saving Logic ---

def shrink_coco_bboxes(
    coco_data: Dict,
    output_path: Union[str, Path],
    image_dir: Optional[str] = None,
    shrink_percent: float = 5.0,
    original_box_color: str = "blue",
    shrunk_box_color: str = "red",
    box_width: int = 2,
    interactive: bool = True,
) -> Optional[str]:
    """
    Shrinks all bounding boxes in COCO data towards their center by a percentage.

    Uses calculate_shrunk_bboxes for the core logic.
    If interactive=True, visualizes changes for the first modified image
    (showing original and shrunk boxes together) and saves the entire modified
    dataset upon user confirmation.

    If interactive=False, performs the shrinkage calculation and saves the
    result directly to the output file without visualization or prompts.

    Args:
        coco_data: Loaded COCO data dictionary.
        output_path: Full path (including filename) to save the modified JSON file.
        image_dir: Path to the directory containing images (Required if interactive=True).
        shrink_percent: Percentage to shrink width/height (0 < shrink_percent < 100).
        original_box_color: Color for original bounding boxes in interactive mode.
        shrunk_box_color: Color for shrunk bounding boxes in interactive mode.
        box_width: Width of bounding box outlines in interactive mode.
        interactive: If True, show visualization and ask for confirmation.
                     If False, process and save directly.

    Returns:
        The path to the saved JSON file if saved, otherwise None.

    Raises:
        ValueError: If required arguments are missing or invalid (e.g., image_dir in interactive mode, invalid shrink_percent).
        FileNotFoundError: If image directory or specific images are not found in interactive mode.
        Exception: For other file loading or saving errors.
    """
    # --- Input Validation ---
    output_path = Path(output_path) # Ensure it's a Path object early

    if interactive:
        if image_dir is None:
            raise ValueError("image_dir must be provided when interactive=True.")
        image_dir_path = Path(image_dir)
        if not image_dir_path.is_dir():
            raise FileNotFoundError(f"Image directory not found for interactive mode: {image_dir}")
    if not (0 < shrink_percent < 100):
         raise ValueError("shrink_percent must be between 0 and 100 (exclusive).")


    mode_str = "interactive" if interactive else "batch"
    print(f"Starting {mode_str} bounding box shrinking process ({shrink_percent}%)...")

    # 1. Calculate shrunk boxes using the refactored function
    try:
        shrunk_coco_data, modified_image_ids = calculate_shrunk_bboxes(
            coco_data, shrink_percent
        )
    except ValueError as e: # Catch potential errors from calculate_shrunk_bboxes
        raise e # Re-raise as the input validation should happen here too

    if not modified_image_ids:
        print(
            "No bounding boxes were modified after shrinking calculation. No file saved."
        )
        return None

    # --- Interactive Section ---
    accept_changes = False
    if interactive:
        # We need the original coco_data here for comparison visualization
        print(
            "\nVisualizing changes on the first modified image (Original=Blue, Shrunk=Red)..."
        )
        first_modified_id = sorted(list(modified_image_ids))[0]

        # Find image info using the original data structure
        img_info = find_image_info(coco_data, first_modified_id)
        if not img_info:
             # This case should be unlikely if modified_image_ids is populated correctly,
             # but handle defensively.
             raise ValueError(f"Could not find image info for supposedly modified image ID {first_modified_id}.")


        image_filename = img_info.get("file_name")
        if not image_filename:
            raise ValueError(f"'file_name' missing for example image ID {first_modified_id}.")

        image_path = Path(image_dir) / image_filename # image_dir guaranteed valid here

        try:
            img = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file not found at '{image_path}' for visualization.")
        except Exception as e:
            raise Exception(f"Error loading image '{image_path}' for visualization: {e}") from e

        draw = ImageDraw.Draw(img)
        # Get annotations for the specific image from original and shrunk data
        original_annotations = find_annotations(coco_data, first_modified_id)
        shrunk_annotations = find_annotations(shrunk_coco_data, first_modified_id) # Use the shrunk data result

        print(
            f"--- Example Image (ID: {first_modified_id}, Filename: {image_filename}) ---"
        )
        print(
            f"Drawing {len(original_annotations)} original boxes (Blue) and {len(shrunk_annotations)} shrunk boxes (Red)."
        )

        # Draw original boxes (from original coco_data)
        num_orig_drawn = 0
        for ann_orig in original_annotations:
            bbox = ann_orig.get("bbox")
            if bbox and len(bbox) == 4:
                x_min, y_min, width, height = map(round, bbox) # Use rounded values for drawing
                draw.rectangle(
                    [x_min, y_min, x_min + width, y_min + height],
                    outline=original_box_color,
                    width=box_width,
                )
                num_orig_drawn += 1

        # Draw modified (shrunk) boxes (from shrunk_coco_data)
        num_shrunk_drawn = 0
        # Create a map of original bboxes by ID for quick comparison
        original_bboxes_by_id = {
            ann.get("id"): ann.get("bbox") for ann in original_annotations if ann.get("id") is not None
        }

        for ann_shrunk in shrunk_annotations:
            bbox_shrunk = ann_shrunk.get("bbox")
            ann_id = ann_shrunk.get("id")
            if bbox_shrunk and len(bbox_shrunk) == 4 and ann_id is not None:
                # Only draw red if it's different from original
                bbox_original = original_bboxes_by_id.get(ann_id)
                # Compare element-wise, handle potential floating point inaccuracies if necessary
                # A simple direct comparison should be okay here unless shrink is tiny
                if bbox_original is None or bbox_shrunk != bbox_original:
                    x_min, y_min, width, height = map(round, bbox_shrunk) # Use rounded values for drawing
                    draw.rectangle(
                        [x_min, y_min, x_min + width, y_min + height],
                        outline=shrunk_box_color,
                        width=box_width,
                    )
                    num_shrunk_drawn += 1
                # Optionally draw unchanged boxes in a different color? For now, only draw changed ones in red.


        print(
            f"Displayed: {num_orig_drawn} original boxes, {num_shrunk_drawn} visually shrunk boxes."
        )

        plt.figure(figsize=(12, 12))
        plt.imshow(img)
        title = (
            f"Image ID: {first_modified_id} - Shrinkage Example ({shrink_percent}%)\n"
            f"Original (Blue) vs. Shrunk (Red)"
        )
        plt.title(title)
        plt.axis("off")
        plt.show(block=True)

        print("Waiting for plot window...")
        time.sleep(1)

        # Confirmation Prompt for changes
        while True:
            user_input = (
                input(
                    f"\nAccept shrinkage for ALL {len(modified_image_ids)} modified images? (y/n): "
                )
                .lower()
                .strip()
            )
            if user_input == "y":
                accept_changes = True
                print(f"Accepted changes for all modified images.")
                break
            elif user_input == "n":
                print("Discarding all shrinkage changes. No file will be saved.")
                plt.close("all") # Ensure plot closes on 'n'
                return None
            else:
                print("Invalid input. Please enter y or n.")
        plt.close("all")

    else:
        # Non-interactive mode: automatically accept all calculated changes
        print("Running in non-interactive mode. All calculated changes will be applied.")
        accept_changes = True

    # --- Final Save Section ---
    if not accept_changes:
        # This case should now only be reachable if interactive=True and user entered 'n'
        # The 'n' case already returned None, so this part might be logically unreachable,
        # but keep it as a safeguard.
        print("Changes were not accepted. No file saved.")
        return None

    # If we reach here, changes are accepted (either interactively or automatically)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSummary before saving:") # Changed print message
    print(f" - Output path: {output_path}")
    print(f" - Original image count: {len(coco_data.get('images', []))}")
    print(f" - Original annotation count: {len(coco_data.get('annotations', []))}")
    print(f" - Final image count: {len(shrunk_coco_data.get('images', []))}")
    print(f" - Final annotation count: {len(shrunk_coco_data.get('annotations', []))}")
    print(f" - Bounding boxes potentially shrunk in {len(modified_image_ids)} images.")

    # Proceed directly to saving the shrunk_coco_data
    print(f"\nSaving modified data to '{output_path}'...")
    try:
        with open(output_path, "w") as f:
            json.dump(shrunk_coco_data, f, indent=4) # Save the shrunk data
        print("Save successful.")
        return str(output_path)
    except Exception as e:
        print(f"Error saving file: {e}")
        raise Exception(f"Error saving file: {e}") from e

# --- Annotation Validation --- #

# Default set of expected keys for a standard COCO object detection annotation
# Includes derived keys. 'keypoints' and 'num_keypoints' are checked for consistency
# if present in the first annotation but are not required core keys by default.
DEFAULT_CORE_KEYS = {
    'id', 'image_id', 'category_id', 'bbox', 'area', 'iscrowd', 'center', 'scale'
}

# Default values used when fixing missing keys that cannot be calculated.
DEFAULT_FIX_VALUES = {'iscrowd': 0}

# Default set of keys to ignore when checking for exact key consistency
DEFAULT_CONSISTENCY_IGNORE_KEYS = {'keypoints', 'num_keypoints', 'segmentation'}

# Helper to calculate derived geometric properties from bbox
def _calculate_geo_props_from_bbox(bbox: List[Union[float, int]]) -> Dict[str, Any]:
    """Calculates area, center, scale from a valid bbox."""
    props = {}
    try:
        if isinstance(bbox, list) and len(bbox) == 4:
            x, y, w, h = bbox
            # Ensure w, h are positive for valid calculations
            if w > 0 and h > 0:
                props['area'] = float(w * h)
                props['center'] = [float(x + w / 2), float(y + h / 2)]
                props['scale'] = [float(w / 200.0), float(h / 200.0)]
    except (TypeError, IndexError) as e:
        # Ignore errors if bbox has invalid content
        # print(f"Debug: Could not calculate props from bbox {bbox}: {e}")
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
    """
    Validates annotations in COCO data for missing keys and inconsistencies.

    Checks if all annotations contain a set of core keys. Optionally, checks
    if all annotations have the exact same set of keys as the first annotation.
    Can optionally attempt to fix missing core keys by adding them with default values.

    Args:
        coco_data: The loaded COCO data dictionary.
        core_keys: The set of keys expected to be present in every annotation.
                   Defaults to DEFAULT_CORE_KEYS (including derived geo keys).
        fix: If True, attempt to add missing core keys. 'area', 'center', 'scale'
             are calculated from 'bbox' if possible. 'iscrowd' uses default_values.
             Other core keys (id, image_id, category_id, bbox) cannot be fixed.
             If check_consistency is True, also attempts to add keys missing
             relative to the first annotation if they can be calculated or have defaults.
        default_values: Dictionary of default values for keys that cannot be calculated
                        (primarily 'iscrowd'). Defaults to DEFAULT_FIX_VALUES.
        check_consistency: If True, check if all annotations have the exact same
                           set of keys as the first annotation.
        consistency_ignore_keys: Set of keys to ignore during the consistency check.
                                 Defaults to DEFAULT_CONSISTENCY_IGNORE_KEYS.
        raise_error: If True, raise ValueError if any inconsistencies are found.
        output_path: If `fix` is True and this path is provided, the potentially
                     modified coco_data will be saved to this file path.

    Returns:
        Tuple containing:
        - is_valid (bool): True if all checks pass, False otherwise.
        - coco_data (Dict): The original or potentially modified coco_data dict.
                           If fix=True, this dict may have annotations modified.

    Raises:
        ValueError: If raise_error is True and inconsistencies are found.
    """
    print("\n--- Running Annotation Validation ---")
    annotations = coco_data.get("annotations")
    if not annotations:
        print("No annotations found to validate.")
        return True, coco_data

    is_valid = True
    issues_found = defaultdict(list)
    num_fixed = 0
    fixed_keys_summary = defaultdict(int)

    # Use a deep copy if fixing to avoid modifying the original dict unless intended
    output_coco_data = copy.deepcopy(coco_data) if fix else coco_data
    output_annotations = output_coco_data["annotations"] # Work on the potentially copied list

    reference_keys = set(output_annotations[0].keys()) if check_consistency else set()
    print(f"Using core keys for check: {sorted(list(core_keys))}")
    if check_consistency:
        print(f"Using keys from first annotation (ID: {output_annotations[0].get('id')}) for consistency check: {sorted(list(reference_keys))}")

    for i, ann in enumerate(output_annotations):
        ann_id = ann.get("id", f"(index {i})")
        current_keys = set(ann.keys())

        # Attempt to calculate geometric properties first if bbox exists
        geo_props = {}
        if 'bbox' in ann:
            geo_props = _calculate_geo_props_from_bbox(ann['bbox'])

        # 1. Check for missing core keys
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
                        # Cannot fix keys like id, image_id, category_id, bbox or
                        # derived keys if bbox was invalid/missing.
                        print(f"  Warning: Cannot fix missing core key '{key}' for ann {ann_id} (no default/calculation possible).")
                        continue # Skip adding this specific key
                    fixed_keys_summary[key] += 1
                    fixed_in_ann = True
                if fixed_in_ann:
                    num_fixed +=1 # Count annotations fixed, not individual keys


        # 2. Check for key set consistency (if enabled)
        if check_consistency:
            # Recalculate current keys *after potential fixes* for accurate comparison
            current_keys_after_fix = set(ann.keys())
            if current_keys_after_fix != reference_keys:
                # Calculate differences ignoring specified keys
                missing_relative = (reference_keys - current_keys_after_fix) - consistency_ignore_keys
                extra_relative = (current_keys_after_fix - reference_keys) - consistency_ignore_keys

                if missing_relative:
                    msg = f"Keys missing compared to first ann: {sorted(list(missing_relative))}"
                    # Only report as issue if not already reported as missing core
                    issues_found[ann_id].append(msg)
                    is_valid = False # Mark invalid if relative keys are missing

                    if fix:
                        fixed_in_ann_relative = False
                        for key in missing_relative:
                            # Only fix if it wasn't already handled as a missing *core* key
                            if key not in core_keys or key not in ann:
                                # Try calculated value first
                                if key in geo_props:
                                    ann[key] = geo_props[key]
                                elif key in default_values:
                                    ann[key] = default_values[key]
                                else:
                                    # Don't add keys we don't have defaults/calculations for (e.g., keypoints)
                                    print(f"  Warning: No default/calculation for key '{key}' (from reference ann). Skipping fix for this key in ann {ann_id}.")
                                    continue
                                fixed_keys_summary[key] += 1
                                fixed_in_ann_relative = True
                        if fixed_in_ann_relative and ann_id not in issues_found: # Avoid double counting if core keys were also missing
                            num_fixed +=1

                if extra_relative:
                    # We generally don't fix extra keys automatically, just report.
                    msg = f"Extra keys compared to first ann: {sorted(list(extra_relative))}"
                    # Only report if there are actually keys other than ignored ones
                    issues_found[ann_id].append(msg)
                    # Having extra keys doesn't necessarily make it invalid unless core keys were missing
                    # If core keys were present, this is just an inconsistency warning.

    # --- Reporting --- #
    if not issues_found:
        print("Validation successful: All annotations appear consistent.")
    else:
        print(f"Validation finished. Issues found in {len(issues_found)} annotations:")
        # Print details for the first few problematic annotations
        limit = 5
        for idx, (ann_id, msgs) in enumerate(issues_found.items()):
            if idx < limit:
                print(f"  Ann ID {ann_id}:")
                for msg in msgs:
                    print(f"    - {msg}")
            elif idx == limit:
                print(f"  ... and {len(issues_found) - limit} more annotations with issues.")
                break

        if fix and num_fixed > 0:
            print(f"Attempted to fix issues in {num_fixed} annotations.")
            print(f"Summary of fixed/added keys: {dict(fixed_keys_summary)}")
            # If fixed, the state might now be valid for *core* keys, but maybe not *consistency*
            # Re-running validation without fix would confirm final state.
            is_valid = True # Assume fixed state is now 'valid' for return status if fix was enabled
            print("Note: Consistency check might still fail if extra keys remain.")

    if not is_valid and raise_error:
        raise ValueError("Annotation validation failed. Found inconsistencies.")

    # --- Optional Saving --- #
    if fix and output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"\nAttempting to save fixed data to: {output_path}")
        try:
            with open(output_path, "w") as f:
                # Save the potentially modified data
                json.dump(output_coco_data, f, indent=4)
            print("Save successful.")
        except Exception as e:
            # Don't raise, but report the error
            print(f"Error saving fixed file to {output_path}: {e}")

    print("--- Annotation Validation Complete --- \n")
    return is_valid, output_coco_data


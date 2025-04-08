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


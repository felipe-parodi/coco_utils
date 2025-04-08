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
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from .coco_viz_utils import find_annotations, find_image_info

def load_coco_data(coco_path: Union[str, Path]) -> Dict:
    with open(coco_path, "r") as f:
        coco_data = json.load(f)
    return coco_data


def shrink_coco_bboxes(
    coco_data: Dict,
    output_path: Union[str, Path],
    image_dir: Optional[str] = None,
    shrink_percent: float = 5.0,
    original_box_color: str = "blue",
    shrunk_box_color: str = "red",
    box_width: int = 2,
    interactive: bool = True
) -> Optional[str]:
    """
    Shrinks all bounding boxes in COCO data towards their center by a percentage.

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
        ValueError: If required arguments are missing (e.g., image_dir in interactive mode).
        FileNotFoundError: If image directory or specific images are not found in interactive mode.
        Exception: For other file loading or saving errors.
    """
    # --- Input Validation ---
    output_path = Path(output_path) # Ensure it's a Path object early

    if interactive:
        if image_dir is None:
             raise ValueError("image_dir must be provided when interactive=True.")
        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f"Image directory not found for interactive mode: {image_dir}")


    mode_str = "interactive" if interactive else "batch"
    print(f"Starting {mode_str} bounding box shrinking process ({shrink_percent}%)...")

    # 1. Deep copy the data
    modified_coco_data = copy.deepcopy(coco_data)
    modified_image_ids = set()

    # 2. Iterate and shrink boxes in the copy
    print("Calculating shrunk bounding boxes...")
    num_boxes_shrunk = 0
    for i, ann in enumerate(modified_coco_data.get("annotations", [])):
        if "bbox" in ann and ann["bbox"] is not None and len(ann["bbox"]) == 4:
            # Get the original bbox from the *original* data for comparison
            original_ann = next(
                (
                    orig_ann
                    for orig_ann in coco_data.get("annotations", [])
                    if orig_ann.get("id") == ann.get("id")
                ),
                None,
            )
            # Use original annotation's bbox if found, otherwise current annotation's
            bbox_to_shrink = (
                original_ann["bbox"] if original_ann and original_ann.get("bbox") else ann["bbox"]
            )
            if not bbox_to_shrink or len(bbox_to_shrink) != 4: # Added check for valid bbox_to_shrink
                continue # Skip if no valid bbox found to shrink

            image_id = ann["image_id"]
            img_info = find_image_info(modified_coco_data, image_id)
            img_width = img_info.get("width") if img_info else None
            img_height = img_info.get("height") if img_info else None

            new_bbox = shrink_bbox(
                bbox_to_shrink, shrink_percent, img_width, img_height
            )

            if new_bbox != bbox_to_shrink:
                ann["bbox"] = new_bbox
                modified_image_ids.add(image_id)
                num_boxes_shrunk += 1


    if not modified_image_ids:
        print(
            "No valid bounding boxes found or no changes made after shrinking. No file saved."
        )
        return None


    print(
        f"Shrinking calculation complete. {num_boxes_shrunk} boxes potentially modified across {len(modified_image_ids)} images."
    )

    # --- Interactive Section ---
    accept_changes = False
    if interactive:
        print(
            "\nVisualizing changes on the first modified image (Original=Blue, Shrunk=Red)..."
        )
        first_modified_id = sorted(list(modified_image_ids))[0]

        img_info = find_image_info(coco_data, first_modified_id)

        image_filename = img_info.get("file_name")
        if not image_filename:
            raise ValueError(f"'file_name' missing for example image ID {first_modified_id}.")

        image_path = os.path.join(image_dir, image_filename) # image_dir guaranteed valid here

        try:
            img = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file not found at '{image_path}' for visualization.")
        except Exception as e:
            raise Exception(f"Error loading image '{image_path}' for visualization: {e}") from e

        draw = ImageDraw.Draw(img)
        original_annotations = find_annotations(coco_data, first_modified_id)
        current_modified_annotations = find_annotations(modified_coco_data, first_modified_id) # Use local var name

        print(
            f"--- Example Image (ID: {first_modified_id}, Filename: {image_filename}) ---"
        )
        print(
            f"Drawing {len(original_annotations)} original boxes (Blue) and {len(current_modified_annotations)} shrunk boxes (Red)."
        )

        # Draw original boxes
        num_orig_drawn = 0
        for ann_orig in original_annotations: # Use different var name
            bbox = ann_orig.get("bbox")
            if bbox and len(bbox) == 4:
                x_min, y_min, width, height = bbox
                x_max = x_min + width
                y_max = y_min + height
                draw.rectangle(
                    [x_min, y_min, x_max, y_max],
                    outline=original_box_color,
                    width=box_width,
                )
                num_orig_drawn += 1

        # Draw modified (shrunk) boxes
        num_shrunk_drawn = 0
        for ann_mod in current_modified_annotations: # Use different var name
            bbox = ann_mod.get("bbox")
            if bbox and len(bbox) == 4:
                x_min, y_min, width, height = bbox
                x_max = x_min + width
                y_max = y_min + height
                # Find corresponding original to check if it actually changed
                original_ann_match = next(
                    (
                        orig_ann
                        for orig_ann in original_annotations
                        if orig_ann.get("id") == ann_mod.get("id")
                    ),
                    None,
                )
                # Only draw red if it's different from original (or if original didn't exist)
                if not original_ann_match or original_ann_match.get("bbox") != bbox:
                    draw.rectangle(
                        [x_min, y_min, x_max, y_max],
                        outline=shrunk_box_color,
                        width=box_width,
                    )
                    num_shrunk_drawn += 1

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
                plt.close("all")
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
    print(f" - Final image count: {len(modified_coco_data['images'])}")
    print(f" - Final annotation count: {len(modified_coco_data['annotations'])}")
    print(f" - Bounding boxes shrunk by {shrink_percent}% in {len(modified_image_ids)} images.")

    # Proceed directly to saving
    print(f"\nSaving modified data to '{output_path}'...")
    try:
        with open(output_path, "w") as f:
            json.dump(modified_coco_data, f, indent=4)
        print("Save successful.")
        return str(output_path)
    except Exception as e:
        print(f"Error saving file: {e}")
        raise Exception(f"Error saving file: {e}") from e


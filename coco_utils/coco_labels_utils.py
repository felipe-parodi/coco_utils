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


def shrink_bbox(
    bbox: List[float],
    shrink_percent: float,
    img_width: Optional[int] = None,
    img_height: Optional[int] = None,
) -> List[float]:
    """Calculates new bounding box coordinates shrunk towards the center.

    Args:
        bbox: Original bounding box [x_min, y_min, width, height].
        shrink_percent: Percentage to shrink width and height (e.g., 5 for 5%).
        img_width: Optional image width for clamping coordinates.
        img_height: Optional image height for clamping coordinates.

    Returns:
        New bounding box [x_min, y_min, width, height]. Returns original if
        shrink_percent is invalid or causes non-positive dimensions.
    """
    if not (0 < shrink_percent < 100):
        print(
            f"Warning: Invalid shrink_percent {shrink_percent}. Must be between 0 and 100. Returning original bbox."
        )
        return bbox

    x_min, y_min, width, height = bbox
    shrink_factor = 1.0 - (shrink_percent / 100.0)

    new_width = width * shrink_factor
    new_height = height * shrink_factor

    if new_width <= 0 or new_height <= 0:
        print(
            f"Warning: Shrinking by {shrink_percent}% results in non-positive dimensions. Returning original bbox."
        )
        return bbox

    center_x = x_min + width / 2.0
    center_y = y_min + height / 2.0

    new_x_min = center_x - new_width / 2.0
    new_y_min = center_y - new_height / 2.0

    # Clamp coordinates to be within image bounds if provided, otherwise just non-negative
    new_x_min = max(0.0, new_x_min)
    new_y_min = max(0.0, new_y_min)

    if img_width is not None:
        new_x_min = min(new_x_min, img_width - 1.0)  # Ensure x_min is within width
        new_width = min(
            new_width, img_width - new_x_min
        )  # Adjust width if x_min+width exceeds boundary
    if img_height is not None:
        new_y_min = min(new_y_min, img_height - 1.0)  # Ensure y_min is within height
        new_height = min(
            new_height, img_height - new_y_min
        )  # Adjust height if y_min+height exceeds boundary

    # Ensure width and height remain positive after clamping
    new_width = max(1.0, new_width)  # Use 1.0 as minimum dimension
    new_height = max(1.0, new_height)

    # Round to a reasonable number of decimal places, matching typical COCO precision
    # Or keep as float if that's preferred. Let's keep float for now.
    # new_bbox = [round(v, 2) for v in [new_x_min, new_y_min, new_width, new_height]]
    new_bbox = [new_x_min, new_y_min, new_width, new_height]

    return new_bbox


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

        # image_dir is guaranteed to be valid here due to check at start
        image_path = os.path.join(image_dir, image_filename)

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

        # Confirmation Prompt
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
                print("Discarding all shrinkage changes.")
                plt.close("all")
                return None
            else:
                print("Invalid input. Please enter y or n.")
        plt.close("all")

    else:
        print("Running in non-interactive mode. All calculated changes will be applied.")
        accept_changes = True

    # --- Final Save Section ---
    if not accept_changes:
        print("Changes were not accepted.")
        return None

    # output_path is already a Path object and guaranteed non-None
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nFinal check:")
    print(f" - Original image count: {len(coco_data.get('images', []))}")
    print(f" - Original annotation count: {len(coco_data.get('annotations', []))}")
    print(f" - Final image count: {len(modified_coco_data['images'])} (Unaffected by shrinkage)")
    print(f" - Final annotation count: {len(modified_coco_data['annotations'])} (Unaffected by shrinkage)")
    print(f" - Bounding boxes shrunk by {shrink_percent}% in {len(modified_image_ids)} images.")


    save_confirmed = False
    if interactive:
         while True:
            save_confirm_input = (
                input(
                    f"Save the modified COCO data to '{output_path}'? (y/n): "
                )
                .lower()
                .strip()
            )
            if save_confirm_input == "y":
                save_confirmed = True
                break
            elif save_confirm_input == "n":
                print("Save cancelled. Discarding accepted changes.")
                return None
            else:
                print("Invalid input. Please enter y or n.")
    else:
        save_confirmed = True
        print(f"Saving modified data automatically to '{output_path}'...")

    if save_confirmed:
        try:
            with open(output_path, "w") as f:
                json.dump(modified_coco_data, f, indent=4) # Keep indent=4
            print("Save successful.")
            return str(output_path)
        except Exception as e:
            print(f"Error saving file: {e}")
            raise Exception(f"Error saving file: {e}") from e
    else:
         print("Save was not confirmed.")
         return None


def merge_coco_files(
    coco_paths: List[Union[str, Path]], output_path: Union[str, Path]
) -> None:
    """Merges multiple COCO-formatted JSON files into a single file.

    Handles re-assigning unique IDs for images and annotations, and merges
    categories based on their names. Assumes category definitions are
    consistent across files if names match.

    Args:
        coco_paths: A list of paths to the COCO JSON files to merge.
        output_path: The path where the merged COCO JSON file will be saved.
    """
    if not coco_paths:
        print("Error: No COCO file paths provided.")
        return

    merged_images: List[Dict[str, Any]] = []
    merged_annotations: List[Dict[str, Any]] = []
    merged_categories: List[Dict[str, Any]] = []

    # Track max IDs encountered so far across all files
    max_image_id = 0
    max_annotation_id = 0
    next_category_id = 1  # COCO IDs are typically 1-based

    # Map category NAME to its assigned ID in the merged dataset
    merged_category_name_to_id: Dict[str, int] = {}
    # Map category NAME to the actual category dict (to preserve skeleton etc.)
    merged_category_name_to_def: Dict[str, Dict] = {}

    print(f"Starting merge of {len(coco_paths)} COCO files...")

    for i, coco_path in enumerate(coco_paths):
        coco_path = Path(coco_path)  # Ensure it's a Path object
        print(f"Processing file {i + 1}/{len(coco_paths)}: {coco_path.name}...")
        try:
            with open(coco_path, "r") as f:
                current_coco = json.load(f)
        except FileNotFoundError:
            print(f"  Error: File not found: {coco_path}. Skipping.")
            continue
        except json.JSONDecodeError as e:
            print(f"  Error: Invalid JSON in file: {coco_path}. Skipping. ({e})")
            continue
        except Exception as e:
            print(f"  Error loading file {coco_path}: {e}. Skipping.")
            continue

        # --- Data Validation (Basic) ---
        if not all(k in current_coco for k in ["images", "annotations", "categories"]):
            print(
                f"  Warning: File {coco_path.name} is missing one of 'images', 'annotations', 'categories'. Skipping."
            )
            continue

        # --- Process Categories for the current file ---
        current_cat_id_map: Dict[
            int, int
        ] = {}  # Maps old cat ID (this file) -> new cat ID (merged)
        for cat in current_coco.get("categories", []):
            cat_name = cat.get("name")
            original_cat_id = cat.get("id")

            if not cat_name or original_cat_id is None:
                print(
                    f"  Warning: Skipping category with missing name or ID in {coco_path.name}: {cat}"
                )
                continue

            if cat_name in merged_category_name_to_id:
                # Category name already exists, map to existing merged ID
                merged_id = merged_category_name_to_id[cat_name]
                current_cat_id_map[original_cat_id] = merged_id
                # Optional: Check for consistency (e.g., supercategory, keypoints, skeleton)
                # existing_def = merged_category_name_to_def[cat_name]
                # if existing_def != cat: # Basic check, might need deep comparison
                #     print(f"  Warning: Category '{cat_name}' definition differs between files. Using definition from first encounter.")
            else:
                # New category name, assign new ID and add to merged list
                new_merged_id = next_category_id
                merged_category_name_to_id[cat_name] = new_merged_id
                merged_category_name_to_def[cat_name] = cat  # Store full definition

                current_cat_id_map[original_cat_id] = new_merged_id
                cat["id"] = new_merged_id  # Update the ID in the dict before appending
                merged_categories.append(cat)
                next_category_id += 1

        # --- Process Images and Annotations for the current file ---
        current_img_id_map: Dict[
            int, int
        ] = {}  # Maps old img ID (this file) -> new img ID (merged)
        current_file_image_count = 0
        current_file_annotation_count = 0

        for img in current_coco.get("images", []):
            original_img_id = img.get("id")
            if original_img_id is None:
                print(
                    f"  Warning: Skipping image with missing ID in {coco_path.name}: {img}"
                )
                continue

            new_image_id = max_image_id + 1
            max_image_id += 1

            current_img_id_map[original_img_id] = new_image_id
            img["id"] = new_image_id  # Update the ID in the dict
            merged_images.append(img)
            current_file_image_count += 1

        for ann in current_coco.get("annotations", []):
            original_ann_id = ann.get("id")
            original_img_id = ann.get("image_id")
            original_cat_id = ann.get("category_id")

            # Validate required fields and references
            if original_ann_id is None:
                print(
                    f"  Warning: Skipping annotation with missing ID in {coco_path.name}: {ann}"
                )
                continue
            if original_img_id not in current_img_id_map:
                print(
                    f"  Warning: Skipping annotation {original_ann_id} in {coco_path.name}: its image_id ({original_img_id}) was not found or skipped in this file's images."
                )
                continue
            if original_cat_id not in current_cat_id_map:
                print(
                    f"  Warning: Skipping annotation {original_ann_id} in {coco_path.name}: its category_id ({original_cat_id}) was not found or skipped in this file's categories."
                )
                continue

            new_annotation_id = max_annotation_id + 1
            max_annotation_id += 1

            ann["id"] = new_annotation_id  # Update the ID
            ann["image_id"] = current_img_id_map[
                original_img_id
            ]  # Update image reference
            ann["category_id"] = current_cat_id_map[
                original_cat_id
            ]  # Update category reference

            merged_annotations.append(ann)
            current_file_annotation_count += 1

        print(
            f"  Processed: {current_file_image_count} images, {current_file_annotation_count} annotations."
        )

    # --- Create and Save Final Merged COCO File ---
    if not merged_images and not merged_annotations:
        print(
            "Error: No valid images or annotations found after processing all files. Nothing to save."
        )
        return

    final_coco = {
        # Ensure categories are sorted by their new ID for consistency
        "categories": sorted(merged_categories, key=lambda x: x["id"]),
        "images": merged_images,
        "annotations": merged_annotations,
        # Add other top-level keys if desired (e.g., 'info', 'licenses') - requires handling potential conflicts
    }

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        print(f"\nSaving merged data to: {output_path}")
        with open(output_path, "w") as f:
            json.dump(final_coco, f, indent=2)  # Use indent for readability

        print("Merge complete.")
        print(f"Total Categories: {len(final_coco['categories'])}")
        print(f"Total Images:     {len(final_coco['images'])}")
        print(f"Total Annotations:{len(final_coco['annotations'])}")

    except Exception as e:
        print(f"Error saving final merged file: {e}")


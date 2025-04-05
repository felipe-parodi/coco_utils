"""
COCO File Operations Utilities.

This module provides functions for high-level operations on COCO JSON
files, such as merging multiple files into one or splitting a single
dataset into train, validation, and test sets.
"""

# coco_file_utils.py
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


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

            if original_ann_id is None:
                print(
                    f"  Warning: Skipping annotation with missing ID in {coco_path.name}: {ann}"
                )
                continue
            if original_img_id not in current_img_id_map:
                print(
                    f"  Warning: Skipping annotation {original_ann_id} in {coco_path.name}: its image_id ({original_img_id}) was not found or skipped."
                )
                continue
            if original_cat_id not in current_cat_id_map:
                print(
                    f"  Warning: Skipping annotation {original_ann_id} in {coco_path.name}: its category_id ({original_cat_id}) was not found or skipped."
                )
                continue

            new_annotation_id = max_annotation_id + 1
            max_annotation_id += 1

            ann["id"] = new_annotation_id
            ann["image_id"] = current_img_id_map[original_img_id]
            ann["category_id"] = current_cat_id_map[original_cat_id]

            merged_annotations.append(ann)
            current_file_annotation_count += 1

        print(
            f"  Processed: {current_file_image_count} images, {current_file_annotation_count} annotations."
        )

    if not merged_images and not merged_annotations:
        print(
            "Error: No valid images or annotations found after processing all files. Nothing to save."
        )
        return

    final_coco = {
        "categories": sorted(merged_categories, key=lambda x: x["id"]),
        "images": merged_images,
        "annotations": merged_annotations,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        print(f"\nSaving merged data to: {output_path}")
        with open(output_path, "w") as f:
            json.dump(final_coco, f, indent=2)

        print("Merge complete.")
        print(f"Total Categories: {len(final_coco['categories'])}")
        print(f"Total Images:     {len(final_coco['images'])}")
        print(f"Total Annotations:{len(final_coco['annotations'])}")

    except Exception as e:
        print(f"Error saving final merged file: {e}")


def split_coco_dataset(
    input_json_path: Union[str, Path],
    output_dir: Union[str, Path],
    images_dir: Union[str, Path],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    copy_images: bool = True,
    seed: Optional[int] = None,
) -> Optional[Tuple[Path, Path, Path]]:
    """
    Splits a COCO dataset into train, validation, and test sets.

    Creates train.json, val.json, test.json in the output directory and
    optionally copies corresponding images into train/val/test subdirectories.

    Args:
        input_json_path: Path to the input COCO JSON file.
        output_dir: Directory to save the output JSON files and image folders.
        images_dir: Path to the directory containing the source images.
        train_ratio: Proportion of images for the training set.
        val_ratio: Proportion of images for the validation set.
        test_ratio: Proportion of images for the test set.
                    (train_ratio + val_ratio + test_ratio must equal 1.0)
        copy_images: If True, copy images to corresponding train/val/test folders.
        seed: Optional random seed for shuffling images for reproducibility.

    Returns:
        A tuple containing the paths to the created train.json, val.json,
        and test.json files, or None if an error occurs.
    """
    input_json_path = Path(input_json_path)
    output_dir = Path(output_dir)
    images_dir = Path(images_dir)

    if not input_json_path.exists():
        print(f"Error: Input JSON file not found: {input_json_path}")
        return None
    if not images_dir.exists() and copy_images:
        # Only raise error if we intend to copy images and source doesn't exist
        print(f"Error: Source images directory not found: {images_dir}")
        return None
    if not (
        0.0 < train_ratio < 1.0 and 0.0 <= val_ratio < 1.0 and 0.0 <= test_ratio < 1.0
    ):
        print("Error: Ratios must be between 0 and 1.")
        return None
    if not abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-9:
        print(
            f"Error: Ratios must sum to 1.0 (Current sum: {train_ratio + val_ratio + test_ratio})"
        )
        return None

    # Load JSON data
    try:
        with open(input_json_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file {input_json_path}: {e}")
        return None

    if not all(k in data for k in ["images", "annotations", "categories"]):
        print(
            "Error: Input JSON is missing required keys: 'images', 'annotations', 'categories'."
        )
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories for images if copying
    train_img_dir = output_dir / "train"
    val_img_dir = output_dir / "val"
    test_img_dir = output_dir / "test"
    if copy_images:
        train_img_dir.mkdir(exist_ok=True)
        val_img_dir.mkdir(exist_ok=True)
        test_img_dir.mkdir(exist_ok=True)

    # Shuffle the image IDs
    image_ids = [img["id"] for img in data["images"]]
    if not image_ids:
        print("Error: No images found in the input JSON.")
        return None

    if seed is not None:
        random.seed(seed)
    random.shuffle(image_ids)

    # Calculate the number of images for each dataset
    total_images = len(image_ids)
    train_size = int(total_images * train_ratio)
    val_size = int(total_images * val_ratio)
    # Test size is the remainder to avoid floating point issues
    test_size = total_images - train_size - val_size

    if (
        train_size == 0
        or (val_ratio > 0 and val_size == 0)
        or (test_ratio > 0 and test_size == 0)
    ):
        print(
            "Warning: Ratios result in zero images for one or more splits. Adjust ratios or dataset size."
        )
        # Continue, but be aware splits might be empty

    # Split the image IDs
    train_image_ids = set(image_ids[:train_size])
    val_image_ids = set(image_ids[train_size : train_size + val_size])
    test_image_ids = set(image_ids[train_size + val_size :])

    # Split the images lists
    train_images = [img for img in data["images"] if img["id"] in train_image_ids]
    val_images = [img for img in data["images"] if img["id"] in val_image_ids]
    test_images = [img for img in data["images"] if img["id"] in test_image_ids]

    # Organize annotations by image_id for efficient splitting
    annotations_by_image_id = defaultdict(list)
    for ann in data["annotations"]:
        annotations_by_image_id[ann["image_id"]].append(ann)

    # Split the annotations
    train_annotations = [
        ann for img_id in train_image_ids for ann in annotations_by_image_id[img_id]
    ]
    val_annotations = [
        ann for img_id in val_image_ids for ann in annotations_by_image_id[img_id]
    ]
    test_annotations = [
        ann for img_id in test_image_ids for ann in annotations_by_image_id[img_id]
    ]

    # Create new JSON data dictionaries
    train_data = {
        "categories": data["categories"],  # Categories are shared
        "images": train_images,
        "annotations": train_annotations,
        # Consider adding 'info' and 'licenses' if they exist in original data
        **{
            k: v
            for k, v in data.items()
            if k not in ["categories", "images", "annotations"]
        },
    }
    val_data = {
        "categories": data["categories"],
        "images": val_images,
        "annotations": val_annotations,
        **{
            k: v
            for k, v in data.items()
            if k not in ["categories", "images", "annotations"]
        },
    }
    test_data = {
        "categories": data["categories"],
        "images": test_images,
        "annotations": test_annotations,
        **{
            k: v
            for k, v in data.items()
            if k not in ["categories", "images", "annotations"]
        },
    }

    # Save the new JSON files
    train_json_path = output_dir / "train.json"
    val_json_path = output_dir / "val.json"
    test_json_path = output_dir / "test.json"

    try:
        with open(train_json_path, "w") as f:
            json.dump(train_data, f, indent=4)
        with open(val_json_path, "w") as f:
            json.dump(val_data, f, indent=4)
        with open(test_json_path, "w") as f:
            json.dump(test_data, f, indent=4)
    except Exception as e:
        print(f"Error writing output JSON files: {e}")
        return None

    print(f"Saved JSON splits: {train_json_path}, {val_json_path}, {test_json_path}")

    # --- Copy images ---
    if copy_images:
        print("Copying images...")
        copied_counts = {"train": 0, "val": 0, "test": 0}
        skipped_counts = {"train": 0, "val": 0, "test": 0}

        def _copy_image_set(images, src_dir, dst_dir, split_name):
            for img in images:
                if "file_name" not in img or not img["file_name"]:
                    print(
                        f"Warning: Skipping image copy for {split_name} (id={img.get('id')}) - missing 'file_name'"
                    )
                    skipped_counts[split_name] += 1
                    continue

                # Handle potential subdirectories in file_name (though less common in COCO)
                file_name = Path(img["file_name"]).name  # Extract base filename
                src = src_dir / file_name  # Assume images are directly in images_dir
                # Alternative if file_name includes paths: src = src_dir / img["file_name"]

                dst = dst_dir / file_name

                if src.exists():
                    try:
                        # Create subdirectory in destination if needed (if file_name had path components)
                        # dst.parent.mkdir(parents=True, exist_ok=True) # Uncomment if needed
                        shutil.copy2(str(src), str(dst))  # copy2 preserves metadata
                        copied_counts[split_name] += 1
                    except Exception as e:
                        print(f"Error copying {src} to {dst}: {e}")
                        skipped_counts[split_name] += 1
                else:
                    print(f"Warning: Source image not found for {split_name}: {src}")
                    skipped_counts[split_name] += 1

        _copy_image_set(train_images, images_dir, train_img_dir, "train")
        _copy_image_set(val_images, images_dir, val_img_dir, "val")
        _copy_image_set(test_images, images_dir, test_img_dir, "test")

        print("Image copying complete.")
        print(
            f"  Train: Copied={copied_counts['train']}, Skipped/Errors={skipped_counts['train']}"
        )
        print(
            f"  Val:   Copied={copied_counts['val']}, Skipped/Errors={skipped_counts['val']}"
        )
        print(
            f"  Test:  Copied={copied_counts['test']}, Skipped/Errors={skipped_counts['test']}"
        )

    print(f"\nDataset split complete:")
    print(f"  Train images: {len(train_images)}, annotations: {len(train_annotations)}")
    print(f"  Val images:   {len(val_images)}, annotations: {len(val_annotations)}")
    print(f"  Test images:  {len(test_images)}, annotations: {len(test_annotations)}")

    return train_json_path, val_json_path, test_json_path

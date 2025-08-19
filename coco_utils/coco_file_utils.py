"""COCO File Operations Utilities.

This module provides functions for high-level operations on COCO JSON
files, such as merging multiple files into one or splitting a single
dataset into train, validation, and test sets.
"""

import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .coco_labels_utils import validate_annotations
from .exceptions import (
    FileOperationError,
    InvalidCOCOFormatError,
    ValidationError,
)
from .logger import get_logger

logger = get_logger(__name__)


def keep_first_n_images(
    input_json_path: Union[str, Path],
    output_json_path: Union[str, Path],
    num_images_to_keep: int,
    validate_before: bool = True,
    validate_after: bool = True,
) -> Dict[str, Any]:
    """Create a subset of a COCO dataset containing only the first N images.

    Creates a subset containing only the first N images and their annotations
    based on the order in the original JSON file.

    Args:
        input_json_path: Path to the input COCO JSON file.
        output_json_path: Path to save the subset COCO JSON file.
        num_images_to_keep: Number of images to keep from the beginning
            of the image list in the input JSON.
        validate_before: If True, validate input data before creating subset.
        validate_after: If True, validate subset data before saving.

    Returns:
        Dictionary containing the subset COCO data.

    Raises:
        FileNotFoundError: If the input JSON file doesn't exist.
        ValueError: If num_images_to_keep is not positive.
        InvalidCOCOFormatError: If the input JSON is missing required keys.
        ValidationError: If validation fails when enabled.
        FileOperationError: If there's an error saving the output file.

    Example:
        >>> subset_data = keep_first_n_images(
        ...     "coco_annotations.json",
        ...     "coco_subset.json",
        ...     num_images_to_keep=100
        ... )
        >>> print(f"Kept {len(subset_data['images'])} images")
    """
    input_json_path = Path(input_json_path)
    output_json_path = Path(output_json_path)

    if not input_json_path.exists():
        raise FileNotFoundError(f"Input JSON file not found: {input_json_path}")

    if num_images_to_keep <= 0:
        raise ValueError(f"Number of images to keep must be positive. Got {num_images_to_keep}")

    # Load JSON data
    try:
        logger.info(f"Loading input JSON: {input_json_path}")
        with open(input_json_path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise InvalidCOCOFormatError(f"Invalid JSON in file {input_json_path}: {e}")
    except Exception as e:
        raise FileOperationError(f"Error loading JSON file {input_json_path}: {e}")

    # Optional validation before subset
    if validate_before:
        logger.info("Validating input annotations before creating subset")
        try:
            is_valid, data = validate_annotations(data, fix=False, raise_error=True)
            logger.info("Input annotation validation successful")
        except ValueError as e:
            raise ValidationError(f"Input annotation validation failed: {e}")
        except Exception as e:
            raise ValidationError(f"Unexpected error during input validation: {e}")

    if not all(k in data for k in ["images", "annotations", "categories"]):
        raise InvalidCOCOFormatError(
            "Input JSON is missing required keys: 'images', 'annotations', 'categories'"
        )

    all_images = data.get("images", [])
    total_original_images = len(all_images)

    if total_original_images == 0:
        logger.warning("Input file contains no images. Output file will be empty.")
        num_actually_kept = 0
        kept_images = []
    elif num_images_to_keep >= total_original_images:
        logger.warning(
            f"Number to keep ({num_images_to_keep}) >= total images "
            f"({total_original_images}). Keeping all images."
        )
        num_actually_kept = total_original_images
        kept_images = all_images
    else:
        logger.info(f"Keeping the first {num_images_to_keep} images out of {total_original_images}")
        num_actually_kept = num_images_to_keep
        kept_images = all_images[:num_images_to_keep]

    # Create a set of the image IDs to keep for efficient lookup
    kept_image_ids = {img["id"] for img in kept_images if "id" in img}

    # Filter annotations to keep only those associated with the kept images
    kept_annotations = [
        ann for ann in data.get("annotations", []) if ann.get("image_id") in kept_image_ids
    ]

    # Create the subset data structure
    subset_data = {
        "categories": data.get("categories", []),
        "images": kept_images,
        "annotations": kept_annotations,
        **{k: v for k, v in data.items() if k not in ["categories", "images", "annotations"]},
    }

    # Optional validation after subset
    if validate_after:
        logger.info("Validating subset annotations before saving")
        try:
            is_valid, subset_data = validate_annotations(subset_data, fix=False, raise_error=True)
            logger.info("Subset annotation validation successful")
        except ValueError as e:
            raise ValidationError(f"Subset annotation validation failed: {e}")
        except Exception as e:
            raise ValidationError(f"Unexpected error during subset validation: {e}")

    # Save the subset JSON file
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        logger.info(f"Saving subset data to: {output_json_path}")
        with open(output_json_path, "w") as f:
            json.dump(subset_data, f, indent=2)

        logger.info(
            f"Subset creation complete. Kept {num_actually_kept} images "
            f"and {len(kept_annotations)} annotations."
        )

    except Exception as e:
        raise FileOperationError(f"Error saving subset JSON file: {e}")

    return subset_data


def _get_user_choice(prompt: str, num_options: int) -> int:
    """Get validated integer input from the user within a range.

    Args:
        prompt: The prompt to display to the user.
        num_options: The number of available options.

    Returns:
        Zero-based index of the user's choice.

    Raises:
        EOFError: If EOF is detected during input.
    """
    while True:
        try:
            choice = input(prompt).strip()
            if not choice:
                logger.warning("Input cannot be empty. Please enter a number.")
                continue
            choice_int = int(choice)
            if 1 <= choice_int <= num_options:
                return choice_int - 1
            else:
                logger.warning(
                    f"Invalid choice. Please enter a number between 1 and {num_options}."
                )
        except ValueError:
            logger.warning("Invalid input. Please enter a number.")
        except EOFError:
            logger.info("EOF detected, exiting selection.")
            raise


def merge_coco_files(
    coco_paths: List[Union[str, Path]],
    output_path: Union[str, Path],
    validate_after_merge: bool = True,
) -> Dict[str, Any]:
    """Merge multiple COCO-formatted JSON files into a single file.

    If images with the same 'file_name' are found in multiple input files,
    the user is prompted once to choose a single file that will take
    priority. For any duplicate filename, only the image and its associated
    annotations from the chosen priority file will be kept.

    Handles re-assigning unique IDs for images and annotations, and merges
    categories based on their names. Assumes category definitions are
    consistent across files if names match.

    Args:
        coco_paths: List of paths to the COCO JSON files to merge.
        output_path: Path where the merged COCO JSON file will be saved.
        validate_after_merge: If True, validate merged data before saving.

    Returns:
        Dictionary containing the merged COCO data.

    Raises:
        ValueError: If no COCO file paths are provided.
        FileNotFoundError: If any input file doesn't exist.
        InvalidCOCOFormatError: If any file has invalid JSON or missing keys.
        ValidationError: If validation fails when enabled.
        FileOperationError: If there's an error saving the output file.

    Example:
        >>> merged_data = merge_coco_files(
        ...     ["coco1.json", "coco2.json"],
        ...     "merged_coco.json"
        ... )
        >>> print(f"Merged {len(merged_data['images'])} images")
    """
    if not coco_paths:
        raise ValueError("No COCO file paths provided")

    resolved_coco_paths = [Path(p) for p in coco_paths]

    # Pass 1: Scan for duplicate filenames and get user choices
    logger.info("Pass 1: Scanning for duplicate filenames")
    filename_to_sources: Dict[str, List[Tuple[int, Path]]] = defaultdict(list)
    loaded_data_cache: Dict[Path, Dict] = {}

    for i, coco_path in enumerate(resolved_coco_paths):
        logger.info(f"Scanning file {i + 1}/{len(resolved_coco_paths)}: {coco_path.name}")
        try:
            with open(coco_path, "r") as f:
                current_coco = json.load(f)
                loaded_data_cache[coco_path] = current_coco
        except FileNotFoundError:
            logger.error(f"File not found: {coco_path}. Skipping.")
            continue
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in file: {coco_path}. Skipping. ({e})")
            continue
        except Exception as e:
            logger.error(f"Error loading file {coco_path}: {e}. Skipping.")
            continue

        if "images" not in current_coco:
            logger.warning(f"File {coco_path.name} is missing 'images' key. Skipping.")
            continue

        for img in current_coco.get("images", []):
            file_name = img.get("file_name")
            if file_name:
                filename_to_sources[file_name].append((i, coco_path))
            else:
                logger.warning(
                    f"Skipping image with missing 'file_name' in {coco_path.name}: "
                    f"{img.get('id', 'N/A')}"
                )

    duplicate_filenames = {
        name: sources for name, sources in filename_to_sources.items() if len(sources) > 1
    }
    priority_file_index: Optional[int] = None

    if duplicate_filenames:
        logger.info(f"Found {len(duplicate_filenames)} filenames present in multiple files")
        print("Please choose which file's images should take priority in case of duplicates:")
        for idx, file_path in enumerate(resolved_coco_paths):
            print(f"  {idx + 1}: {file_path.name}")

        try:
            choice_prompt = (
                f"Enter the number (1-{len(resolved_coco_paths)}) of the priority file: "
            )
            priority_file_index = _get_user_choice(choice_prompt, len(resolved_coco_paths))
            logger.info(
                f"Using file {priority_file_index + 1} "
                f"({resolved_coco_paths[priority_file_index].name}) as priority for duplicates"
            )

        except EOFError:
            logger.info("Merge aborted due to EOF during priority file selection")
            raise

    else:
        logger.info("No duplicate filenames found across files")

    # Pass 2: Merge data based on choices
    logger.info("Pass 2: Merging data")
    merged_images: List[Dict[str, Any]] = []
    merged_annotations: List[Dict[str, Any]] = []
    merged_categories: List[Dict[str, Any]] = []

    max_image_id = 0
    max_annotation_id = 0
    next_category_id = 1

    merged_category_name_to_id: Dict[str, int] = {}
    merged_category_name_to_def: Dict[str, Dict] = {}

    for i, coco_path in enumerate(resolved_coco_paths):
        logger.info(
            f"Processing file {i + 1}/{len(resolved_coco_paths)}: {coco_path.name} for merge"
        )

        # Use cached data if available, otherwise attempt to load again
        current_coco = loaded_data_cache.get(coco_path)
        if not current_coco:
            try:
                with open(coco_path, "r") as f:
                    current_coco = json.load(f)
            except Exception as e:
                logger.error(f"Error reloading file {coco_path} during merge pass: {e}. Skipping.")
                continue

        # Data validation
        if not all(k in current_coco for k in ["images", "annotations", "categories"]):
            logger.warning(
                f"File {coco_path.name} is missing one of 'images', 'annotations', "
                f"'categories'. Skipping merge for this file."
            )
            continue

        # Process Categories for the current file
        current_cat_id_map: Dict[int, int] = {}
        for cat in current_coco.get("categories", []):
            cat_name = cat.get("name")
            original_cat_id = cat.get("id")

            if not cat_name or original_cat_id is None:
                logger.warning(
                    f"Skipping category with missing name or ID in {coco_path.name}: {cat}"
                )
                continue

            if cat_name not in merged_category_name_to_id:
                # New category name, assign new ID and add to merged list
                new_merged_id = next_category_id
                merged_category_name_to_id[cat_name] = new_merged_id
                merged_category_name_to_def[cat_name] = cat
                current_cat_id_map[original_cat_id] = new_merged_id
                cat["id"] = new_merged_id
                merged_categories.append(cat)
                next_category_id += 1
            else:
                # Category name already exists, map to existing merged ID
                merged_id = merged_category_name_to_id[cat_name]
                current_cat_id_map[original_cat_id] = merged_id

        # Process Images and Annotations for the current file
        current_img_id_map: Dict[int, int] = {}
        current_file_kept_image_count = 0
        current_file_kept_annotation_count = 0

        # Process Images, applying duplicate choices
        for img in current_coco.get("images", []):
            original_img_id = img.get("id")
            file_name = img.get("file_name")

            if original_img_id is None:
                logger.warning(f"Skipping image with missing ID in {coco_path.name}: {img}")
                continue
            if file_name is None:
                continue

            keep_this_image = False
            is_duplicate = file_name in duplicate_filenames

            if is_duplicate:
                # It's a duplicate, only keep if from the priority file
                if priority_file_index is not None and i == priority_file_index:
                    keep_this_image = True
            else:
                # Not a duplicate, keep it
                keep_this_image = True

            if keep_this_image:
                new_image_id = max_image_id + 1
                max_image_id += 1
                current_img_id_map[original_img_id] = new_image_id
                img["id"] = new_image_id
                merged_images.append(img)
                current_file_kept_image_count += 1

        # Process Annotations, only keeping those linked to kept images
        for ann in current_coco.get("annotations", []):
            original_ann_id = ann.get("id")
            original_img_id = ann.get("image_id")
            original_cat_id = ann.get("category_id")

            if original_ann_id is None:
                logger.warning(f"Skipping annotation with missing ID in {coco_path.name}: {ann}")
                continue

            # Check if the image this annotation belongs to was kept
            if original_img_id in current_img_id_map:
                # Check if the category ID is valid and mapped
                if original_cat_id in current_cat_id_map:
                    new_annotation_id = max_annotation_id + 1
                    max_annotation_id += 1

                    ann["id"] = new_annotation_id
                    ann["image_id"] = current_img_id_map[original_img_id]
                    ann["category_id"] = current_cat_id_map[original_cat_id]

                    merged_annotations.append(ann)
                    current_file_kept_annotation_count += 1
                else:
                    logger.warning(
                        f"Skipping annotation {original_ann_id} in {coco_path.name}: "
                        f"its category_id ({original_cat_id}) was not found or skipped."
                    )

        logger.info(
            f"Merged from {coco_path.name}: {current_file_kept_image_count} images, "
            f"{current_file_kept_annotation_count} annotations"
        )

    # Final output validation
    if not merged_images and not merged_annotations:
        raise InvalidCOCOFormatError(
            "No valid images or annotations selected after processing all files"
        )

    # Ensure categories are sorted by their final ID
    final_merged_categories = sorted(merged_categories, key=lambda x: x["id"])

    # Prepare other top-level keys (info, licenses) to add to final dict
    other_keys = {}
    if loaded_data_cache:
        first_valid_path = next(iter(loaded_data_cache))
        other_keys = {
            k: v
            for k, v in loaded_data_cache[first_valid_path].items()
            if k not in ["images", "annotations", "categories"]
        }

    final_coco = {
        "categories": final_merged_categories,
        "images": merged_images,
        "annotations": merged_annotations,
        **other_keys,
    }

    # Optional validation before saving
    if validate_after_merge:
        logger.info("Validating final merged annotations")
        try:
            is_valid, final_coco = validate_annotations(final_coco, fix=False, raise_error=True)
            logger.info("Merged annotation validation successful")
        except ValueError as e:
            raise ValidationError(f"Merged annotation validation failed: {e}")
        except Exception as e:
            raise ValidationError(f"Unexpected error during final validation: {e}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(f"Saving merged data to: {output_path}")
        with open(output_path, "w") as f:
            json.dump(final_coco, f, indent=2)

        logger.info(
            f"Merge complete. Total - Categories: {len(final_coco['categories'])}, "
            f"Images: {len(final_coco['images'])}, "
            f"Annotations: {len(final_coco['annotations'])}"
        )

    except Exception as e:
        raise FileOperationError(f"Error saving final merged file: {e}")

    return final_coco


def split_coco_dataset(
    input_json_path: Union[str, Path],
    output_dir: Union[str, Path],
    images_dir: Union[str, Path],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    copy_images: bool = True,
    seed: Optional[int] = None,
    validate_before_split: bool = True,
) -> Tuple[Path, Path, Path]:
    """Split a COCO dataset into train, validation, and test sets.

    Creates train.json, val.json, test.json in the output directory and
    optionally copies corresponding images into train/val/test subdirectories.

    Args:
        input_json_path: Path to the input COCO JSON file.
        output_dir: Directory to save the output JSON files and image folders.
        images_dir: Path to the directory containing the source images.
        train_ratio: Proportion of images for the training set (default: 0.7).
        val_ratio: Proportion of images for the validation set (default: 0.15).
        test_ratio: Proportion of images for the test set (default: 0.15).
            Note: train_ratio + val_ratio + test_ratio must equal 1.0.
        copy_images: If True, copy images to train/val/test folders.
        seed: Optional random seed for reproducibility.
        validate_before_split: If True, validate input data before splitting.

    Returns:
        Tuple containing paths to the created train.json, val.json,
        and test.json files.

    Raises:
        FileNotFoundError: If input JSON or images directory doesn't exist.
        ValueError: If ratios are invalid or don't sum to 1.0.
        InvalidCOCOFormatError: If input JSON is missing required keys.
        ValidationError: If validation fails when enabled.
        FileOperationError: If there's an error writing output files.

    Example:
        >>> train_path, val_path, test_path = split_coco_dataset(
        ...     "coco_annotations.json",
        ...     "output_splits",
        ...     "images",
        ...     train_ratio=0.7,
        ...     val_ratio=0.2,
        ...     test_ratio=0.1
        ... )
        >>> print(f"Created splits: {train_path}, {val_path}, {test_path}")
    """
    input_json_path = Path(input_json_path)
    output_dir = Path(output_dir)
    images_dir = Path(images_dir)

    if not input_json_path.exists():
        raise FileNotFoundError(f"Input JSON file not found: {input_json_path}")

    if not images_dir.exists() and copy_images:
        raise FileNotFoundError(f"Source images directory not found: {images_dir}")

    if not (0.0 < train_ratio < 1.0 and 0.0 <= val_ratio < 1.0 and 0.0 <= test_ratio < 1.0):
        raise ValueError("Ratios must be between 0 and 1")

    if not abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-9:
        raise ValueError(
            f"Ratios must sum to 1.0 (Current sum: {train_ratio + val_ratio + test_ratio})"
        )

    # Load JSON data
    try:
        logger.info(f"Loading input JSON: {input_json_path}")
        with open(input_json_path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise InvalidCOCOFormatError(f"Invalid JSON in file {input_json_path}: {e}")
    except Exception as e:
        raise FileOperationError(f"Error loading JSON file {input_json_path}: {e}")

    # Optional validation after loading
    if validate_before_split:
        logger.info("Validating input annotations before splitting")
        try:
            is_valid, data = validate_annotations(data, fix=False, raise_error=True)
            logger.info("Input annotation validation successful")
        except ValueError as e:
            raise ValidationError(f"Input annotation validation failed: {e}")
        except Exception as e:
            raise ValidationError(f"Unexpected error during input validation: {e}")

    if not all(k in data for k in ["images", "annotations", "categories"]):
        raise InvalidCOCOFormatError(
            "Input JSON is missing required keys: 'images', 'annotations', 'categories'"
        )

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
        raise InvalidCOCOFormatError("No images found in the input JSON")

    if seed is not None:
        random.seed(seed)
    random.shuffle(image_ids)

    # Calculate the number of images for each dataset
    total_images = len(image_ids)
    train_size = int(total_images * train_ratio)
    val_size = int(total_images * val_ratio)
    test_size = total_images - train_size - val_size

    if train_size == 0 or (val_ratio > 0 and val_size == 0) or (test_ratio > 0 and test_size == 0):
        logger.warning(
            "Ratios result in zero images for one or more splits. " "Adjust ratios or dataset size."
        )

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
    val_annotations = [ann for img_id in val_image_ids for ann in annotations_by_image_id[img_id]]
    test_annotations = [ann for img_id in test_image_ids for ann in annotations_by_image_id[img_id]]

    # Create new JSON data dictionaries
    train_data = {
        "categories": data["categories"],
        "images": train_images,
        "annotations": train_annotations,
        **{k: v for k, v in data.items() if k not in ["categories", "images", "annotations"]},
    }
    val_data = {
        "categories": data["categories"],
        "images": val_images,
        "annotations": val_annotations,
        **{k: v for k, v in data.items() if k not in ["categories", "images", "annotations"]},
    }
    test_data = {
        "categories": data["categories"],
        "images": test_images,
        "annotations": test_annotations,
        **{k: v for k, v in data.items() if k not in ["categories", "images", "annotations"]},
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
        raise FileOperationError(f"Error writing output JSON files: {e}")

    logger.info(f"Saved JSON splits: {train_json_path}, {val_json_path}, {test_json_path}")

    # Copy images if requested
    if copy_images:
        logger.info("Copying images to split directories")
        copied_counts = {"train": 0, "val": 0, "test": 0}
        skipped_counts = {"train": 0, "val": 0, "test": 0}

        def _copy_image_set(images, src_dir, dst_dir, split_name):
            for img in images:
                if "file_name" not in img or not img["file_name"]:
                    logger.warning(
                        f"Skipping image copy for {split_name} (id={img.get('id')}) - "
                        f"missing 'file_name'"
                    )
                    skipped_counts[split_name] += 1
                    continue

                # Handle potential subdirectories in file_name
                file_name = Path(img["file_name"]).name
                src = src_dir / file_name
                dst = dst_dir / file_name

                if src.exists():
                    try:
                        shutil.copy2(str(src), str(dst))
                        copied_counts[split_name] += 1
                    except Exception as e:
                        logger.error(f"Error copying {src} to {dst}: {e}")
                        skipped_counts[split_name] += 1
                else:
                    logger.warning(f"Source image not found for {split_name}: {src}")
                    skipped_counts[split_name] += 1

        _copy_image_set(train_images, images_dir, train_img_dir, "train")
        _copy_image_set(val_images, images_dir, val_img_dir, "val")
        _copy_image_set(test_images, images_dir, test_img_dir, "test")

        logger.info(
            f"Image copying complete. "
            f"Train: Copied={copied_counts['train']}, Skipped/Errors={skipped_counts['train']}, "
            f"Val: Copied={copied_counts['val']}, Skipped/Errors={skipped_counts['val']}, "
            f"Test: Copied={copied_counts['test']}, Skipped/Errors={skipped_counts['test']}"
        )

    logger.info(
        f"Dataset split complete. "
        f"Train: {len(train_images)} images, {len(train_annotations)} annotations, "
        f"Val: {len(val_images)} images, {len(val_annotations)} annotations, "
        f"Test: {len(test_images)} images, {len(test_annotations)} annotations"
    )

    return train_json_path, val_json_path, test_json_path

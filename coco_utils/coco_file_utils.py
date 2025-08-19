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

# Import the validation function
from .coco_labels_utils import validate_annotations



def keep_first_n_images(
    input_json_path: Union[str, Path],
    output_json_path: Union[str, Path],
    num_images_to_keep: int,
    validate_before: bool = True,
    validate_after: bool = True,
) -> None:
    """
    Creates a subset of a COCO dataset containing only the first N images
    (and their annotations) based on the order in the original JSON file.

    Args:
        input_json_path: Path to the input COCO JSON file.
        output_json_path: Path to save the subset COCO JSON file.
        num_images_to_keep: The number of images to keep from the beginning
                              of the image list in the input JSON.
        validate_before: If True, run annotation validation on the input
                         data before creating the subset. Raises error if invalid.
        validate_after: If True, run annotation validation on the final
                        subset data before saving. Raises error if invalid.
    """
    input_json_path = Path(input_json_path)
    output_json_path = Path(output_json_path)

    if not input_json_path.exists():
        print(f"Error: Input JSON file not found: {input_json_path}")
        return
    if num_images_to_keep <= 0:
        print(f"Error: Number of images to keep must be positive. Got {num_images_to_keep}")
        return

    # Load JSON data
    try:
        print(f"Loading input JSON: {input_json_path}")
        with open(input_json_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file {input_json_path}: {e}")
        return

    # --- Optional Validation Before Subset --- #
    if validate_before:
        print("--- Validating input annotations before creating subset ---")
        try:
            is_valid, data = validate_annotations(
                data,
                fix=False, # Don't fix the input data automatically
                raise_error=True # Halt if validation fails
            )
            print("--- Input annotation validation successful ---")
        except ValueError as e:
            print(f"Error: Input annotation validation failed: {e}")
            print("Subset creation aborted due to validation errors.")
            return
        except Exception as e:
            print(f"An unexpected error occurred during input validation: {e}")
            print("Subset creation aborted.")
            return

    if not all(k in data for k in ["images", "annotations", "categories"]):
        print(
            "Error: Input JSON is missing required keys: 'images', 'annotations', 'categories'."
        )
        return

    all_images = data.get("images", [])
    total_original_images = len(all_images)

    if total_original_images == 0:
        print("Warning: Input file contains no images. Output file will be empty.")
        num_actually_kept = 0
        kept_images = []
    elif num_images_to_keep >= total_original_images:
        print(f"Warning: Number to keep ({num_images_to_keep}) >= total images ({total_original_images}). Keeping all images.")
        num_actually_kept = total_original_images
        kept_images = all_images # Keep all
    else:
        print(f"Keeping the first {num_images_to_keep} images out of {total_original_images}.")
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
        "categories": data.get("categories", []), # Keep original categories
        "images": kept_images,
        "annotations": kept_annotations,
        # Keep other top-level keys like 'info', 'licenses'
        **{k: v for k, v in data.items() if k not in ["categories", "images", "annotations"]}
    }

    # --- Optional Validation After Subset --- #
    if validate_after:
        print("--- Validating subset annotations before saving ---")
        try:
            is_valid, subset_data = validate_annotations(
                subset_data,
                fix=False, # Don't fix the subset automatically
                raise_error=True # Halt if validation fails
            )
            print("--- Subset annotation validation successful ---")
        except ValueError as e:
            print(f"Error: Subset annotation validation failed: {e}")
            print("Saving aborted due to validation errors.")
            return
        except Exception as e:
            print(f"An unexpected error occurred during subset validation: {e}")
            print("Saving aborted.")
            return

    # Save the subset JSON file
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        print(f"Saving subset data to: {output_json_path}")
        with open(output_json_path, "w") as f:
            json.dump(subset_data, f, indent=2) # Use indent=2 for better readability

        print("\nSubset creation complete.")
        print(f"Kept {num_actually_kept} images.")
        print(f"Kept {len(kept_annotations)} annotations.")

    except Exception as e:
        print(f"Error saving subset JSON file: {e}")


# Helper function for user input validation
def _get_user_choice(prompt: str, num_options: int) -> int:
    """Gets validated integer input from the user within a range."""
    while True:
        try:
            choice = input(prompt).strip()
            if not choice: # Handle empty input
                print("  Input cannot be empty. Please enter a number.")
                continue
            choice_int = int(choice)
            if 1 <= choice_int <= num_options:
                return choice_int - 1  # Return 0-based index
            else:
                print(f"  Invalid choice. Please enter a number between 1 and {num_options}.")
        except ValueError:
            print("  Invalid input. Please enter a number.")
        except EOFError:
            print("\n  EOF detected, exiting selection.")
            raise # Re-raise to signal exit if needed upstream


def merge_coco_files(
    coco_paths: List[Union[str, Path]], output_path: Union[str, Path], validate_after_merge: bool = True
) -> None:
    # TODO: add option to move images to one dir similar to copy_images in split_coco_dataset?
    """Merges multiple COCO-formatted JSON files into a single file.

    If images with the same 'file_name' are found in multiple input files,
    the user is prompted *once* to choose a single file that will take
    priority. For any duplicate filename, only the image (and its associated
    annotations) from the chosen priority file will be kept.

    Handles re-assigning unique IDs for images and annotations, and merges
    categories based on their names. Assumes category definitions are
    consistent across files if names match.

    Args:
        coco_paths: A list of paths to the COCO JSON files to merge.
        output_path: The path where the merged COCO JSON file will be saved.
        validate_after_merge: If True, run annotation validation on the final
                              merged data before saving. Raises error if invalid.
    """
    if not coco_paths:
        print("Error: No COCO file paths provided.")
        return

    resolved_coco_paths = [Path(p) for p in coco_paths] # Resolve paths early

    # --- Pass 1: Scan for duplicate filenames and get user choices ---
    print("--- Pass 1: Scanning for duplicate filenames ---")
    filename_to_sources: Dict[str, List[Tuple[int, Path]]] = defaultdict(list)
    loaded_data_cache: Dict[Path, Dict] = {} # Cache loaded data to avoid reloading in pass 2

    for i, coco_path in enumerate(resolved_coco_paths):
        print(f"Scanning file {i + 1}/{len(resolved_coco_paths)}: {coco_path.name}...")
        try:
            with open(coco_path, "r") as f:
                current_coco = json.load(f)
                loaded_data_cache[coco_path] = current_coco # Cache the data
        except FileNotFoundError:
            print(f"  Error: File not found: {coco_path}. Skipping.")
            continue
        except json.JSONDecodeError as e:
            print(f"  Error: Invalid JSON in file: {coco_path}. Skipping. ({e})")
            continue
        except Exception as e:
            print(f"  Error loading file {coco_path}: {e}. Skipping.")
            continue

        if "images" not in current_coco:
            print(f"  Warning: File {coco_path.name} is missing 'images' key. Skipping.")
            continue

        for img in current_coco.get("images", []):
            file_name = img.get("file_name")
            if file_name:
                filename_to_sources[file_name].append((i, coco_path))
            else:
                 print(f"  Warning: Skipping image with missing 'file_name' in {coco_path.name}: {img.get('id', 'N/A')}")


    duplicate_filenames = {
        name: sources
        for name, sources in filename_to_sources.items()
        if len(sources) > 1
    }
    filename_to_choice: Dict[str, int] = {} # Stores chosen source index for duplicates
    priority_file_index: Optional[int] = None # Stores the single user-chosen priority index

    if duplicate_filenames:
        print(f"\nFound {len(duplicate_filenames)} filenames present in multiple files.")
        print("Please choose which file's images should take priority in case of duplicates:")
        prompt_lines = []
        for idx, file_path in enumerate(resolved_coco_paths):
             prompt_lines.append(f"  {idx + 1}: {file_path.name}")
        print("\n".join(prompt_lines))

        try:
            choice_prompt = f"Enter the number (1-{len(resolved_coco_paths)}) of the priority file: "
            priority_file_index = _get_user_choice(choice_prompt, len(resolved_coco_paths))
            print(f"\n--- Using file {priority_file_index + 1} ({resolved_coco_paths[priority_file_index].name}) as priority for duplicates. ---")

        except EOFError:
             print("\nMerge aborted due to EOF during priority file selection.")
             return # Exit the function if user signals EOF

        # Commenting out per-file choice logic
        # try:
        #     for filename, sources in duplicate_filenames.items():
        #         print(f"\nDuplicate filename: '{filename}' found in:")
        #         prompt_lines = []
        #         for idx, (file_index, file_path) in enumerate(sources):
        #             prompt_lines.append(f"  {idx + 1}: {file_path.name} (File {file_index + 1})")
        #         print("\n".join(prompt_lines))
        #         choice_prompt = f"Enter the number (1-{len(sources)}) of the file to keep for '{filename}': "
        #         chosen_source_idx_in_list = _get_user_choice(choice_prompt, len(sources))
        #         # Store the *original file index* (0-based) from coco_paths
        #         filename_to_choice[filename] = sources[chosen_source_idx_in_list][0]
        # except EOFError:
        #      print("\nMerge aborted due to EOF during duplicate selection.")
        #      return # Exit the function if user signals EOF
        # print("\n--- Finished duplicate resolution ---")
    else:
        print("\nNo duplicate filenames found across files.")

    # --- Pass 2: Merge data based on choices ---
    print("\n--- Pass 2: Merging data ---")
    merged_images: List[Dict[str, Any]] = []
    merged_annotations: List[Dict[str, Any]] = []
    merged_categories: List[Dict[str, Any]] = []

    max_image_id = 0
    max_annotation_id = 0
    next_category_id = 1

    merged_category_name_to_id: Dict[str, int] = {}
    # Store full category def to preserve potential extra keys like 'skeleton'
    merged_category_name_to_def: Dict[str, Dict] = {}


    for i, coco_path in enumerate(resolved_coco_paths):
        print(f"Processing file {i + 1}/{len(resolved_coco_paths)}: {coco_path.name} for merge...")

        # Use cached data if available, otherwise attempt to load again (should usually be cached)
        current_coco = loaded_data_cache.get(coco_path)
        if not current_coco:
             # This should ideally not happen if scan was successful, but handle defensively
            try:
                with open(coco_path, "r") as f:
                    current_coco = json.load(f)
            except Exception as e:
                 print(f"  Error reloading file {coco_path} during merge pass: {e}. Skipping.")
                 continue # Skip this file in the merge phase


        # --- Data Validation (Basic) ---
        if not all(k in current_coco for k in ["images", "annotations", "categories"]):
            print(
                f"  Warning: File {coco_path.name} is missing one of 'images', 'annotations', 'categories'. Skipping merge for this file."
            )
            continue

        # --- Process Categories for the current file ---
        # This logic remains the same: build the global category list
        # and map original category IDs for *this file* to the final merged category IDs.
        current_cat_id_map: Dict[int, int] = {}
        for cat in current_coco.get("categories", []):
            cat_name = cat.get("name")
            original_cat_id = cat.get("id")

            if not cat_name or original_cat_id is None:
                print(
                    f"  Warning: Skipping category with missing name or ID in {coco_path.name}: {cat}"
                )
                continue

            if cat_name not in merged_category_name_to_id:
                # New category name, assign new ID and add to merged list
                new_merged_id = next_category_id
                merged_category_name_to_id[cat_name] = new_merged_id
                merged_category_name_to_def[cat_name] = cat # Store full definition
                current_cat_id_map[original_cat_id] = new_merged_id
                # Update the category dict *before* potentially adding it
                cat["id"] = new_merged_id
                merged_categories.append(cat) # Add the updated category object
                next_category_id += 1
            else:
                 # Category name already exists, map to existing merged ID
                merged_id = merged_category_name_to_id[cat_name]
                current_cat_id_map[original_cat_id] = merged_id
                # Ensure the definition in merged_categories matches the first one encountered
                # (or implement a more complex merge if definitions can differ)
                if cat_name in merged_category_name_to_def:
                    # Optional: Could add checks here if supercategory/skeleton differs
                    pass


        # --- Process Images and Annotations for the current file ---
        # Map original img ID (this file) -> new merged img ID (only for kept images)
        current_img_id_map: Dict[int, int] = {}
        current_file_kept_image_count = 0
        current_file_kept_annotation_count = 0

        # Process Images, applying duplicate choices
        for img in current_coco.get("images", []):
            original_img_id = img.get("id")
            file_name = img.get("file_name")

            if original_img_id is None:
                print(f"  Warning: Skipping image with missing ID in {coco_path.name}: {img}")
                continue
            if file_name is None:
                 # Already warned during scan pass, but double-check
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
                current_img_id_map[original_img_id] = new_image_id # Map original ID -> new ID
                img["id"] = new_image_id  # Update the ID in the image dict
                merged_images.append(img) # Add the modified image dict
                current_file_kept_image_count += 1

        # Process Annotations, only keeping those linked to kept images
        for ann in current_coco.get("annotations", []):
            original_ann_id = ann.get("id")
            original_img_id = ann.get("image_id")
            original_cat_id = ann.get("category_id")

            if original_ann_id is None:
                print(f"  Warning: Skipping annotation with missing ID in {coco_path.name}: {ann}")
                continue

            # Check if the image this annotation belongs to was kept
            if original_img_id in current_img_id_map:
                # Check if the category ID is valid and mapped
                if original_cat_id in current_cat_id_map:
                    new_annotation_id = max_annotation_id + 1
                    max_annotation_id += 1

                    ann["id"] = new_annotation_id
                    ann["image_id"] = current_img_id_map[original_img_id] # Use new image ID
                    ann["category_id"] = current_cat_id_map[original_cat_id] # Use new category ID

                    merged_annotations.append(ann)
                    current_file_kept_annotation_count += 1
                else:
                    # Image was kept, but category was bad/skipped earlier
                     print(f"  Warning: Skipping annotation {original_ann_id} in {coco_path.name}: its category_id ({original_cat_id}) was not found or skipped.")
            # Else: Annotation's image was not kept (either skipped or duplicate from another file), so silently skip annotation


        print(
            f"  Merged from this file: {current_file_kept_image_count} images, {current_file_kept_annotation_count} annotations."
        )

    # --- Final Output ---
    if not merged_images and not merged_annotations:
        print(
            "\nError: No valid images or annotations selected after processing all files. Nothing to save."
        )
        return

    # Ensure categories are sorted by their final ID
    final_merged_categories = sorted(merged_categories, key=lambda x: x["id"])

    # Prepare other top-level keys (info, licenses) to add to final dict
    other_keys = {}
    if loaded_data_cache: # Check if cache is not empty
        # Try to get keys from the first successfully loaded file
        first_valid_path = next(iter(loaded_data_cache))
        other_keys = { k: v for k, v in loaded_data_cache[first_valid_path].items()
                       if k not in ['images', 'annotations', 'categories'] }

    final_coco = {
        "categories": final_merged_categories,
        "images": merged_images,
        "annotations": merged_annotations,
        # Add other top-level keys if needed (e.g., 'info', 'licenses')
        # This requires inspecting the first file or merging logic for these keys.
        **other_keys
    }

    # --- Optional Validation Before Saving ---
    if validate_after_merge:
        print("\n--- Validating final merged annotations ---")
        try:
            # Use raise_error=True to halt if validation fails
            is_valid, final_coco = validate_annotations(
                final_coco,
                fix=False, # Typically don't fix automatically here, just report/halt
                raise_error=True
            )
            # If validate_annotations doesn't raise an error, it passed.
            print("--- Merged annotation validation successful ---")
        except ValueError as e:
            print(f"\nError: Merged annotation validation failed: {e}")
            print("Merge aborted before saving due to validation errors.")
            return # Stop before saving
        except Exception as e:
            # Catch other potential errors during validation
            print(f"\nAn unexpected error occurred during final validation: {e}")
            print("Merge aborted before saving.")
            return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        print(f"\nSaving merged data to: {output_path}")
        with open(output_path, "w") as f:
            json.dump(final_coco, f, indent=2) # Use indent=2 for readability like original

        print("\nMerge complete.")
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
    validate_before_split: bool = True,
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
        validate_before_split: If True, run annotation validation on the input
                             data before attempting the split. Raises error if invalid.

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

    # --- Optional Validation After Loading ---
    if validate_before_split:
        print("\n--- Validating input annotations before splitting ---")
        try:
            is_valid, data = validate_annotations(
                data,
                fix=False, # Don't fix the input data automatically before splitting
                raise_error=True # Halt if validation fails
            )
            # If validate_annotations doesn't raise an error, it passed.
            print("--- Input annotation validation successful ---")
        except ValueError as e:
            print(f"\nError: Input annotation validation failed: {e}")
            print("Splitting aborted due to validation errors.")
            return None # Stop before splitting
        except Exception as e:
            # Catch other potential errors during validation
            print(f"\nAn unexpected error occurred during input validation: {e}")
            print("Splitting aborted.")
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

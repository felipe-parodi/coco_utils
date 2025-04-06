# `coco_utils` Package Modules

This directory contains the core Python modules for the `coco_utils` package.

## Purpose

The `coco_utils` package provides utilities to simplify common tasks when working with datasets in the COCO format, such as visualization, label manipulation, and file operations.

## Modules Overview

*   **`coco_viz_utils.py`**:
    *   Provides functions for visualizing COCO data.
    *   Key functions: `visualize_bbox`, `visualize_keypoints`.
    *   Depends on `matplotlib` and `Pillow`.

*   **`coco_labels_utils.py`**:
    *   Focuses on modifying label data within a loaded COCO dictionary structure.
    *   Key functions: `shrink_coco_bboxes` (with interactive option), `shrink_bbox`.
    *   Relies on `coco_viz_utils` for interactive feedback components.

*   **`coco_file_utils.py`**:
    *   Handles operations acting on entire COCO JSON files.
    *   Key functions: `merge_coco_files`, `split_coco_dataset`.

## Installation & Usage

Refer to the `README.md` and `setup.py` files in the parent directory (`../`) for instructions on how to install this package and for detailed usage examples.

## License

This package is distributed under the MIT License. See the `LICENSE` file in this directory for details. 
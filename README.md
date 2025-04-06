# COCO Utilities (`coco_utils`)

This package provides a collection of Python utilities for working with datasets in the COCO (Common Objects in Context) format. It helps with visualizing annotations, manipulating label data, and performing common file-level operations like merging and splitting datasets.

## Installation

You can install this package directly from GitHub using pip:

```bash
pip install git+https://github.com/felipe-parodi/coco_utils.git
```

Alternatively, clone the repository and install in editable mode for development:

```bash
git clone https://github.com/felipe-parodi/coco_utils.git
cd coco_utils
pip install -e .
```

## Modules

*   **`coco_utils.coco_viz_utils`**: Contains functions for visualizing annotations (bounding boxes, keypoints) on images.
*   **`coco_utils.coco_labels_utils`**: Includes functions for modifying annotation data *within* a loaded COCO structure (e.g., shrinking bounding boxes interactively).
*   **`coco_utils.coco_file_utils`**: Offers functions for operating on COCO JSON files themselves (e.g., merging multiple COCO files, splitting a dataset into train/val/test).

## Core Functionality Examples

### Visualization (`coco_viz_utils.py`)

```python
import json

from coco_utils import visualize_bbox, visualize_keypoints

# Load data
coco_file = 'path/to/your/annotations.json'
image_dir = 'path/to/your/images/'
with open(coco_file, 'r') as f:
    coco_data = json.load(f)

image_id_to_show = 123 # Example image ID

# Visualize all bounding boxes for an image
visualize_bbox(coco_data, image_id_to_show, image_dir)

# Visualize keypoints (and skeleton, if defined) for an image
visualize_keypoints(coco_data, image_id_to_show, image_dir)
```

### Label Manipulation (`coco_labels_utils.py`)

```python
import json
from coco_utils import shrink_coco_bboxes

# Load data
coco_file = 'path/to/your/annotations.json'
image_dir = 'path/to/your/images/'
output_dir = 'path/to/save/shrunk/annotations/'
with open(coco_file, 'r') as f:
    coco_data = json.load(f)

# Shrink all bounding boxes by 10%, visualize the first change,
# ask for confirmation, and save if approved.
saved_path = shrink_coco_bboxes_interactive(
    coco_data,
    image_dir,
    shrink_percent=10.0,
    json_output_dir=output_dir,
    output_filename="coco_shrunk_10pct.json"
)

if saved_path:
    print(f"Shrunk annotations saved to: {saved_path}")
```

### File Operations (`coco_file_utils.py`)

**Merging Files:**

```python

from coco_utils import merge_coco_files
    
coco_files_to_merge = [
    'path/to/coco1.json',
    'path/to/coco2.json',
    'path/to/another_coco.json'
]
output_merged_file = 'path/to/save/merged_coco.json'

merge_coco_files(coco_files_to_merge, output_merged_file)
```

**Splitting Dataset:**

```python
from coco_utils import split_coco_dataset

input_coco_file = 'path/to/full_dataset.json'
output_split_dir = 'path/to/output/splits/'
source_image_dir = 'path/to/all/images/'

# Split into 70% train, 15% val, 15% test and copy images
train_path, val_path, test_path = split_coco_dataset(
    input_json_path=input_coco_file,
    output_dir=output_split_dir,
    images_dir=source_image_dir,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    copy_images=True,
    seed=42 # for reproducible splits
)

if train_path:
    print(f"Dataset split complete. Train JSON: {train_path}")
```

## Next Steps

*   **Testing:** Add unit tests for core utility functions (especially merging and splitting logic).
*   **Error Handling:** Enhance error handling and reporting in file operations (e.g., provide more detail on skipping malformed annotations/images).
*   **Documentation:** Expand function docstrings with more detail on parameters and potential edge cases. Consider generating Sphinx documentation.
*   **Consistency Checks:** Add optional stricter checks in `merge_coco_files` for category definition consistency (e.g., keypoints, skeleton).
*   **New Utilities:**
    *   Function to filter COCO data (e.g., keep only specific categories, remove images without annotations).
    *   Function to analyze dataset statistics (e.g., category distribution, image sizes).
    *   Add sampling options back to `merge_coco_files`.
    *   Support for other annotation formats or conversions.
*   **Efficiency:** Profile and optimize file operations for very large datasets. 
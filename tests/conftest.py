"""Pytest configuration and fixtures for coco_utils tests."""

import json
import tempfile
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def sample_coco_data() -> Dict[str, Any]:
    """Create sample COCO format data for testing.

    Returns:
        A dictionary containing sample COCO data with images, annotations, and categories.
    """
    return {
        "info": {
            "description": "Test COCO dataset",
            "version": "1.0",
            "year": 2024,
            "contributor": "coco_utils tests",
        },
        "licenses": [{"id": 1, "name": "Test License", "url": "http://example.com"}],
        "images": [
            {
                "id": 1,
                "file_name": "test_image_1.jpg",
                "width": 640,
                "height": 480,
            },
            {
                "id": 2,
                "file_name": "test_image_2.jpg",
                "width": 800,
                "height": 600,
            },
            {
                "id": 3,
                "file_name": "test_image_3.jpg",
                "width": 1024,
                "height": 768,
            },
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [100, 100, 50, 50],
                "area": 2500,
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 2,
                "bbox": [200, 200, 100, 100],
                "area": 10000,
                "iscrowd": 0,
            },
            {
                "id": 3,
                "image_id": 2,
                "category_id": 1,
                "bbox": [150, 150, 75, 75],
                "area": 5625,
                "iscrowd": 0,
            },
        ],
        "categories": [
            {
                "id": 1,
                "name": "cat",
                "supercategory": "animal",
            },
            {
                "id": 2,
                "name": "dog",
                "supercategory": "animal",
            },
        ],
    }


@pytest.fixture
def sample_coco_with_keypoints() -> Dict[str, Any]:
    """Create sample COCO data with keypoint annotations.

    Returns:
        A dictionary containing COCO data with keypoint annotations.
    """
    return {
        "info": {
            "description": "Test COCO dataset with keypoints",
            "version": "1.0",
        },
        "images": [
            {
                "id": 1,
                "file_name": "test_keypoints.jpg",
                "width": 640,
                "height": 480,
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [100, 100, 200, 300],
                "keypoints": [
                    150,
                    150,
                    2,  # visible keypoint
                    200,
                    200,
                    2,  # visible keypoint
                    250,
                    250,
                    1,  # occluded keypoint
                    0,
                    0,
                    0,  # not labeled
                    300,
                    350,
                    2,  # visible keypoint
                ],
                "num_keypoints": 4,
                "area": 60000,
                "iscrowd": 0,
            }
        ],
        "categories": [
            {
                "id": 1,
                "name": "person",
                "supercategory": "person",
                "keypoints": ["nose", "left_eye", "right_eye", "left_ear", "right_ear"],
                "skeleton": [[1, 2], [1, 3], [2, 4], [3, 5]],
            }
        ],
    }


@pytest.fixture
def temp_dir(tmp_path) -> Path:
    """Create a temporary directory for test files.

    Args:
        tmp_path: Pytest's tmp_path fixture.

    Returns:
        Path to the temporary directory.
    """
    return tmp_path


@pytest.fixture
def sample_json_file(temp_dir, sample_coco_data) -> Path:
    """Create a temporary JSON file with sample COCO data.

    Args:
        temp_dir: Temporary directory path.
        sample_coco_data: Sample COCO data dictionary.

    Returns:
        Path to the created JSON file.
    """
    json_path = temp_dir / "test_coco.json"
    with open(json_path, "w") as f:
        json.dump(sample_coco_data, f)
    return json_path


@pytest.fixture
def sample_images(temp_dir) -> Path:
    """Create sample image files for testing.

    Args:
        temp_dir: Temporary directory path.

    Returns:
        Path to the directory containing test images.
    """
    image_dir = temp_dir / "images"
    image_dir.mkdir(exist_ok=True)

    # Create test images with different sizes
    test_images = [
        ("test_image_1.jpg", (640, 480)),
        ("test_image_2.jpg", (800, 600)),
        ("test_image_3.jpg", (1024, 768)),
        ("test_keypoints.jpg", (640, 480)),
    ]

    for filename, size in test_images:
        # Create a simple RGB image with random colors
        img_array = np.random.randint(0, 255, (*size[::-1], 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(image_dir / filename)

    return image_dir


@pytest.fixture
def multiple_coco_files(temp_dir, sample_coco_data) -> list[Path]:
    """Create multiple COCO JSON files for merge testing.

    Args:
        temp_dir: Temporary directory path.
        sample_coco_data: Base sample COCO data.

    Returns:
        List of paths to created COCO JSON files.
    """
    files = []

    # Create first file with original data
    file1 = temp_dir / "coco1.json"
    with open(file1, "w") as f:
        json.dump(sample_coco_data, f)
    files.append(file1)

    # Create second file with modified IDs
    data2 = json.loads(json.dumps(sample_coco_data))  # Deep copy
    # Offset IDs to avoid conflicts
    for img in data2["images"]:
        img["id"] += 100
        img["file_name"] = f"set2_{img['file_name']}"
    for ann in data2["annotations"]:
        ann["id"] += 100
        ann["image_id"] += 100

    file2 = temp_dir / "coco2.json"
    with open(file2, "w") as f:
        json.dump(data2, f)
    files.append(file2)

    return files


@pytest.fixture
def invalid_coco_data() -> Dict[str, Any]:
    """Create invalid COCO data for error testing.

    Returns:
        Dictionary with invalid COCO structure.
    """
    return {
        "images": [
            {
                "id": 1,
                # Missing required fields like file_name, width, height
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 999,  # Non-existent image
                # Missing required fields
            }
        ],
        # Missing categories
    }

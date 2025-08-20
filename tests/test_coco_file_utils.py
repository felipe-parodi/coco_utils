"""Tests for coco_file_utils module."""

import json
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from coco_utils.coco_file_utils import (
    keep_first_n_images,
    merge_coco_files,
    split_coco_dataset,
)
from coco_utils.exceptions import InvalidCOCOFormatError


class TestKeepFirstNImages:
    """Test keep_first_n_images function."""

    def test_keep_first_n_images_basic(
        self, sample_coco_data: Dict[str, Any], tmp_path: Path
    ) -> None:
        """Test keeping first N images with basic functionality."""
        # Create input file
        input_file = tmp_path / "input.json"
        with open(input_file, "w") as f:
            json.dump(sample_coco_data, f)

        # Create output file
        output_file = tmp_path / "output.json"

        # Keep first 2 images
        result = keep_first_n_images(
            input_file,
            output_file,
            num_images_to_keep=2,
            validate_before=False,
            validate_after=False,
        )

        # Verify results
        assert len(result["images"]) == 2
        assert result["images"][0]["id"] == 1
        assert result["images"][1]["id"] == 2

        # Verify only annotations for kept images are included
        kept_image_ids = {1, 2}
        for ann in result["annotations"]:
            assert ann["image_id"] in kept_image_ids

        # Verify output file was created
        assert output_file.exists()
        with open(output_file) as f:
            saved_data = json.load(f)
        assert saved_data == result

    def test_keep_first_n_images_with_validation(
        self, sample_coco_data: Dict[str, Any], tmp_path: Path
    ) -> None:
        """Test keeping first N images with validation enabled."""
        input_file = tmp_path / "input.json"
        with open(input_file, "w") as f:
            json.dump(sample_coco_data, f)

        output_file = tmp_path / "output.json"

        with patch("coco_utils.coco_file_utils.validate_annotations") as mock_validate:
            mock_validate.return_value = (True, sample_coco_data)

            keep_first_n_images(
                input_file,
                output_file,
                num_images_to_keep=1,
                validate_before=True,
                validate_after=True,
            )

            # Validation should be called twice (before and after)
            assert mock_validate.call_count == 2

    def test_keep_first_n_images_file_not_found(self, tmp_path: Path) -> None:
        """Test error when input file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Input JSON file not found"):
            keep_first_n_images(
                tmp_path / "nonexistent.json", tmp_path / "output.json", num_images_to_keep=1
            )

    def test_keep_first_n_images_invalid_num(
        self, sample_coco_data: Dict[str, Any], tmp_path: Path
    ) -> None:
        """Test error when num_images_to_keep is invalid."""
        input_file = tmp_path / "input.json"
        with open(input_file, "w") as f:
            json.dump(sample_coco_data, f)

        with pytest.raises(ValueError, match="must be positive"):
            keep_first_n_images(input_file, tmp_path / "output.json", num_images_to_keep=0)

        with pytest.raises(ValueError, match="must be positive"):
            keep_first_n_images(input_file, tmp_path / "output.json", num_images_to_keep=-1)

    def test_keep_first_n_images_more_than_available(
        self, sample_coco_data: Dict[str, Any], tmp_path: Path
    ) -> None:
        """Test keeping more images than available keeps all."""
        input_file = tmp_path / "input.json"
        with open(input_file, "w") as f:
            json.dump(sample_coco_data, f)

        output_file = tmp_path / "output.json"

        result = keep_first_n_images(
            input_file,
            output_file,
            num_images_to_keep=100,  # More than available
            validate_before=False,
            validate_after=False,
        )

        # Should keep all images
        assert len(result["images"]) == len(sample_coco_data["images"])

    def test_keep_first_n_images_empty_dataset(self, tmp_path: Path) -> None:
        """Test keeping images from empty dataset."""
        empty_data = {"images": [], "annotations": [], "categories": []}

        input_file = tmp_path / "input.json"
        with open(input_file, "w") as f:
            json.dump(empty_data, f)

        output_file = tmp_path / "output.json"

        result = keep_first_n_images(
            input_file,
            output_file,
            num_images_to_keep=5,
            validate_before=False,
            validate_after=False,
        )

        assert len(result["images"]) == 0
        assert len(result["annotations"]) == 0

    def test_keep_first_n_images_invalid_json(self, tmp_path: Path) -> None:
        """Test error with invalid JSON file."""
        input_file = tmp_path / "invalid.json"
        with open(input_file, "w") as f:
            f.write("not valid json{")

        with pytest.raises(InvalidCOCOFormatError, match="Invalid JSON"):
            keep_first_n_images(input_file, tmp_path / "output.json", num_images_to_keep=1)

    def test_keep_first_n_images_missing_keys(self, tmp_path: Path) -> None:
        """Test error when required keys are missing."""
        incomplete_data = {"images": []}  # Missing annotations and categories

        input_file = tmp_path / "input.json"
        with open(input_file, "w") as f:
            json.dump(incomplete_data, f)

        with pytest.raises(InvalidCOCOFormatError, match="missing required keys"):
            keep_first_n_images(
                input_file, tmp_path / "output.json", num_images_to_keep=1, validate_before=False
            )


class TestMergeCOCOFiles:
    """Test merge_coco_files function."""

    def test_merge_coco_files_basic(self, tmp_path: Path) -> None:
        """Test basic merging of two COCO files."""
        # Create two COCO files with different images
        coco1 = {
            "images": [{"id": 1, "file_name": "img1.jpg", "width": 100, "height": 100}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]}],
            "categories": [{"id": 1, "name": "cat"}],
        }

        coco2 = {
            "images": [{"id": 1, "file_name": "img2.jpg", "width": 200, "height": 200}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20]}],
            "categories": [{"id": 1, "name": "dog"}],
        }

        file1 = tmp_path / "coco1.json"
        file2 = tmp_path / "coco2.json"
        output_file = tmp_path / "merged.json"

        with open(file1, "w") as f:
            json.dump(coco1, f)
        with open(file2, "w") as f:
            json.dump(coco2, f)

        result = merge_coco_files([file1, file2], output_file, validate_after_merge=False)

        # Check merged result
        assert len(result["images"]) == 2
        assert len(result["annotations"]) == 2
        assert len(result["categories"]) == 2  # cat and dog

        # Check that IDs were reassigned
        image_ids = {img["id"] for img in result["images"]}
        assert len(image_ids) == 2  # All unique

        ann_ids = {ann["id"] for ann in result["annotations"]}
        assert len(ann_ids) == 2  # All unique

    def test_merge_coco_files_duplicate_categories(self, tmp_path: Path) -> None:
        """Test merging files with duplicate category names."""
        coco1 = {
            "images": [{"id": 1, "file_name": "img1.jpg"}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1}],
            "categories": [{"id": 1, "name": "cat", "supercategory": "animal"}],
        }

        coco2 = {
            "images": [{"id": 1, "file_name": "img2.jpg"}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 2}],
            "categories": [{"id": 2, "name": "cat", "supercategory": "animal"}],
        }

        file1 = tmp_path / "coco1.json"
        file2 = tmp_path / "coco2.json"
        output_file = tmp_path / "merged.json"

        with open(file1, "w") as f:
            json.dump(coco1, f)
        with open(file2, "w") as f:
            json.dump(coco2, f)

        result = merge_coco_files([file1, file2], output_file, validate_after_merge=False)

        # Should have only one "cat" category
        assert len(result["categories"]) == 1
        assert result["categories"][0]["name"] == "cat"

    def test_merge_coco_files_empty_list(self, tmp_path: Path) -> None:
        """Test error when no files provided."""
        with pytest.raises(ValueError, match="No COCO file paths provided"):
            merge_coco_files([], tmp_path / "output.json")

    def test_merge_coco_files_missing_file(self, tmp_path: Path) -> None:
        """Test handling of missing files."""
        existing_file = tmp_path / "exists.json"
        coco_data = {
            "images": [{"id": 1, "file_name": "img.jpg"}],
            "annotations": [],
            "categories": [],
        }
        with open(existing_file, "w") as f:
            json.dump(coco_data, f)

        # Try to merge with non-existent file
        result = merge_coco_files(
            [existing_file, tmp_path / "missing.json"],
            tmp_path / "output.json",
            validate_after_merge=False,
        )

        # Should still process the existing file
        assert len(result["images"]) == 1

    @patch("coco_utils.coco_file_utils._get_user_choice")
    def test_merge_coco_files_duplicate_filenames(
        self, mock_choice: MagicMock, tmp_path: Path
    ) -> None:
        """Test handling of duplicate filenames with user choice."""
        coco1 = {
            "images": [{"id": 1, "file_name": "same.jpg", "width": 100}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1}],
            "categories": [{"id": 1, "name": "cat"}],
        }

        coco2 = {
            "images": [{"id": 2, "file_name": "same.jpg", "width": 200}],
            "annotations": [{"id": 2, "image_id": 2, "category_id": 1}],
            "categories": [{"id": 1, "name": "cat"}],
        }

        file1 = tmp_path / "coco1.json"
        file2 = tmp_path / "coco2.json"
        output_file = tmp_path / "merged.json"

        with open(file1, "w") as f:
            json.dump(coco1, f)
        with open(file2, "w") as f:
            json.dump(coco2, f)

        # Mock user choosing first file for duplicates
        mock_choice.return_value = 0  # Choose first file

        result = merge_coco_files([file1, file2], output_file, validate_after_merge=False)

        # Should keep only one image with "same.jpg"
        assert len(result["images"]) == 1
        # Should keep the one from the first file (width=100)
        assert result["images"][0]["width"] == 100


class TestSplitCOCODataset:
    """Test split_coco_dataset function."""

    def test_split_coco_dataset_basic(
        self, sample_coco_data: Dict[str, Any], tmp_path: Path
    ) -> None:
        """Test basic dataset splitting."""
        input_file = tmp_path / "input.json"
        with open(input_file, "w") as f:
            json.dump(sample_coco_data, f)

        output_dir = tmp_path / "splits"
        images_dir = tmp_path / "images"
        images_dir.mkdir()

        train_path, val_path, test_path = split_coco_dataset(
            input_file,
            output_dir,
            images_dir,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            copy_images=False,
            seed=42,
            validate_before_split=False,
        )

        # Check that files were created
        assert train_path.exists()
        assert val_path.exists()
        assert test_path.exists()

        # Load and check splits
        with open(train_path) as f:
            train_data = json.load(f)
        with open(val_path) as f:
            val_data = json.load(f)
        with open(test_path) as f:
            test_data = json.load(f)

        # Check that all images are accounted for
        total_images = len(sample_coco_data["images"])
        split_total = len(train_data["images"]) + len(val_data["images"]) + len(test_data["images"])
        assert split_total == total_images

        # Check no overlap between splits
        train_ids = {img["id"] for img in train_data["images"]}
        val_ids = {img["id"] for img in val_data["images"]}
        test_ids = {img["id"] for img in test_data["images"]}

        assert len(train_ids & val_ids) == 0
        assert len(train_ids & test_ids) == 0
        assert len(val_ids & test_ids) == 0

    def test_split_coco_dataset_with_image_copy(
        self, sample_coco_data: Dict[str, Any], tmp_path: Path
    ) -> None:
        """Test dataset splitting with image copying."""
        input_file = tmp_path / "input.json"
        with open(input_file, "w") as f:
            json.dump(sample_coco_data, f)

        images_dir = tmp_path / "images"
        images_dir.mkdir()

        # Create dummy image files
        for img in sample_coco_data["images"]:
            img_path = images_dir / img["file_name"]
            img_path.touch()

        output_dir = tmp_path / "splits"

        split_coco_dataset(
            input_file,
            output_dir,
            images_dir,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            copy_images=True,
            seed=42,
            validate_before_split=False,
        )

        # Check that image directories were created
        assert (output_dir / "train").exists()
        assert (output_dir / "val").exists()
        assert (output_dir / "test").exists()

        # Check that some images were copied
        train_images = list((output_dir / "train").iterdir())
        val_images = list((output_dir / "val").iterdir())
        test_images = list((output_dir / "test").iterdir())

        total_copied = len(train_images) + len(val_images) + len(test_images)
        assert total_copied == len(sample_coco_data["images"])

    def test_split_coco_dataset_invalid_ratios(
        self, sample_coco_data: Dict[str, Any], tmp_path: Path
    ) -> None:
        """Test error with invalid split ratios."""
        input_file = tmp_path / "input.json"
        with open(input_file, "w") as f:
            json.dump(sample_coco_data, f)

        images_dir = tmp_path / "images"
        images_dir.mkdir()

        # Ratios don't sum to 1.0
        with pytest.raises(ValueError, match="Ratios must sum to 1.0"):
            split_coco_dataset(
                input_file,
                tmp_path / "splits",
                images_dir,
                train_ratio=0.5,
                val_ratio=0.3,
                test_ratio=0.3,
            )

        # Invalid ratio values
        with pytest.raises(ValueError, match="Ratios must be between 0 and 1"):
            split_coco_dataset(
                input_file,
                tmp_path / "splits",
                images_dir,
                train_ratio=1.5,
                val_ratio=0.0,
                test_ratio=-0.5,
            )

    def test_split_coco_dataset_missing_images_dir(
        self, sample_coco_data: Dict[str, Any], tmp_path: Path
    ) -> None:
        """Test error when images directory doesn't exist and copy_images=True."""
        input_file = tmp_path / "input.json"
        with open(input_file, "w") as f:
            json.dump(sample_coco_data, f)

        with pytest.raises(FileNotFoundError, match="Source images directory not found"):
            split_coco_dataset(
                input_file, tmp_path / "splits", tmp_path / "nonexistent_images", copy_images=True
            )

    def test_split_coco_dataset_reproducible_with_seed(
        self, sample_coco_data: Dict[str, Any], tmp_path: Path
    ) -> None:
        """Test that splitting with same seed produces same results."""
        input_file = tmp_path / "input.json"
        with open(input_file, "w") as f:
            json.dump(sample_coco_data, f)

        images_dir = tmp_path / "images"
        images_dir.mkdir()

        # First split
        output_dir1 = tmp_path / "splits1"
        train1, val1, test1 = split_coco_dataset(
            input_file,
            output_dir1,
            images_dir,
            seed=123,
            copy_images=False,
            validate_before_split=False,
        )

        # Second split with same seed
        output_dir2 = tmp_path / "splits2"
        train2, val2, test2 = split_coco_dataset(
            input_file,
            output_dir2,
            images_dir,
            seed=123,
            copy_images=False,
            validate_before_split=False,
        )

        # Load and compare
        with open(train1) as f:
            train_data1 = json.load(f)
        with open(train2) as f:
            train_data2 = json.load(f)

        # Should have same image IDs in train set
        ids1 = {img["id"] for img in train_data1["images"]}
        ids2 = {img["id"] for img in train_data2["images"]}
        assert ids1 == ids2

    def test_split_coco_dataset_empty_dataset(self, tmp_path: Path) -> None:
        """Test splitting empty dataset."""
        empty_data = {"images": [], "annotations": [], "categories": []}

        input_file = tmp_path / "input.json"
        with open(input_file, "w") as f:
            json.dump(empty_data, f)

        images_dir = tmp_path / "images"
        images_dir.mkdir()

        with pytest.raises(InvalidCOCOFormatError, match="No images found"):
            split_coco_dataset(
                input_file, tmp_path / "splits", images_dir, validate_before_split=False
            )

"""Tests for coco_viz_utils module."""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from PIL import Image

from coco_utils.coco_viz_utils import (
    find_image_info,
    find_annotations,
    visualize_bbox,
    visualize_keypoints,
)


class TestFindImageInfo:
    """Test cases for find_image_info function."""
    
    def test_find_existing_image(self, sample_coco_data):
        """Test finding an existing image by ID."""
        result = find_image_info(sample_coco_data, 1)
        assert result is not None
        assert result["id"] == 1
        assert result["file_name"] == "test_image_1.jpg"
        
    def test_find_non_existing_image(self, sample_coco_data):
        """Test finding a non-existing image ID."""
        result = find_image_info(sample_coco_data, 999)
        assert result is None
        
    def test_empty_images_list(self):
        """Test with empty images list."""
        data = {"images": []}
        result = find_image_info(data, 1)
        assert result is None
        
    def test_missing_images_key(self):
        """Test with missing 'images' key."""
        data = {}
        result = find_image_info(data, 1)
        assert result is None


class TestFindAnnotations:
    """Test cases for find_annotations function."""
    
    def test_find_existing_annotations(self, sample_coco_data):
        """Test finding annotations for an existing image."""
        result = find_annotations(sample_coco_data, 1)
        assert len(result) == 2
        assert all(ann["image_id"] == 1 for ann in result)
        
    def test_find_no_annotations(self, sample_coco_data):
        """Test finding annotations for image with no annotations."""
        result = find_annotations(sample_coco_data, 3)
        assert result == []
        
    def test_empty_annotations_list(self):
        """Test with empty annotations list."""
        data = {"annotations": []}
        result = find_annotations(data, 1)
        assert result == []
        
    def test_missing_annotations_key(self):
        """Test with missing 'annotations' key."""
        data = {}
        result = find_annotations(data, 1)
        assert result == []


class TestVisualizeBbox:
    """Test cases for visualize_bbox function."""
    
    @patch("coco_utils.coco_viz_utils.plt.show")
    def test_visualize_bbox_success(
        self, mock_show, sample_coco_data, sample_images
    ):
        """Test successful bbox visualization."""
        # Should complete without errors
        visualize_bbox(sample_coco_data, 1, str(sample_images))
        mock_show.assert_called_once()
        
    @patch("coco_utils.coco_viz_utils.plt.show")
    def test_visualize_bbox_no_annotations(
        self, mock_show, sample_coco_data, sample_images
    ):
        """Test visualization with no annotations."""
        visualize_bbox(sample_coco_data, 3, str(sample_images))
        mock_show.assert_called_once()
        
    def test_visualize_bbox_image_not_found(self, sample_coco_data, temp_dir):
        """Test visualization when image file doesn't exist."""
        from coco_utils.exceptions import ImageNotFoundError
        # Should raise ImageNotFoundError for missing image
        with pytest.raises(ImageNotFoundError):
            visualize_bbox(sample_coco_data, 1, str(temp_dir))
        
    def test_visualize_bbox_invalid_image_id(self, sample_coco_data, sample_images):
        """Test visualization with invalid image ID."""
        from coco_utils.exceptions import VisualizationError
        # Should raise VisualizationError for invalid image ID
        with pytest.raises(VisualizationError):
            visualize_bbox(sample_coco_data, 999, str(sample_images))
        
    @patch("coco_utils.coco_viz_utils.plt.show")
    def test_visualize_bbox_custom_colors(
        self, mock_show, sample_coco_data, sample_images
    ):
        """Test bbox visualization with custom colors."""
        visualize_bbox(
            sample_coco_data, 
            1, 
            str(sample_images),
            box_color="blue",
            box_width=5
        )
        mock_show.assert_called_once()


class TestVisualizeKeypoints:
    """Test cases for visualize_keypoints function."""
    
    @patch("coco_utils.coco_viz_utils.plt.show")
    def test_visualize_keypoints_success(
        self, mock_show, sample_coco_with_keypoints, sample_images
    ):
        """Test successful keypoint visualization."""
        visualize_keypoints(sample_coco_with_keypoints, 1, str(sample_images))
        mock_show.assert_called_once()
        
    @patch("coco_utils.coco_viz_utils.plt.show")
    def test_visualize_keypoints_no_keypoints(
        self, mock_show, sample_coco_data, sample_images
    ):
        """Test visualization when annotations have no keypoints."""
        visualize_keypoints(sample_coco_data, 1, str(sample_images))
        mock_show.assert_called_once()
        
    def test_visualize_keypoints_image_not_found(
        self, sample_coco_with_keypoints, temp_dir
    ):
        """Test keypoint visualization when image file doesn't exist."""
        from coco_utils.exceptions import ImageNotFoundError
        # Should raise ImageNotFoundError for missing image
        with pytest.raises(ImageNotFoundError):
            visualize_keypoints(sample_coco_with_keypoints, 1, str(temp_dir))
        
    @patch("coco_utils.coco_viz_utils.plt.show")
    def test_visualize_keypoints_with_skeleton(
        self, mock_show, sample_coco_with_keypoints, sample_images
    ):
        """Test keypoint visualization with skeleton connections."""
        # The sample data includes skeleton information
        visualize_keypoints(sample_coco_with_keypoints, 1, str(sample_images))
        mock_show.assert_called_once()
        
    @patch("coco_utils.coco_viz_utils.plt.show")
    def test_visualize_keypoints_custom_colors(
        self, mock_show, sample_coco_with_keypoints, sample_images
    ):
        """Test keypoint visualization with custom settings."""
        visualize_keypoints(
            sample_coco_with_keypoints,
            1,
            str(sample_images),
            keypoint_color="green",
            keypoint_radius=10,
            skeleton_color="red",
            skeleton_width=3,
        )
        mock_show.assert_called_once()
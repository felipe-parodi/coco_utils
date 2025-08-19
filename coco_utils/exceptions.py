"""Custom exceptions for the coco_utils package.

This module defines custom exception classes for better error handling
and more informative error messages throughout the package.
"""


class COCOUtilsError(Exception):
    """Base exception class for all coco_utils errors."""

    pass


class FileOperationError(COCOUtilsError):
    """Raised when file operations fail."""

    pass


class InvalidCOCOFormatError(COCOUtilsError):
    """Raised when COCO data doesn't conform to expected format."""

    pass


class AnnotationError(COCOUtilsError):
    """Raised when there are issues with annotations."""

    pass


class ValidationError(COCOUtilsError):
    """Raised when validation of COCO data fails."""

    pass


class VisualizationError(COCOUtilsError):
    """Raised when visualization operations fail."""

    pass


class ImageNotFoundError(COCOUtilsError):
    """Raised when an image file cannot be found."""

    pass


class MergeError(COCOUtilsError):
    """Raised when merging COCO files fails."""

    pass


class SplitError(COCOUtilsError):
    """Raised when splitting COCO dataset fails."""

    pass


class InvalidBoundingBoxError(COCOUtilsError):
    """Raised when a bounding box has invalid dimensions or parameters."""

    pass

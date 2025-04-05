"""
COCO Visualization Utilities.

This module provides functions to visualize COCO dataset annotations,
including bounding boxes and keypoints, on their corresponding images.
It depends on PIL (Pillow) and Matplotlib.
"""

import json
import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

def find_image_info(data: Dict, image_id: int) -> Optional[Dict]:
    """Finds the image dictionary for a given image_id.

    Args:
        data: The loaded COCO dataset dictionary.
        image_id: The ID of the image to find.

    Returns:
        The dictionary containing image info, or None if not found.
    """
    for img_info in data.get('images', []):
        if img_info['id'] == image_id:
            return img_info
    return None

def find_annotations(data: Dict, image_id: int) -> List[Dict]:
    """Finds all annotations for a given image_id.

    Args:
        data: The loaded COCO dataset dictionary.
        image_id: The ID of the image whose annotations are sought.

    Returns:
        A list of annotation dictionaries for the given image_id.
    """
    return [ann for ann in data.get('annotations', []) if ann['image_id'] == image_id]

def visualize_bbox(
    coco_data: Dict,
    image_id: int,
    image_dir: str,
    box_color: str = "red",
    box_width: int = 3,
) -> None:
    """Visualizes **all** bounding boxes found for a given image_id.

    Args:
        coco_data: Loaded COCO data as a Python dictionary.
        image_id: The ID of the image to visualize.
        image_dir: The directory containing the image files referenced in coco_data.
        box_color: Color for the bounding box outlines.
        box_width: Width of the bounding box outlines.
    """
    img_info = find_image_info(coco_data, image_id)
    if not img_info:
        print(f"Error: Image with ID {image_id} not found in coco_data['images'].")
        return

    annotations = find_annotations(coco_data, image_id)
    if not annotations:
        print(f"Warning: No annotations found for image ID {image_id}.")

    image_filename = img_info.get('file_name')
    if not image_filename:
        print(f"Error: 'file_name' missing for image ID {image_id}.")
        return
    image_path = os.path.join(image_dir, image_filename)

    try:
        img = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image file not found at '{image_path}'")
        return
    except Exception as e:
        print(f"Error loading image '{image_path}': {e}")
        return

    draw = ImageDraw.Draw(img)
    num_boxes_drawn = 0

    # Iterate through all annotations found for this image
    for annotation in annotations:
        ann_id = annotation.get('id', 'N/A')
        bbox = annotation.get('bbox') # COCO format: [x_min, y_min, width, height]

        if bbox and len(bbox) == 4:
            x_min, y_min, width, height = bbox
            # PIL draw.rectangle expects [x0, y0, x1, y1]
            x_max = x_min + width
            y_max = y_min + height
            draw.rectangle([x_min, y_min, x_max, y_max], outline=box_color, width=box_width)
            print(f"Drawing bbox {bbox} for annotation ID {ann_id} on image ID {image_id}.")
            num_boxes_drawn += 1
        else:
            # Only print warning if an annotation exists but lacks a valid bbox
            if 'bbox' in annotation:
                print(f"Warning: Bounding box invalid for annotation ID {ann_id} on image ID {image_id}.")
            # If 'bbox' key doesn't even exist, perhaps it's a keypoints-only annotation, no warning needed.


    if num_boxes_drawn == 0 and annotations:
         print(f"Warning: Found {len(annotations)} annotations for image ID {image_id}, but none had valid bounding boxes.")
    elif num_boxes_drawn == 0 and not annotations:
         print(f"Image ID {image_id} has no annotations.") # Message adjusted from earlier check


    # Display the image
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    title = f"Image: {image_filename} (ID: {image_id}) | {num_boxes_drawn} Bounding Box(es)"
    plt.title(title)
    plt.axis('off')
    plt.show()

def visualize_keypoints(
    coco_data: Dict,
    image_id: int,
    image_dir: str,
    point_colors: List[str] = ["blue", "red", "green"],
    point_radius: int = 5,
    skeleton_color: str = "white",
    line_width: int = 2,
) -> None:
    """Visualizes keypoints and skeleton for a given image_id in COCO format.

    Uses specified colors (default: blue, red, green) for keypoints of the
    first few detected individuals (up to len(point_colors)).
    Draws skeleton connections if defined in categories.

    Args:
        coco_data: Loaded COCO data as a Python dictionary.
        image_id: The ID of the image to visualize.
        image_dir: The directory containing the image files.
        point_colors: List of colors to cycle through for different individuals.
        point_radius: Radius of the drawn keypoints.
        skeleton_color: Color for the skeleton lines.
        line_width: Width of the skeleton lines.
    """
    img_info = find_image_info(coco_data, image_id)
    if not img_info:
        print(f"Error: Image with ID {image_id} not found.")
        return

    annotations = find_annotations(coco_data, image_id)
    if not annotations:
        print(f"Error: No annotations found for image ID {image_id}.")
        return

    # Filter annotations that actually have keypoints
    keypoint_annotations = [
        ann for ann in annotations
        if 'keypoints' in ann and ann.get('num_keypoints', 0) > 0
    ]
    if not keypoint_annotations:
        print(f"Warning: No annotations *with keypoints* found for image ID {image_id}.")
        # Optionally, still show the image? For now, we return.
        return

    image_filename = img_info.get('file_name')
    if not image_filename:
         print(f"Error: 'file_name' missing for image ID {image_id}.")
         return
    image_path = os.path.join(image_dir, image_filename)

    try:
        img = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image file not found at '{image_path}'")
        return
    except Exception as e:
        print(f"Error loading image '{image_path}': {e}")
        return

    draw = ImageDraw.Draw(img)

    # Attempt to find skeleton definition from categories
    skeleton = None
    category_id = keypoint_annotations[0].get('category_id') # Assume all keypoint annotations for an image share a category ID
    if category_id is not None:
        for cat in coco_data.get('categories', []):
            if cat['id'] == category_id:
                skeleton = cat.get('skeleton') # COCO skeleton: list of [idx1, idx2] (1-based)
                keypoint_names = cat.get('keypoints') # List of keypoint names
                print(f"Found category '{cat.get('name', 'N/A')}' (ID: {category_id}) with {len(keypoint_names)} keypoints and {len(skeleton) if skeleton else 0} skeleton connections.")
                break
        if not skeleton:
             print(f"Warning: Skeleton not defined in category ID {category_id}.")

    num_individuals_to_draw = min(len(keypoint_annotations), len(point_colors))
    print(f"Visualizing keypoints for {num_individuals_to_draw} individual(s) out of {len(keypoint_annotations)} found.")

    for i, ann in enumerate(keypoint_annotations[:num_individuals_to_draw]):
        keypoints = ann.get('keypoints') # Format: [x1, y1, v1, x2, y2, v2, ...]
        if not keypoints: continue

        color = point_colors[i % len(point_colors)]
        ann_id = ann.get('id', 'N/A')
        print(f"  Individual {i+1} (Ann ID: {ann_id}): Color {color}")

        kp_coords_vis = [] # Store (x, y, v) for drawing skeleton
        num_visible = 0
        # Draw individual keypoints first
        for k in range(0, len(keypoints), 3):
            x, y, v = keypoints[k], keypoints[k+1], keypoints[k+2]
            kp_coords_vis.append((x, y, v))
            # v=0: not labeled, v=1: labeled but not visible, v=2: labeled and visible
            if v > 0: # Draw if labeled, could differentiate based on visibility if needed
                 if v == 2: num_visible += 1
                 # Draw an ellipse (circle) for the keypoint
                 draw.ellipse(
                    (x - point_radius, y - point_radius, x + point_radius, y + point_radius),
                    fill=color,
                    outline=color # Use same color for outline unless distinction needed
                 )
        print(f"    Drawn {num_visible} visible keypoints (out of {len(kp_coords_vis)} total labeled points).")


        # Draw skeleton connections for this individual
        if skeleton:
            connections_drawn = 0
            for conn_idx, connection in enumerate(skeleton):
                # COCO uses 1-based indexing for skeleton points
                kp_idx1 = connection[0] - 1
                kp_idx2 = connection[1] - 1

                # Check if indices are valid for the current keypoint list length
                if 0 <= kp_idx1 < len(kp_coords_vis) and 0 <= kp_idx2 < len(kp_coords_vis):
                    x1, y1, v1 = kp_coords_vis[kp_idx1]
                    x2, y2, v2 = kp_coords_vis[kp_idx2]

                    # Draw connection only if *both* keypoints are at least labeled (v>0)
                    # Stronger condition: only if both are visible (v==2) might be desired too.
                    if v1 > 0 and v2 > 0:
                        draw.line([(x1, y1), (x2, y2)], fill=skeleton_color, width=line_width)
                        connections_drawn += 1
                else:
                    print(f"    Warning: Invalid skeleton connection indices {connection} for keypoint list length {len(kp_coords_vis)}.")
            print(f"    Drawn {connections_drawn} skeleton connections.")
        elif i == 0: # Only print skeleton warning once
             print("    Skipping skeleton drawing as 'skeleton' definition was not found in categories.")


    # Display the image
    plt.figure(figsize=(12, 12))
    plt.imshow(img)
    plt.title(f"Image: {image_filename} (ID: {image_id}) | Keypoints ({num_individuals_to_draw} individuals)")
    plt.axis('off')
    plt.show()
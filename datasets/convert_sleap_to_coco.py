
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np


def get_keypoint_mapping() -> Dict[str, int]:
    """Create mapping from SLEAP node names to correct keypoint order

    Returns:
        Dict[str, int]: mapping from SLEAP node names to keypoint indices
    """
    keypoint_order = [
        "HeadTop_Mid",
        # "RightEarTop_Join",
        # "RightEarTop_High",
        # "RightEar_Outer",
        # "RightEarBottom_Low",
        # "RightEarBottom_Join",
        # "RightEar_Tragus",
        "OutlineTop_Mid",
        "OutlineTop_Right",
        "OutlineRight_Brow",
        "OutlineRight_Indent",
        "OutlineRight_Cheek",
        "OutlineRight_Mouth",
        "OutlineChin_Mid",
        "OutlineLeft_Mouth",
        "OutlineLeft_Cheek",
        "OutlineLeft_Indent",
        "OutlineLeft_Brow",
        "OutlineTop_Left",
        # "LeftEarTop_Join",
        # "LeftEarTop_High",
        # "LeftEar_Outer",
        # "LeftEarBottom_Low",
        # "LeftEarBottom_Join",
        # "LeftEar_Tragus",
        "Eyes_MidPoint",
        "RightEye_Inner",
        "RightEye_Top",
        "RightEye_Outer",
        "RightEye_Bottom",
        "RightEye_Pupil",
        "LeftEye_Inner",
        "LeftEye_Top",
        "LeftEye_Outer",
        "LeftEye_Bottom",
        "LeftEye_Pupil",
        "RightBrow_Outer",
        "RightBrow_Top",
        "RightBrow_Inner",
        "Brow_MidPoint",
        "LeftBrow_Inner",
        "LeftBrow_Top",
        "LeftBrow_Outer",
        "RightNostrils_Top",
        "RightNostrils_Bottom",
        "LeftNostrils_Bottom",
        "LeftNostrils_Top",
        "NostrilsTop_Centre",
        "UpperLip_Centre",
        "LowerLip_Centre",
        "MidPoint_Nostrils_Mouth",
        "Nose_Bridge_Top",
        "Nose_Bridge_Lower",
        "Right_UpperLip_1",
        "Right_UpperLip_2",
        "Left_UpperLip_2",
        "Left_UpperLip_1",
        "Left_LowerLip_1",
        "Left_LowerLip_2",
        "Right_LowerLip_2",
        "Right_LowerLip_1",

    ]

    return {name: idx for idx, name in enumerate(keypoint_order)}


def convert_skeleton_indices(
    skeleton_edges: List[Dict[str, Any]],
    sleap_nodes: List[Dict[str, Any]],
    keypoint_mapping: Dict[str, int],
) -> List[List[int]]:
    """Convert skeleton edges from SLEAP node indices to new keypoint indices

    Args:
        skeleton_edges (List[Dict[str, Any]]): list of skeleton edges
        sleap_nodes (List[Dict[str, Any]]): list of sleap nodes
        keypoint_mapping (Dict[str, int]): mapping from SLEAP node names to keypoint indices

    Returns:
        List of [source, target]: index pairs of converted skeleton edges
    """
    # Create mapping from SLEAP node index to node name
    sleap_idx_to_name = {i: node["name"] for i, node in enumerate(sleap_nodes)}

    # Convert each edge
    converted_edges = []
    for edge in skeleton_edges:
        source_name = sleap_idx_to_name[int(edge["source"])]
        target_name = sleap_idx_to_name[int(edge["target"])]

        # Map to new indices
        if source_name in keypoint_mapping and target_name in keypoint_mapping:
            new_source = keypoint_mapping[source_name]
            new_target = keypoint_mapping[target_name]
            converted_edges.append([new_source, new_target])

    return converted_edges

def convert_sleap_to_coco(
    sleap_json_path: Union[str, Path],
    output_path: Union[str, Path],
    reference_coco_path: Union[str, Path],
) -> None:
    """Convert SLEAP JSON to COCO JSON format using a reference COCO file.

    Args:
        sleap_json_path (Union[str, Path]): path to SLEAP JSON file
        output_path (Union[str, Path]): path to output COCO JSON file
        reference_coco_path (Union[str, Path]): path to reference COCO JSON file

    Returns:
        None
    """
    # Validate inputs
    sleap_json_path = Path(sleap_json_path)
    output_path = Path(output_path)
    reference_coco_path = Path(reference_coco_path) # Add validation for reference path

    if not sleap_json_path.exists():
        raise FileNotFoundError(f"SLEAP JSON file not found: {sleap_json_path}")
    if not reference_coco_path.exists():
        raise FileNotFoundError(f"Reference COCO JSON file not found: {reference_coco_path}")

    try:
        with open(sleap_json_path, "r") as f:
            sleap_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding SLEAP JSON file: {e}")

    try:
        with open(reference_coco_path, "r") as f:
            ref_coco_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding Reference COCO JSON file: {e}")


    # --- Keypoint and Category Handling using Reference COCO ---

    # Extract reference category (assuming the first category is the relevant one)
    if not ref_coco_data.get("categories"):
        raise ValueError("Reference COCO file missing 'categories' field.")
    ref_category = ref_coco_data["categories"][0] # TODO: Add check if categories list is empty
    ref_keypoints = ref_category.get("keypoints")
    if not ref_keypoints:
        raise ValueError("Reference COCO category missing 'keypoints' field.")

    # Extract SLEAP node names
    sleap_nodes = sleap_data.get("nodes")
    if not sleap_nodes:
        raise ValueError("SLEAP data missing 'nodes' field.")
    sleap_node_names = [node["name"] for node in sleap_nodes]

    # Validate keypoint counts
    if len(ref_keypoints) != len(sleap_node_names):
        raise ValueError(
            f"Mismatch in keypoint counts: Reference COCO has {len(ref_keypoints)}, "
            f"SLEAP file has {len(sleap_node_names)} unique nodes."
        )

    # Create keypoint mapping from reference COCO keypoint order
    keypoint_mapping = {name: idx for idx, name in enumerate(ref_keypoints)}

    # Comment out the old keypoint mapping function call
    # keypoint_mapping = get_keypoint_mapping()

    # Initialize COCO format structure using reference category
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [ref_category], # Use the category from the reference file
    }

    # Comment out the old hardcoded category structure
    # coco_data = {
    #     "images": [],
    #     "annotations": [],
    #     "categories": [
    #         {
    #             "supercategory": "face",
    #             "id": 1,
    #             "name": "macface",
    #             "keypoints": list(keypoint_mapping.keys()),
    #             "skeleton": [],
    #         }
    #     ],
    # }

    # Comment out skeleton conversion - use skeleton from reference COCO directly
    # if sleap_data["skeletons"]:
    #     skeleton = sleap_data["skeletons"][0]
    #     converted_skeleton = convert_skeleton_indices(
    #         skeleton["links"], sleap_data["nodes"], keypoint_mapping
    #     )
    #     coco_data["categories"][0]["skeleton"] = converted_skeleton

    # Generate unique IDs
    id_pool = np.arange(0, 10_000_000)
    np.random.shuffle(id_pool)
    frame_id_counter = 0
    ann_id_counter = 0

    # Process images and annotations
    image_id_map = {}  # Map to track video+frame to unique image ID

    for label in sleap_data["labels"]:
        video_idx = int(label["video"])
        frame_idx = label["frame_idx"]
        video_data = sleap_data["videos"][video_idx]

        # Generate unique image ID
        if (video_idx, frame_idx) not in image_id_map:
            frame_id_uniq = int(id_pool[frame_id_counter])
            image_id_map[(video_idx, frame_idx)] = frame_id_uniq
            frame_id_counter += 1

            # Extract filename from video path
            video_path = video_data["backend"]["filename"]
            filename = os.path.basename(video_path)
            # Skip files ending in .mp4
            if filename.lower().endswith(".mp4"):
                continue
            # Add image info
            try:
                height = video_data["backend"]["height_"]
                width = video_data["backend"]["width_"]
            except KeyError:
                height = 2160
                width = 3840
            coco_data["images"].append(
                {
                    "id": frame_id_uniq,
                    "file_name": filename,
                    "height": height,
                    "width": width,
                }
            )

        image_id = image_id_map[(video_idx, frame_idx)]

        # Process instances
        for instance in label["_instances"]:
            # Initialize keypoints array with zeros
            keypoints = [0] * (len(keypoint_mapping) * 3)
            num_keypoints = 0
            points_data = instance["_points"]

            # Process keypoints and calculate bbox
            valid_points = []
            for node_idx, point in points_data.items():
                node_name = sleap_data["nodes"][int(node_idx)]["name"]
                if node_name in keypoint_mapping:
                    idx = keypoint_mapping[node_name] * 3
                    # Always store coordinates
                    keypoints[idx] = float(point["x"])
                    keypoints[idx + 1] = float(point["y"])

                    # Set visibility flag - only 2 if visible, 0 otherwise
                    if point["visible"]:
                        keypoints[idx + 2] = 2  # visible
                        num_keypoints += 1
                        valid_points.append((float(point["x"]), float(point["y"])))
                    else:
                        keypoints[idx + 2] = 0  # not visible (even if labeled)

            if valid_points:
                x_coords, y_coords = zip(*valid_points)
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                width = x_max - x_min
                height = y_max - y_min
                bbox = [float(x_min), float(y_min), float(width), float(height)]
                area = round(width * height, 2)
                center = [float(x_min + width / 2), float(y_min + height / 2)]
                scale = [float(width / 200), float(height / 200)]

                # Create annotation with unique ID
                annotation = {
                    "id": int(ann_id_counter),
                    "image_id": image_id,
                    "category_id": 1,
                    "keypoints": keypoints,
                    "num_keypoints": num_keypoints,
                    "area": area,
                    "bbox": bbox,
                    "center": center,
                    "scale": scale,
                    "iscrowd": 0,
                }

                coco_data["annotations"].append(annotation)
                ann_id_counter += 1

    # Save COCO JSON
    with open(output_path, "w") as f:
        json.dump(coco_data, f, indent=2)


def main():
    """Command line interface for convert_sleap_to_coco.py"""
    parser = argparse.ArgumentParser(description="Convert SLEAP JSON to COCO JSON")
    parser.add_argument("sleap_json_path", type=str, help="Path to SLEAP JSON file")
    parser.add_argument("output_path", type=str, help="Path to output COCO JSON file")
    parser.add_argument("reference_coco_path", type=str, help="Path to reference COCO JSON file")
    args = parser.parse_args()

    convert_sleap_to_coco(args.sleap_json_path, args.output_path, args.reference_coco_path)
    print(f"Conversion complete. Output saved to {args.output_path}")


if __name__ == "__main__":
    main()

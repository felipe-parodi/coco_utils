# refine_boxes.py

#TODO:
# - add docstring
# - why does the gui look low-res?


import json
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict, List, Set, Tuple, Union
import copy
import argparse

import cv2
import numpy as np
from PIL import Image, ImageTk

# Import the shrinking utility
from coco_utils.coco_labels_utils import calculate_shrunk_bboxes

# Define the fixed shrink percentage
SHRINK_PERCENTAGE = 5.0

# --- COCO Data Loading ---


def load_coco_data(
    coco_json_path: str, img_dir: str
) -> Tuple[Dict[int, Dict], Dict[int, List[Dict]], List[Dict], int]:
    """Loads COCO data and organizes it for the GUI."""
    print(f"Loading COCO data from: {coco_json_path}")
    # Strip potential quotes from the path (common issue with 'Copy as path' on Windows)
    cleaned_coco_path = coco_json_path.strip('\"\'')
    print(f"Attempting to open cleaned path: {cleaned_coco_path}")
    try:
        with open(cleaned_coco_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        messagebox.showerror("Error", f"COCO JSON file not found: {cleaned_coco_path}")
        return None, None, None, -1
    except json.JSONDecodeError:
        messagebox.showerror(
            "Error", f"Error decoding COCO JSON file: {cleaned_coco_path}"
        )
        return None, None, None, -1
    except OSError as e:
        messagebox.showerror(
            "Error", f"Error opening COCO JSON file: {cleaned_coco_path}\n{e}"
        )
        return None, None, None, -1

    # Strip potential quotes from the image directory path as well
    cleaned_img_dir = img_dir.strip('\"\'')
    print(f"Using cleaned image directory: {cleaned_img_dir}")

    image_map = {}
    max_image_id = 0
    print("Mapping images...")
    for img in data.get("images", []):
        img_id = img["id"]
        img_path = os.path.join(cleaned_img_dir, img["file_name"]) # Use cleaned path
        if not os.path.exists(img_path):
            print(f"Warning: Image file not found, skipping: {img_path}") # Show the path checked
            continue
        image_map[img_id] = {
            "path": img_path,
            "width": img["width"],
            "height": img["height"],
            "file_name": img["file_name"],  # Store filename for saving
        }
        max_image_id = max(max_image_id, img_id)

    annotation_map = {
        img_id: [] for img_id in image_map
    }  # Initialize for all valid images
    max_ann_id = 0
    print("Mapping annotations...")
    for ann in data.get("annotations", []):
        img_id = ann["image_id"]
        if img_id in image_map:  # Only add annotations for images that exist
            annotation_map[img_id].append(
                ann.copy()
            )  # Use copy to avoid modifying original data directly
            max_ann_id = max(max_ann_id, ann.get("id", 0))

    categories = data.get("categories", [])
    print(f"Loaded {len(image_map)} images and annotations for them.")
    if not image_map:
        messagebox.showerror(
            "Error",
            f"No valid image paths found based on COCO file and image directory.",
        )
        return None, None, None, -1

    # Calculate next available ID for new annotations
    next_new_ann_id = max_ann_id + 1

    return image_map, annotation_map, categories, next_new_ann_id


# --- Main Refinement GUI ---


class COCORefinementGUI:
    """Tkinter GUI for refining COCO annotations."""

    def __init__(
        self,
        image_map: Dict[int, Dict],
        annotation_map: Dict[int, List[Dict]],
        shrunk_annotation_map: Dict[int, List[Dict]],
        categories: List[Dict],
        initial_view_mode: str,
        coco_json_path: str,
        next_start_ann_id: int,
    ):
        self.image_map = image_map
        self.categories = categories
        self.coco_json_path = coco_json_path
        self.output_path = coco_json_path.replace(".json", "_refined.json")

        # Store original and shrunk annotations separately
        # We need deep copies here to avoid modifying the originals inadvertently
        # during the GUI operations before a decision is made.
        self.original_annotations = copy.deepcopy(annotation_map)
        self.shrunk_annotations = copy.deepcopy(shrunk_annotation_map)

        # This holds the currently active annotations for the image being viewed/edited.
        # It starts as a copy of the original, and will be updated when the user
        # chooses original/shrunk or makes manual edits.
        self.modified_annotations = copy.deepcopy(self.original_annotations)

        self.image_ids = sorted(list(image_map.keys()))  # Process in a consistent order

        if not self.image_ids:
            messagebox.showerror("Error", "No valid images loaded. Exiting.")
            return  # Or raise an exception

        self.current_idx = 0
        self.view_mode = initial_view_mode
        self.next_start_ann_id = next_start_ann_id

        # Determine initial decision state based on whether shrinking was done
        # If shrunk_annotation_map is empty or None, assume shrinking wasn't requested/successful
        start_in_comparison_mode = bool(shrunk_annotation_map) # True if shrunk_anno_data is not empty/None

        # Initialize main window
        self.root = tk.Tk()
        self.root.title("COCO Annotation Refinement")

        # Mouse interaction state
        self.dragging = False
        self.drag_start = None
        self.selected_ann_id = None
        self.selected_element_type = None  # 'box', 'handle', 'keypoint'
        self.drag_type = None  # 'move', 'resize', 'create', None
        self.selected_corner = None  # For bbox resizing
        self.new_box_start = None
        self.creating_new_box = False
        self.handle_size = 8  # Size of corner handles

        self.image_decision_state: Dict[int, str] = { # Track user choice per image
            img_id: 'undecided' if start_in_comparison_mode else 'original'
            for img_id in self.image_ids
        }
        self.deleted_image_ids = set()
        self.modifications_made = False  # Track if any changes occurred

        # Extract skeleton links if categories exist
        self.skeleton_links = []
        if self.categories:
            # Assuming the first category defines the skeleton
            skeleton = self.categories[0].get("skeleton", [])
            keypoints_map = {
                name: i + 1
                for i, name in enumerate(self.categories[0].get("keypoints", []))
            }  # Assumes keypoints are ordered 1-based for skeleton
            # Convert skeleton pairs (which might be 1-based) to 0-based indices
            # Or adjust logic if skeleton is already 0-based or name-based
            # For now, assuming skeleton uses 1-based indexing as common in COCO
            self.skeleton_links = [[s[0] - 1, s[1] - 1] for s in skeleton]

        self._setup_gui()
        self._load_current_image()  # Load the first valid image

    def _setup_gui(self):
        """Setup GUI layout and controls"""
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Top control frame
        top_ctrl_frame = ttk.Frame(main_frame)
        top_ctrl_frame.pack(fill=tk.X, pady=5)
        ttk.Button(
            top_ctrl_frame, text="Toggle View (t)", command=self._toggle_view_mode
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            top_ctrl_frame, text="New Box (n)", command=self._toggle_new_box_mode
        ).pack(side=tk.LEFT, padx=5)
        self.view_mode_label = tk.StringVar(
            value=f"View: {self.view_mode.capitalize()}"
        )
        ttk.Label(top_ctrl_frame, textvariable=self.view_mode_label).pack(
            side=tk.LEFT, padx=10
        )
        self.new_box_label = tk.StringVar(value="")
        ttk.Label(
            top_ctrl_frame, textvariable=self.new_box_label, foreground="blue"
        ).pack(side=tk.LEFT, padx=10)

        # Add Image Counter/Jumper Entry in top-left
        self.image_jumper_var = tk.StringVar()
        self.image_jumper_entry = ttk.Entry(top_ctrl_frame, textvariable=self.image_jumper_var, width=15)
        self.image_jumper_entry.pack(side=tk.LEFT, padx=10)
        self.image_jumper_entry.bind("<Return>", self._jump_to_image)

        # Canvas for image display
        self.canvas = tk.Canvas(
            main_frame, bg="gray", width=800, height=600
        )  # Initial size
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Bottom control frame
        bottom_ctrl_frame = ttk.Frame(main_frame)
        bottom_ctrl_frame.pack(fill=tk.X, pady=5)
        ttk.Button(
            bottom_ctrl_frame, text="Previous (<-)", command=self._prev_image
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(bottom_ctrl_frame, text="Next (->)", command=self._next_image).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(
            bottom_ctrl_frame,
            text="Delete Image (Del)",
            command=self._delete_current_image,
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            bottom_ctrl_frame, text="Complete && Save (Esc)", command=self._complete
        ).pack(side=tk.RIGHT, padx=5)

        # Bind mouse events
        self.canvas.bind("<Button-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
        self.canvas.bind("<Button-3>", self._on_right_click)

        # Bind keyboard shortcuts
        self.root.bind("<Left>", lambda e: self._prev_image())
        self.root.bind("<Right>", lambda e: self._next_image())
        self.root.bind("<Delete>", lambda e: self._delete_current_image())
        self.root.bind("<Escape>", lambda e: self._complete())
        self.root.bind("<t>", lambda e: self._toggle_view_mode())
        self.root.bind("<n>", lambda e: self._toggle_new_box_mode())
        # Add bindings for A and D keys
        self.root.bind("<KeyPress-a>", self._accept_original)
        self.root.bind("<KeyPress-d>", self._accept_shrunk)
        # Add binding for temporary save
        self.root.bind("<Control-s>", self._save_temp_progress)

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close_window)

    def _find_valid_index(self, direction=1):
        """Find the next/previous valid (not deleted) image index."""
        if not self.image_ids:
            return -1  # No images left

        new_idx = self.current_idx
        checked_all = False
        while not checked_all:
            new_idx = (new_idx + direction) % len(self.image_ids)
            if self.image_ids[new_idx] not in self.deleted_image_ids:
                return new_idx
            if new_idx == self.current_idx:  # Cycled through all
                checked_all = True

        # If all images are deleted
        messagebox.showinfo("Info", "All images have been marked for deletion.")
        return -1

    def _load_current_image(self):
        """Load and display current image with annotations"""
        if not self.image_ids:
            self.canvas.delete("all")
            return

        current_img_id = self.image_ids[self.current_idx]
        img_info = self.image_map[current_img_id]
        img_path = img_info["path"]

        try:
            self.current_cv_image = cv2.imread(img_path)
            if self.current_cv_image is None:
                raise IOError(f"Failed to load image: {img_path}")
            img_rgb = cv2.cvtColor(self.current_cv_image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            messagebox.showerror(
                "Error Loading Image", f"Error loading {img_path}:\n{e}"
            )
            # Mark as deleted and move on? Or just skip? Let's mark as deleted.
            self.deleted_image_ids.add(current_img_id)
            next_idx = self._find_valid_index(1)
            if next_idx == -1 or next_idx == self.current_idx:
                self._complete()  # No more valid images
                return
            else:
                self.current_idx = next_idx
                self._load_current_image()  # Try loading the next one
                return

        h, w = img_rgb.shape[:2]

        # Fit to canvas while maintaining aspect ratio
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w <= 1 or canvas_h <= 1:  # Canvas not yet rendered
            canvas_w, canvas_h = 800, 600  # Use default size

        self.scale = min(canvas_w / w, canvas_h / h)
        self.display_w = int(w * self.scale)
        self.display_h = int(h * self.scale)

        # Calculate offsets for centering
        self.offset_x = (canvas_w - self.display_w) // 2
        self.offset_y = (canvas_h - self.display_h) // 2

        # Convert to PhotoImage
        image_pil = Image.fromarray(img_rgb)
        image_pil = image_pil.resize(
            (self.display_w, self.display_h), Image.Resampling.LANCZOS
        )
        self.photo = ImageTk.PhotoImage(image_pil)

        # Update canvas
        self.canvas.delete("all")  # Clear previous drawings
        self.canvas.config(
            width=canvas_w, height=canvas_h
        )  # Ensure canvas size is updated
        self.canvas.create_image(
            self.offset_x, self.offset_y, anchor=tk.NW, image=self.photo
        )

        # --- Display filename in top-right ---
        filename = img_info.get("file_name", "N/A")
        self.canvas.create_text(
            canvas_w - 10, 10, # Position near top-right corner
            text=filename,
            anchor=tk.NE, # Anchor to North-East corner
            fill="yellow",
            font=("Arial", 10),
            tags="filename_text"
        )

        # --- Update image counter entry widget ---
        total_images = len(self.image_ids)
        counter_text = f"{self.current_idx + 1} / {total_images}"
        self.image_jumper_var.set(counter_text)

        self._draw_annotations()

        self.root.update_idletasks()  # Ensure canvas resizes

    def _draw_annotations(self):
        """Draw annotations based on current image decision state and view mode."""
        self.canvas.delete(
            "box", "handle", "pose", "keypoint", "skeleton", "comparison_box" # Clear specific tags
        )

        current_img_id = self.image_ids[self.current_idx]
        state = self.image_decision_state.get(current_img_id, 'undecided')

        if state == 'undecided':
            # Comparison mode: Show original vs shrunk
            self.canvas.itemconfig("handle", state=tk.HIDDEN) # Hide handles in comparison
            self._draw_comparison_boxes(current_img_id)
            # Optionally, add status text
            self.canvas.create_text(
                self.canvas.winfo_width() / 2, 20,
                text="Compare: Original (Blue) vs Shrunk (Red). Press 'A' to keep Original, 'D' for Shrunk.",
                fill="yellow", font=("Arial", 12), tags="status_text"
            )
        else:
            # Active mode (original or shrunk chosen): Show editable annotations
            self.canvas.delete("status_text") # Clear comparison text
            self.canvas.itemconfig("handle", state=tk.NORMAL) # Show handles
            self._draw_active_annotations(current_img_id)

    def _draw_comparison_boxes(self, img_id: int):
        """Draws original (blue) and different shrunk (red) boxes for comparison."""
        original_anns = self.original_annotations.get(img_id, [])
        shrunk_anns = self.shrunk_annotations.get(img_id, [])

        # Create maps for easier lookup by annotation ID
        original_bboxes_by_id = {
            ann.get("id"): ann.get("bbox") for ann in original_anns if ann.get("id") is not None and ann.get("bbox")
        }
        shrunk_bboxes_by_id = {
            ann.get("id"): ann.get("bbox") for ann in shrunk_anns if ann.get("id") is not None and ann.get("bbox")
        }

        # Draw original boxes (Blue)
        for ann_id, bbox in original_bboxes_by_id.items():
            x, y, w, h = bbox
            x1 = x * self.scale + self.offset_x
            y1 = y * self.scale + self.offset_y
            x2 = (x + w) * self.scale + self.offset_x
            y2 = (y + h) * self.scale + self.offset_y
            self.canvas.create_rectangle(
                x1, y1, x2, y2, outline="blue", width=2, tags=("comparison_box", f"ann{ann_id}_orig")
            )

        # Draw shrunk boxes (Red) ONLY if they differ from the original
        for ann_id, shrunk_bbox in shrunk_bboxes_by_id.items():
            original_bbox = original_bboxes_by_id.get(ann_id)
            # Compare, draw red if shrunk exists and is different from original
            if shrunk_bbox and (original_bbox is None or shrunk_bbox != original_bbox):
                x, y, w, h = shrunk_bbox
                x1 = x * self.scale + self.offset_x
                y1 = y * self.scale + self.offset_y
                x2 = (x + w) * self.scale + self.offset_x
                y2 = (y + h) * self.scale + self.offset_y
                self.canvas.create_rectangle(
                    x1, y1, x2, y2, outline="red", width=2, tags=("comparison_box", f"ann{ann_id}_shrunk")
                )

    def _draw_active_annotations(self, img_id: int):
        """Draw annotations from self.modified_annotations with editing handles/poses."""
        annotations = self.modified_annotations.get(img_id, [])
        if not annotations:
            return

        self._draw_active_boxes(annotations)
        if self.view_mode == "pose":
            self._draw_active_poses(annotations)

    def _draw_active_boxes(self, annotations: List[Dict]):
        """Draw editable bounding boxes and handles from the active set."""
        for ann in annotations:
            ann_id = ann["id"]
            bbox = ann.get("bbox")  # COCO format: [x, y, width, height]
            if not bbox or len(bbox) != 4: # Added check for valid bbox
                continue

            x, y, w, h = bbox
            x1 = x * self.scale + self.offset_x
            y1 = y * self.scale + self.offset_y
            x2 = (x + w) * self.scale + self.offset_x
            y2 = (y + h) * self.scale + self.offset_y

            # Draw box
            box_tag = f"ann{ann_id}"
            # Ensure box is drawn with correct tags for later interaction
            self.canvas.create_rectangle(
                x1, y1, x2, y2, outline="red", width=2, tags=("box", box_tag, "editable")
            )

            # Draw handles
            handles_coords = [(x1, y1, "topleft"), (x2, y1, "topright"), (x1, y2, "bottomleft"), (x2, y2, "bottomright")]
            for hx, hy, corner_tag in handles_coords:
                self.canvas.create_rectangle(
                    hx - self.handle_size / 2,
                    hy - self.handle_size / 2,
                    hx + self.handle_size / 2,
                    hy + self.handle_size / 2,
                    fill="white",
                    outline="red",
                    # Add corner tag for easier identification during resize
                    tags=("handle", box_tag, corner_tag, "editable"),
                )

    def _draw_active_poses(self, annotations: List[Dict]):
        """Draw keypoints and skeletons from the active set."""
        kpt_radius = 4
        line_width = 2

        for ann in annotations:
            ann_id = ann["id"]
            kpts_flat = ann.get("keypoints")
            if not kpts_flat:
                continue

            # Reshape keypoints: [x1, y1, v1, x2, y2, v2, ...]
            kpts = np.array(kpts_flat).reshape(-1, 3)

            # Draw skeleton lines
            if self.skeleton_links:
                for start_idx, end_idx in self.skeleton_links:
                    # Check if indices are valid for this annotation's keypoints
                    if start_idx < len(kpts) and end_idx < len(kpts):
                        pt_start = kpts[start_idx]
                        pt_end = kpts[end_idx]

                        # Draw line only if both keypoints are visible (v=1 or v=2)
                        if pt_start[2] > 0 and pt_end[2] > 0:
                            x1 = pt_start[0] * self.scale + self.offset_x
                            y1 = pt_start[1] * self.scale + self.offset_y
                            x2 = pt_end[0] * self.scale + self.offset_x
                            y2 = pt_end[1] * self.scale + self.offset_y
                            self.canvas.create_line(
                                x1,
                                y1,
                                x2,
                                y2,
                                fill="cyan",
                                width=line_width,
                                tags=("skeleton", f"ann{ann_id}"),
                            )

            # Draw keypoints
            for i, (x, y, v) in enumerate(kpts):
                if v > 0:  # Draw only visible keypoints
                    center_x = x * self.scale + self.offset_x
                    center_y = y * self.scale + self.offset_y
                    self.canvas.create_oval(
                        center_x - kpt_radius,
                        center_y - kpt_radius,
                        center_x + kpt_radius,
                        center_y + kpt_radius,
                        fill="lime",
                        outline="black",
                        tags=("keypoint", f"ann{ann_id}", f"kpt{i}"),
                    )

    def _toggle_view_mode(self):
        """Toggle between bbox and pose view."""
        # Only toggle if not in comparison mode, or decide how it should interact
        current_img_id = self.image_ids[self.current_idx]
        state = self.image_decision_state.get(current_img_id, 'undecided')
        if state != 'undecided':
            self.view_mode = "pose" if self.view_mode == "bbox" else "bbox"
            self.view_mode_label.set(f"View: {self.view_mode.capitalize()}")
            self._draw_annotations() # Redraw using the active annotations
        else:
            messagebox.showinfo("Info", "Please choose Original ('A') or Shrunk ('D') before toggling view.")

    def _toggle_new_box_mode(self, event=None):
        """Toggle new box creation mode"""
        self.creating_new_box = not self.creating_new_box
        if self.creating_new_box:
            self.canvas.config(cursor="cross")
            self.new_box_label.set("Mode: Create New Box")
        else:
            self.canvas.config(cursor="")
            self.new_box_label.set("")

    # --- Placeholder Interaction Handlers ---
    def _on_mouse_down(self, event):
        # Set focus to the canvas to take it away from the Entry widget
        self.canvas.focus_set()

        # --- Check if editing is allowed ---
        current_img_id = self.image_ids[self.current_idx]
        state = self.image_decision_state.get(current_img_id, 'undecided')
        if state == 'undecided':
            print("In comparison mode. Choose 'A' or 'D' to enable editing.")
            return # Disable mouse interactions in comparison mode

        print(f"Mouse Down at ({event.x}, {event.y})")
        self.drag_start = (event.x, event.y)
        self.selected_ann_id = None
        self.drag_type = None
        self.selected_corner = None

        # 1. Check for Handle Clicks (Highest Priority)
        items = self.canvas.find_overlapping(
            event.x - 2, event.y - 2, event.x + 2, event.y + 2 # Slightly larger overlap check
        )
        for item in reversed(items): # Check topmost first
            tags = self.canvas.gettags(item)
            if "handle" in tags:
                ann_tag = next((t for t in tags if t.startswith("ann")), None)
                corner = next((t for t in tags if t in ["topleft", "topright", "bottomleft", "bottomright"]), None)
                if ann_tag and corner:
                    self.selected_ann_id = int(ann_tag[3:])
                    self.selected_corner = corner
                    self.drag_type = "resize"
                    print(f"Selected handle for ann {self.selected_ann_id}, corner: {self.selected_corner}")
                    return # Found handle, stop searching

        # 2. Check for Click Inside a Box (if no handle was clicked)
        current_img_id = self.image_ids[self.current_idx]
        annotations = self.modified_annotations.get(current_img_id, [])
        # Iterate through annotations data, not just canvas items
        for ann in reversed(annotations): # Check potentially overlapping boxes from top one down
            ann_id = ann["id"]
            bbox = ann.get("bbox")
            if not bbox:
                continue

            # Convert bbox to canvas coordinates
            x, y, w, h = bbox
            x1_canvas = x * self.scale + self.offset_x
            y1_canvas = y * self.scale + self.offset_y
            x2_canvas = (x + w) * self.scale + self.offset_x
            y2_canvas = (y + h) * self.scale + self.offset_y

            # Check if click is strictly inside the canvas box coordinates
            if x1_canvas < event.x < x2_canvas and y1_canvas < event.y < y2_canvas:
                self.selected_ann_id = ann_id
                self.drag_type = "move"
                # Calculate offset from top-left corner for smooth dragging
                self.drag_offset = (event.x - x1_canvas, event.y - y1_canvas)
                print(f"Selected box interior for ann {self.selected_ann_id} for moving")
                return # Found box interior, stop searching

        # 3. Check for New Box Creation Mode
        if self.creating_new_box:
            self.drag_type = "create"
            self.new_box_start = (event.x, event.y)
            print("Starting new box creation")
            return

        # 4. If nothing else, reset state (redundant but clear)
        self.selected_ann_id = None
        self.drag_type = None
        print("Click did not hit a handle or box interior.")

    def _find_annotation_by_id(self, ann_id_to_find: int) -> Dict | None:
        """Helper to find an annotation dict by its ID in the current image's list."""
        current_img_id = self.image_ids[self.current_idx]
        for ann in self.modified_annotations.get(current_img_id, []):
            if ann["id"] == ann_id_to_find:
                return ann
        return None

    def _on_mouse_drag(self, event):
        if self.drag_type == "create" and self.new_box_start:
            # Draw temporary rectangle
            if hasattr(self, "temp_rect_id"):
                self.canvas.delete(self.temp_rect_id)
            else:
                # Ensure it gets created even if mouse down missed the flag somehow
                self.temp_rect_id = None

            self.temp_rect_id = self.canvas.create_rectangle(
                self.new_box_start[0],
                self.new_box_start[1],
                event.x,
                event.y,
                outline="blue",
                width=2,
                tags="temp_box",
            )
        elif self.drag_type in ["move", "resize"] and self.selected_ann_id is not None:
            ann = self._find_annotation_by_id(self.selected_ann_id)
            if not ann or "bbox" not in ann:
                print(
                    f"Error: Could not find annotation {self.selected_ann_id} to drag."
                )
                self.drag_type = None  # Stop dragging if annotation disappears
                return

            # Convert current canvas coords to image coords
            current_x_img = (event.x - self.offset_x) / self.scale
            current_y_img = (event.y - self.offset_y) / self.scale

            # Get original bbox [x, y, w, h]
            x, y, w, h = ann["bbox"]
            # Get image dimensions for clamping
            img_info = self.image_map[ann["image_id"]]
            img_width = img_info["width"]
            img_height = img_info["height"]

            if self.drag_type == "move":
                # Calculate proposed new top-left corner in image space
                new_x1_canvas = event.x - self.drag_offset[0]
                new_y1_canvas = event.y - self.drag_offset[1]
                new_x_img = (new_x1_canvas - self.offset_x) / self.scale
                new_y_img = (new_y1_canvas - self.offset_y) / self.scale

                # Clamp coordinates to image boundaries
                # Ensure top-left doesn't go past 0,0
                new_x_img = max(0, new_x_img)
                new_y_img = max(0, new_y_img)
                # Ensure bottom-right doesn't go past image width/height
                if new_x_img + w > img_width:
                    new_x_img = img_width - w
                if new_y_img + h > img_height:
                    new_y_img = img_height - h
                # Re-check top-left after potential bottom-right clamp
                new_x_img = max(0, new_x_img)
                new_y_img = max(0, new_y_img)

                # Assign clamped coordinates (width and height remain unchanged)
                ann["bbox"][0] = new_x_img
                ann["bbox"][1] = new_y_img

            elif self.drag_type == "resize":
                # Original corners in image space
                orig_x1, orig_y1 = x, y
                orig_x2, orig_y2 = x + w, y + h

                # Proposed new corners based on drag
                new_x, new_y = current_x_img, current_y_img

                # Determine which corner(s) to update based on selected handle
                final_x1, final_y1, final_x2, final_y2 = orig_x1, orig_y1, orig_x2, orig_y2

                if "left" in self.selected_corner:
                    final_x1 = new_x
                if "right" in self.selected_corner:
                    final_x2 = new_x
                if "top" in self.selected_corner:
                    final_y1 = new_y
                if "bottom" in self.selected_corner:
                    final_y2 = new_y

                # Ensure coordinates stay within image bounds
                final_x1 = max(0, min(final_x1, img_width))
                final_y1 = max(0, min(final_y1, img_height))
                final_x2 = max(0, min(final_x2, img_width))
                final_y2 = max(0, min(final_y2, img_height))

                # Ensure x1 <= x2 and y1 <= y2, recalculate w, h
                new_x = min(final_x1, final_x2)
                new_y = min(final_y1, final_y2)
                new_w = abs(final_x1 - final_x2)
                new_h = abs(final_y1 - final_y2)

                # Swap corner if dimensions flipped (needed for subsequent drags)
                if final_x1 > final_x2:
                     corner_map_h = {"topleft": "topright", "topright": "topleft", "bottomleft": "bottomright", "bottomright": "bottomleft"}
                     self.selected_corner = corner_map_h.get(self.selected_corner)
                if final_y1 > final_y2:
                     corner_map_v = {"topleft": "bottomleft", "bottomleft": "topleft", "topright": "bottomright", "bottomright": "topright"}
                     self.selected_corner = corner_map_v.get(self.selected_corner)

                ann["bbox"] = [new_x, new_y, new_w, new_h]

            # Update derived fields like area
            ann["area"] = ann["bbox"][2] * ann["bbox"][3]

            self.modifications_made = True
            self._draw_annotations()  # Redraw with updated position/size

    def _on_mouse_up(self, event):
        if self.drag_type == "create" and self.new_box_start:
            # Finalize new box creation
            if hasattr(self, "temp_rect_id") and self.temp_rect_id is not None:
                self.canvas.delete(self.temp_rect_id)
                del self.temp_rect_id

            # Convert canvas coords back to image coords
            x_start_img = (self.new_box_start[0] - self.offset_x) / self.scale
            y_start_img = (self.new_box_start[1] - self.offset_y) / self.scale
            x_end_img = (event.x - self.offset_x) / self.scale
            y_end_img = (event.y - self.offset_y) / self.scale

            # Ensure positive width/height
            x = min(x_start_img, x_end_img)
            y = min(y_start_img, y_end_img)
            w = abs(x_start_img - x_end_img)
            h = abs(y_start_img - y_end_img)

            if w > 5 and h > 5:  # Only add if reasonably sized
                current_img_id = self.image_ids[self.current_idx]
                new_ann = {
                    "id": self.next_start_ann_id,
                    "image_id": current_img_id,
                    "category_id": self.categories[0]["id"]
                    if self.categories
                    else 1,  # Default category
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                    # Add empty keypoints if needed? For now, focus on bbox
                    "keypoints": [0] * len(self.categories[0].get("keypoints", [])) * 3
                    if self.categories
                    else [],
                    "num_keypoints": 0,
                }
                self.modified_annotations.setdefault(current_img_id, []).append(new_ann)
                self.next_start_ann_id += 1
                self.modifications_made = True
                self._draw_annotations()  # Redraw with the new box

            # Exit creation mode automatically after drawing one box
            self._toggle_new_box_mode()

        # Reset drag state regardless of action
        print(f"Mouse Up at ({event.x}, {event.y}), drag_type was: {self.drag_type}")
        self.dragging = False
        self.drag_start = None
        self.selected_ann_id = None
        self.selected_element_type = None
        self.drag_type = None
        self.selected_corner = None
        self.new_box_start = None
        # Ensure temp box ID is cleared if it exists
        if hasattr(self, "temp_rect_id") and self.temp_rect_id is not None:
            self.canvas.delete(self.temp_rect_id)
            del self.temp_rect_id

    def _on_right_click(self, event):
        # --- Check if editing is allowed ---
        current_img_id = self.image_ids[self.current_idx]
        state = self.image_decision_state.get(current_img_id, 'undecided')
        if state == 'undecided':
            print("In comparison mode. Choose 'A' or 'D' to enable deletion.")
            return # Disable right-click delete in comparison mode

        print(f"Right Click at ({event.x}, {event.y})")

        items = self.canvas.find_overlapping(
            event.x - 1, event.y - 1, event.x + 1, event.y + 1
        )
        if not items:
            return

        deleted_something = False
        for item in reversed(items):
            tags = self.canvas.gettags(item)
            if "box" in tags or "handle" in tags:  # Delete if click is on box or handle
                ann_tag = next((t for t in tags if t.startswith("ann")), None)
                if ann_tag:
                    ann_id_to_delete = int(ann_tag[3:])

                    # Find and remove the annotation
                    current_img_id = self.image_ids[self.current_idx]
                    annotations = self.modified_annotations.get(current_img_id, [])
                    initial_len = len(annotations)
                    self.modified_annotations[current_img_id] = [
                        ann for ann in annotations if ann["id"] != ann_id_to_delete
                    ]

                    if len(self.modified_annotations[current_img_id]) < initial_len:
                        print(f"Deleted annotation {ann_id_to_delete}")
                        self.modifications_made = True
                        deleted_something = True
                        break  # Stop after deleting one annotation

        if deleted_something:
            self._draw_annotations()  # Redraw the canvas

    # --- Navigation and Control ---
    def _jump_to_image(self, event=None):
        """Attempts to jump to the image number entered in the jumper Entry."""
        try:
            entry_text = self.image_jumper_var.get()
            target_num_str = entry_text.split('/')[0].strip()
            target_num = int(target_num_str)
            total_images = len(self.image_ids)

            if 1 <= target_num <= total_images:
                target_idx = target_num - 1
                if target_idx != self.current_idx:
                    # Check for unsaved changes before jumping?
                    # For now, let's assume jump implies potential context switch
                    # self._save_changes_if_needed() # Optional: prompt/save like next/prev
                    print(f"Jumping to image {target_num} (index {target_idx})...")
                    self.current_idx = target_idx
                    self._load_current_image()
                else:
                    # If already on the target image, just reset the text format
                    self._update_jumper_text()
            else:
                messagebox.showwarning("Invalid Image Number", f"Please enter a number between 1 and {total_images}.")
                self._update_jumper_text() # Reset text to current index
        except ValueError:
            messagebox.showwarning("Invalid Input", "Please enter the image number (e.g., '100').")
            self._update_jumper_text() # Reset text to current index
        except Exception as e:
            messagebox.showerror("Error Jumping", f"An unexpected error occurred: {e}")
            self._update_jumper_text()

    def _update_jumper_text(self):
        """Helper to reset the jumper text to the current image index."""
        total_images = len(self.image_ids)
        counter_text = f"{self.current_idx + 1} / {total_images}"
        self.image_jumper_var.set(counter_text)

    def _prev_image(self):
        if self.image_ids:
            # Save changes before moving?
            # self._save_changes_if_needed() # Optional
            next_idx = self._find_valid_index(-1)
            if next_idx != -1 and next_idx != self.current_idx:
                self.current_idx = next_idx
                self._load_current_image()

    def _next_image(self):
        if self.image_ids:
            next_idx = self._find_valid_index(1)
            if next_idx != -1 and next_idx != self.current_idx:
                self.current_idx = next_idx
                self._load_current_image()

    def _delete_current_image(self):
        if not self.image_ids:
            return
        current_img_id = self.image_ids[self.current_idx]
        if messagebox.askyesno(
            "Confirm Deletion",
            f"Mark image {current_img_id} ({self.image_map[current_img_id]['file_name']}) and all its annotations for deletion?",
        ):
            self.deleted_image_ids.add(current_img_id)
            self.modifications_made = True
            # Move to the next image
            next_idx = self._find_valid_index(1)
            if next_idx == -1:  # No more valid images
                messagebox.showinfo("Info", "Last image marked for deletion. Saving...")
                self._complete()
            elif (
                next_idx == self.current_idx
            ):  # Only one image left, which was just deleted
                messagebox.showinfo("Info", "All images marked for deletion. Saving...")
                self._complete()
            else:
                self.current_idx = next_idx
                self._load_current_image()

    def _on_close_window(self):
        """Handle clicking the window's close button."""
        if self.modifications_made:
            # Use askyesnocancel to allow cancelling the close operation
            response = messagebox.askyesnocancel(
                "Unsaved Changes", "You have unsaved changes. Save before closing?"
            )
            if response is True: # Yes
                self._save_refined_annotations()
                if hasattr(self, "save_successful") and self.save_successful:
                    self.root.destroy()
                # else: stay open if save failed
            elif response is False: # No
                print("Discarding changes and closing.")
                self.root.destroy()  # Discard changes
            # else: response is None (Cancel), do nothing
            else:
                print("Close operation cancelled.")
        else:
            self.root.destroy()

    def _complete(self):
        """Save and close the application."""
        if self.modifications_made:
            self._save_refined_annotations()
            # Only destroy if save was successful (or user chose not to save)
            if hasattr(self, "save_successful") and self.save_successful:
                self.root.destroy()
        else:
            messagebox.showinfo("No Changes", "No modifications were made.")
            self.root.destroy()

    def _save_refined_annotations(self):
        """Save the modified annotations to a new COCO JSON file."""
        self.save_successful = False  # Reset flag
        print("Saving refined annotations...")
        output_data = {"images": [], "annotations": [], "categories": self.categories}

        final_image_ids = set(self.image_ids) - self.deleted_image_ids
        print(
            f"Keeping {len(final_image_ids)} images out of {len(self.image_ids)} total."
        )

        # Check if any images remain undecided
        undecided_images = [
            img_id for img_id in final_image_ids
            if self.image_decision_state.get(img_id) == 'undecided'
        ]
        if undecided_images:
            msg = (
                f"Warning: {len(undecided_images)} images have not had a decision made "
                "(Original vs Shrunk):\n" + ", ".join(map(str, undecided_images[:10])) + ("..." if len(undecided_images) > 10 else "") +
                "\n\nAnnotations for these images will be saved based on their current state "
                "(likely the original). Continue saving?"
             )
            if not messagebox.askyesno("Undecided Images", msg):
                print("Save cancelled by user due to undecided images.")
                self.save_successful = False
                return # Abort save

        # Add image info for non-deleted images
        for img_id in self.image_ids:
            if img_id not in self.deleted_image_ids:
                img_info = self.image_map[img_id]
                output_data["images"].append(
                    {
                        "id": img_id,
                        "width": img_info["width"],
                        "height": img_info["height"],
                        "file_name": img_info["file_name"],
                    }
                )

        # Add annotations for non-deleted images and re-ID them
        current_ann_id = 1
        for img_id in final_image_ids:
            for ann in self.modified_annotations.get(img_id, []):
                ann["id"] = current_ann_id  # Assign new sequential ID
                ann["image_id"] = img_id  # Ensure image_id is correct
                output_data["annotations"].append(ann)
                current_ann_id += 1

        print(
            f"Saving {len(output_data['annotations'])} annotations for {len(output_data['images'])} images."
        )

        try:
            with open(self.output_path, "w") as f:
                json.dump(output_data, f, indent=4)  # Use indent for readability
            messagebox.showinfo(
                "Save Successful", f"Refined annotations saved to:\n{self.output_path}"
            )
            self.modifications_made = False  # Reset flag after successful save
            self.save_successful = True
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save annotations:\n{e}")
            self.save_successful = False

    def _save_temp_progress(self, event=None):
        """Saves current progress to the output file without closing."""
        print("Attempting temporary save...")
        self._save_refined_annotations() # Use the main save function
        if hasattr(self, "save_successful") and self.save_successful:
            print("Temporary save successful.")
            self.modifications_made = False # Reset modification flag after successful save
        else:
            print("Temporary save failed.")
            # Optionally provide more feedback
            messagebox.showwarning("Save Failed", "Could not save temporary progress.")

    # --- Hotkey Decision Methods ---
    def _accept_original(self, event=None):
        """Handles the 'A' key press to keep the original annotations."""
        current_img_id = self.image_ids[self.current_idx]
        if self.image_decision_state.get(current_img_id) == 'undecided':
            print(f"Image {current_img_id}: Keeping ORIGINAL annotations.")
            self.image_decision_state[current_img_id] = 'original'
            # Deep copy original annotations into the active modified set for this image
            self.modified_annotations[current_img_id] = copy.deepcopy(
                self.original_annotations.get(current_img_id, [])
            )
            self.modifications_made = True # Mark that a decision was made
            self._draw_annotations() # Redraw in active mode
        else:
            print(f"Decision already made for image {current_img_id}.")

    def _accept_shrunk(self, event=None):
        """Handles the 'D' key press to keep the shrunk annotations."""
        current_img_id = self.image_ids[self.current_idx]
        if self.image_decision_state.get(current_img_id) == 'undecided':
            print(f"Image {current_img_id}: Keeping SHRUNK annotations.")
            self.image_decision_state[current_img_id] = 'shrunk'
            # Deep copy shrunk annotations into the active modified set for this image
            self.modified_annotations[current_img_id] = copy.deepcopy(
                self.shrunk_annotations.get(current_img_id, [])
            )
            self.modifications_made = True # Mark that a decision was made
            self._draw_annotations() # Redraw in active mode
        else:
            print(f"Decision already made for image {current_img_id}.")

    def run(self):
        """Start the Tkinter main loop."""
        if not self.image_ids:
            print("Initialization failed or no images found. GUI not started.")
            return

        self.root.mainloop()


# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactively refine COCO bounding boxes and poses.")
    parser.add_argument("--img_dir", required=True, help="Path to the directory containing images.")
    parser.add_argument("--coco_path", required=True, help="Path to the COCO annotation JSON file.")
    parser.add_argument("--shrink", type=float, default=None, help="Optional percentage (0-100) to shrink boxes. If provided, starts in comparison mode.")
    parser.add_argument("--view_mode", default="bbox", choices=["bbox", "pose"], help="Initial view mode ('bbox' or 'pose').")

    args = parser.parse_args()

    # Validate shrink percentage if provided
    if args.shrink is not None and not (0 < args.shrink < 100):
        print(f"Error: --shrink value must be between 0 and 100 (exclusive). Got {args.shrink}")
        exit(1)

    # Use args directly
    img_dir = args.img_dir
    coco_path = args.coco_path
    view_mode = args.view_mode
    shrink_percentage = args.shrink # Will be None if not provided

    print(f"Image Dir: {img_dir}")
    print(f"COCO Path: {coco_path}")
    print(f"Initial View: {view_mode}")
    if shrink_percentage is not None:
        print(f"Shrink Percentage: {shrink_percentage}%")
    else:
        print("Shrinking: Not requested.")


    image_data, anno_data, category_data, next_ann_id = load_coco_data(
        coco_path, img_dir
    )

    if image_data is not None:
        shrunk_anno_data = None # Initialize
        perform_shrink_comparison = shrink_percentage is not None

        if perform_shrink_comparison:
            print(f"Calculating shrunk annotations ({shrink_percentage}%)...")
            # We need the full coco structure temporarily for calculate_shrunk_bboxes
            temp_coco_data_for_shrinking = {
                "images": [
                    {
                        "id": img_id, "width": info["width"], "height": info["height"],
                         "file_name": info["file_name"]
                    } for img_id, info in image_data.items()
                ],
                "annotations": [
                    ann for anns_list in anno_data.values() for ann in anns_list
                ],
                "categories": category_data # Pass categories if needed by shrink logic
            }
            try:
                shrunk_full_coco_data, _ = calculate_shrunk_bboxes(
                    temp_coco_data_for_shrinking, shrink_percentage
                )
                # Re-map the shrunk annotations back to the dictionary format needed by the GUI
                shrunk_anno_data = {img_id: [] for img_id in image_data}
                for ann in shrunk_full_coco_data.get("annotations", []):
                    img_id = ann["image_id"]
                    if img_id in shrunk_anno_data:
                        shrunk_anno_data[img_id].append(ann)
                print("Shrunk annotation calculation complete.")
            except Exception as e:
                messagebox.showerror("Shrinking Error", f"Failed to calculate shrunk boxes: {e}")
                print(f"Error during shrinking calculation: {e}")
                # Exit if shrinking was requested but failed?
                print("Exiting due to shrinking error.")
                exit(1)
                # shrunk_anno_data = copy.deepcopy(anno_data) # Fallback to original
                # messagebox.showwarning("Shrinking Error", "Proceeding with original boxes only.")
                # perform_shrink_comparison = False # Turn off comparison if shrinking failed
        else:
             # If not shrinking, create a placeholder (or could use None)
             # The GUI __init__ will handle this case.
             shrunk_anno_data = {} # Or copy.deepcopy(anno_data) if needed as fallback

        # --- End Shrinking Step ---

        print("Launching Refinement GUI...")
        gui = COCORefinementGUI(
            image_data,
            anno_data, # Pass original annotations
            shrunk_anno_data if perform_shrink_comparison else {}, # Pass empty dict if not comparing
            category_data,
            view_mode,
            coco_path,
            next_ann_id,
        )
        gui.run()
        print("GUI Closed.")
    else:
        print("Failed to load COCO data. Exiting.")

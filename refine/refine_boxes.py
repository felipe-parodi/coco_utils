# refine_boxes.py

# i have a COCO dataset with a dir of images and a dir of annotations
# i want to refine the boxes in a fast GUI

#TODO:
# - add docstring
# - why does the gui look low-res?


import json
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict, List, Set, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageTk

# --- COCO Data Loading ---


def load_coco_data(
    coco_json_path: str, img_dir: str
) -> Tuple[Dict[int, Dict], Dict[int, List[Dict]], List[Dict], int]:
    """Loads COCO data and organizes it for the GUI."""
    print(f"Loading COCO data from: {coco_json_path}")
    try:
        with open(coco_json_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        messagebox.showerror("Error", f"COCO JSON file not found: {coco_json_path}")
        return None, None, None, -1
    except json.JSONDecodeError:
        messagebox.showerror(
            "Error", f"Error decoding COCO JSON file: {coco_json_path}"
        )
        return None, None, None, -1

    image_map = {}
    max_image_id = 0
    print("Mapping images...")
    for img in data.get("images", []):
        img_id = img["id"]
        img_path = os.path.join(img_dir, img["file_name"])
        if not os.path.exists(img_path):
            print(f"Warning: Image file not found, skipping: {img_path}")
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


# --- Setup Dialog ---


def launch_setup_dialog() -> Union[Tuple[str, str, str], None]:
    """Launches a dialog to get user input paths and view mode."""
    setup_root = tk.Tk()
    setup_root.title("Setup Refinement")
    setup_root.geometry("500x200")

    img_dir = tk.StringVar()
    coco_json_path = tk.StringVar()
    initial_view_mode = tk.StringVar(value="bbox")  # Default to bbox view
    result = None

    def browse_img_dir():
        dir_path = filedialog.askdirectory(title="Select Image Directory")
        if dir_path:
            img_dir.set(dir_path)

    def browse_coco_file():
        file_path = filedialog.askopenfilename(
            title="Select COCO JSON File", filetypes=[("JSON files", "*.json")]
        )
        if file_path:
            coco_json_path.set(file_path)

    def start_refinement():
        nonlocal result
        if not img_dir.get() or not coco_json_path.get():
            messagebox.showwarning(
                "Input Missing",
                "Please select both image directory and COCO JSON file.",
            )
            return
        result = (img_dir.get(), coco_json_path.get(), initial_view_mode.get())
        setup_root.destroy()

    ttk.Label(setup_root, text="Image Directory:").grid(
        row=0, column=0, padx=5, pady=5, sticky="w"
    )
    ttk.Entry(setup_root, textvariable=img_dir, width=40).grid(
        row=0, column=1, padx=5, pady=5
    )
    ttk.Button(setup_root, text="Browse...", command=browse_img_dir).grid(
        row=0, column=2, padx=5, pady=5
    )

    ttk.Label(setup_root, text="COCO JSON File:").grid(
        row=1, column=0, padx=5, pady=5, sticky="w"
    )
    ttk.Entry(setup_root, textvariable=coco_json_path, width=40).grid(
        row=1, column=1, padx=5, pady=5
    )
    ttk.Button(setup_root, text="Browse...", command=browse_coco_file).grid(
        row=1, column=2, padx=5, pady=5
    )

    ttk.Label(setup_root, text="Initial View:").grid(
        row=2, column=0, padx=5, pady=5, sticky="w"
    )
    view_frame = ttk.Frame(setup_root)
    ttk.Radiobutton(
        view_frame, text="Boxes Only", variable=initial_view_mode, value="bbox"
    ).pack(side=tk.LEFT, padx=5)
    ttk.Radiobutton(
        view_frame, text="Boxes and Poses", variable=initial_view_mode, value="pose"
    ).pack(side=tk.LEFT, padx=5)
    view_frame.grid(row=2, column=1, columnspan=2, sticky="w", padx=5, pady=5)

    button_frame = ttk.Frame(setup_root)
    ttk.Button(button_frame, text="Start", command=start_refinement).pack(
        side=tk.LEFT, padx=10
    )
    ttk.Button(button_frame, text="Cancel", command=setup_root.destroy).pack(
        side=tk.LEFT, padx=10
    )
    button_frame.grid(row=3, column=0, columnspan=3, pady=15)

    setup_root.mainloop()
    return result


# --- Main Refinement GUI ---


class COCORefinementGUI:
    """Tkinter GUI for refining COCO annotations."""

    def __init__(
        self,
        image_map: Dict[int, Dict],
        annotation_map: Dict[int, List[Dict]],
        categories: List[Dict],
        initial_view_mode: str,
        coco_json_path: str,
        next_start_ann_id: int,
    ):
        self.image_map = image_map
        self.categories = categories
        self.coco_json_path = coco_json_path
        self.output_path = coco_json_path.replace(".json", "_refined.json")

        self.modified_annotations = {
            img_id: [ann.copy() for ann in anns]
            for img_id, anns in annotation_map.items()
        }  # Deep copy
        self.image_ids = sorted(list(image_map.keys()))  # Process in a consistent order

        if not self.image_ids:
            messagebox.showerror("Error", "No valid images loaded. Exiting.")
            return  # Or raise an exception

        self.current_idx = 0
        self.view_mode = initial_view_mode
        self.deleted_image_ids = set()
        self.modifications_made = False  # Track if any changes occurred
        self.next_new_ann_id = next_start_ann_id  # ID for newly created annotations

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
        self.progress_var = tk.StringVar()
        ttk.Label(bottom_ctrl_frame, textvariable=self.progress_var).pack(
            side=tk.LEFT, padx=20
        )

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
            self.progress_var.set("No images remaining.")
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

        self._draw_annotations()

        # Update progress
        self.progress_var.set(
            f"Image {self.current_idx + 1} of {len(self.image_ids)} (ID: {current_img_id})"
        )
        self.root.update_idletasks()  # Ensure canvas resizes

    def _draw_annotations(self):
        """Draw annotations based on current view mode."""
        self.canvas.delete(
            "box", "handle", "pose", "keypoint", "skeleton"
        )  # Clear specific tags

        current_img_id = self.image_ids[self.current_idx]
        annotations = self.modified_annotations.get(current_img_id, [])

        if not annotations:
            return

        self._draw_boxes(annotations)
        if self.view_mode == "pose":
            self._draw_poses(annotations)

    def _draw_boxes(self, annotations):
        """Draw bounding boxes and handles."""
        for ann in annotations:
            ann_id = ann["id"]
            bbox = ann.get("bbox")  # COCO format: [x, y, width, height]
            if not bbox:
                continue

            x, y, w, h = bbox
            x1 = x * self.scale + self.offset_x
            y1 = y * self.scale + self.offset_y
            x2 = (x + w) * self.scale + self.offset_x
            y2 = (y + h) * self.scale + self.offset_y

            # Draw box
            box_tag = f"ann{ann_id}"
            self.canvas.create_rectangle(
                x1, y1, x2, y2, outline="red", width=2, tags=("box", box_tag)
            )

            # Draw handles
            handles_coords = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
            for hx, hy in handles_coords:
                self.canvas.create_rectangle(
                    hx - self.handle_size / 2,
                    hy - self.handle_size / 2,
                    hx + self.handle_size / 2,
                    hy + self.handle_size / 2,
                    fill="white",
                    outline="red",
                    tags=("handle", box_tag),
                )

    def _draw_poses(self, annotations):
        """Draw keypoints and skeletons."""
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
        self.view_mode = "pose" if self.view_mode == "bbox" else "bbox"
        self.view_mode_label.set(f"View: {self.view_mode.capitalize()}")
        self._draw_annotations()

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
        print(f"Mouse Down at ({event.x}, {event.y})")
        self.drag_start = (event.x, event.y)
        self.selected_ann_id = None
        self.drag_type = None
        self.selected_corner = None

        # Check if in new box creation mode first
        if self.creating_new_box:
            self.drag_type = "create"
            self.new_box_start = (event.x, event.y)
            print("Starting new box creation")
            return

        # Find items under cursor
        items = self.canvas.find_overlapping(
            event.x - 1, event.y - 1, event.x + 1, event.y + 1
        )
        if not items:
            return

        # Prioritize handles, then boxes
        clicked_handle = None
        clicked_box = None

        for item in reversed(items):  # Check topmost items first
            tags = self.canvas.gettags(item)
            if "handle" in tags:
                clicked_handle = item
                break
            elif "box" in tags:
                clicked_box = item
                # Don't break yet, check if a handle is on top

        if clicked_handle:
            tags = self.canvas.gettags(clicked_handle)
            ann_tag = next((t for t in tags if t.startswith("ann")), None)
            if ann_tag:
                self.selected_ann_id = int(ann_tag[3:])  # Extract ann_id
                self.drag_type = "resize"
                # Determine which corner handle was clicked
                coords = self.canvas.coords(clicked_handle)
                cx = (coords[0] + coords[2]) / 2
                cy = (coords[1] + coords[3]) / 2

                # Find the corresponding box to get its corners
                ann = self._find_annotation_by_id(self.selected_ann_id)
                if ann and "bbox" in ann:
                    x, y, w, h = ann["bbox"]
                    x1 = x * self.scale + self.offset_x
                    y1 = y * self.scale + self.offset_y
                    x2 = (x + w) * self.scale + self.offset_x
                    y2 = (y + h) * self.scale + self.offset_y

                    # Check proximity to scaled corners
                    if (
                        abs(cx - x1) < self.handle_size
                        and abs(cy - y1) < self.handle_size
                    ):
                        self.selected_corner = "topleft"
                    elif (
                        abs(cx - x2) < self.handle_size
                        and abs(cy - y1) < self.handle_size
                    ):
                        self.selected_corner = "topright"
                    elif (
                        abs(cx - x1) < self.handle_size
                        and abs(cy - y2) < self.handle_size
                    ):
                        self.selected_corner = "bottomleft"
                    elif (
                        abs(cx - x2) < self.handle_size
                        and abs(cy - y2) < self.handle_size
                    ):
                        self.selected_corner = "bottomright"
                    else:
                        self.drag_type = (
                            None  # Should not happen if handle logic is correct
                        )
                    print(
                        f"Selected handle for ann {self.selected_ann_id}, corner: {self.selected_corner}"
                    )
                else:
                    self.drag_type = None  # Error finding annotation

        elif clicked_box:
            tags = self.canvas.gettags(clicked_box)
            ann_tag = next((t for t in tags if t.startswith("ann")), None)
            if ann_tag:
                self.selected_ann_id = int(ann_tag[3:])
                self.drag_type = "move"
                # Calculate offset from top-left corner for smooth dragging
                ann = self._find_annotation_by_id(self.selected_ann_id)
                if ann and "bbox" in ann:
                    x, y, _, _ = ann["bbox"]
                    x1_canvas = x * self.scale + self.offset_x
                    y1_canvas = y * self.scale + self.offset_y
                    self.drag_offset = (event.x - x1_canvas, event.y - y1_canvas)
                    print(f"Selected box for ann {self.selected_ann_id} for moving")
                else:
                    self.drag_type = None

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

            if self.drag_type == "move":
                new_x1_canvas = event.x - self.drag_offset[0]
                new_y1_canvas = event.y - self.drag_offset[1]
                ann["bbox"][0] = (new_x1_canvas - self.offset_x) / self.scale
                ann["bbox"][1] = (new_y1_canvas - self.offset_y) / self.scale
                # Width and height remain unchanged
            elif self.drag_type == "resize":
                # Update bbox based on corner being dragged
                if self.selected_corner == "topleft":
                    new_x2 = x + w  # Bottom-right x stays
                    new_y2 = y + h  # Bottom-right y stays
                    x = current_x_img
                    y = current_y_img
                    w = new_x2 - x
                    h = new_y2 - y
                elif self.selected_corner == "topright":
                    new_x1 = x  # Top-left x stays
                    new_y2 = y + h  # Bottom-right y stays
                    y = current_y_img
                    w = current_x_img - new_x1
                    h = new_y2 - y
                elif self.selected_corner == "bottomleft":
                    new_x2 = x + w  # Bottom-right x stays
                    new_y1 = y  # Top-left y stays
                    x = current_x_img
                    w = new_x2 - x
                    h = current_y_img - new_y1
                elif self.selected_corner == "bottomright":
                    new_x1 = x  # Top-left x stays
                    new_y1 = y  # Top-left y stays
                    w = current_x_img - new_x1
                    h = current_y_img - new_y1

                # Update bbox ensuring w, h are positive
                # If width/height becomes negative, swap corners and update selected_corner
                if w < 0:
                    x = x + w  # New x is the right edge
                    w = abs(w)
                    if self.selected_corner == "topleft":
                        self.selected_corner = "topright"
                    elif self.selected_corner == "topright":
                        self.selected_corner = "topleft"
                    elif self.selected_corner == "bottomleft":
                        self.selected_corner = "bottomright"
                    elif self.selected_corner == "bottomright":
                        self.selected_corner = "bottomleft"
                if h < 0:
                    y = y + h  # New y is the bottom edge
                    h = abs(h)
                    if self.selected_corner == "topleft":
                        self.selected_corner = "bottomleft"
                    elif self.selected_corner == "bottomleft":
                        self.selected_corner = "topleft"
                    elif self.selected_corner == "topright":
                        self.selected_corner = "bottomright"
                    elif self.selected_corner == "bottomright":
                        self.selected_corner = "topright"

                ann["bbox"] = [x, y, w, h]

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
                    "id": self.next_new_ann_id,
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
                self.next_new_ann_id += 1
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
    def _prev_image(self):
        if self.image_ids:
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
            if messagebox.askyesno(
                "Unsaved Changes", "You have unsaved changes. Save before closing?"
            ):
                self._save_refined_annotations()
                if hasattr(self, "save_successful") and self.save_successful:
                    self.root.destroy()
                # else: stay open if save failed
            else:
                self.root.destroy()  # Discard changes
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

    def run(self):
        """Start the Tkinter main loop."""
        if not self.image_ids:
            print("Initialization failed or no images found. GUI not started.")
            return

        self.root.mainloop()


# --- Main Execution ---

if __name__ == "__main__":
    setup_result = launch_setup_dialog()

    if setup_result:
        img_dir, coco_path, view_mode = setup_result
        print(f"Image Dir: {img_dir}")
        print(f"COCO Path: {coco_path}")
        print(f"Initial View: {view_mode}")

        image_data, anno_data, category_data, next_ann_id = load_coco_data(
            coco_path, img_dir
        )

        if image_data is not None:
            print("Launching Refinement GUI...")
            gui = COCORefinementGUI(
                image_data, anno_data, category_data, view_mode, coco_path, next_ann_id
            )
            gui.run()
            print("GUI Closed.")
        else:
            print("Failed to load COCO data. Exiting.")
    else:
        print("Setup cancelled.")

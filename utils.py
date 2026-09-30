#!/usr/bin/env python3
"""Utility functions for HyperTex dataset management, ENVI hyperspectral loading,

LabelMe annotation parsing, ground-truth map generation, and HDF5 export.
"""

import cv2
import os
import json
import time
from glob import glob
import numpy as np
import h5py
import matplotlib.pyplot as plt
import spectral


def directory_loader(dir):
    """Enumerate available captures in a raw dataset directory.

    Returns list of tuples: [(folder_name, hdr_present, json_present), ...]
    """
    captures = []
    if not os.path.exists(dir):
        return captures

    for f in os.listdir(dir):
        path = os.path.join(dir, f)
        if os.path.isdir(path):
            hyperspectral_hdr = os.path.abspath(
                os.path.join(dir, f, "capture", prefix_hdr(f) + ".hdr")
            )
            if not os.path.exists(hyperspectral_hdr):
                hyperspectral_hdr = os.path.abspath(os.path.join(dir, f, "capture", f + ".hdr"))
            hyperspectral_json = os.path.abspath(os.path.join(dir, f, f + ".json"))

            hdr = "Yes" if os.path.exists(hyperspectral_hdr) else "No"
            labeled = "Yes" if os.path.exists(hyperspectral_json) else "No"

            captures.append((f, hdr, labeled))

    return captures


def prefix_hdr(sample_name):
    return f"REFLECTANCE_{sample_name}"


def gt_ml_to_imgidx(gt_ml):
    """Convert multi-label 3D GT composition array (H, W, 10) to 2D class index map."""
    mask = np.any((gt_ml > 0), axis=2)
    return np.argmax(gt_ml, axis=2) + 1 * mask


def dataset_to_hdf5(captures_dir, h5_name, dataset=None, prefix="REFLECTANCE_"):
    """Export selected sample captures and ground-truth maps into HDF5 dataset.

    Creates groups '/data' (reflectance cubes) and '/gt' (10-channel composition maps).
    """
    if not os.path.exists(captures_dir):
        return [], []

    if dataset is not None:
        if not os.path.exists(dataset):
            print("Dataset file does not exist")
            return {}, None
        with open(dataset, "r") as train_ls:
            Samples = train_ls.read().splitlines()
    else:
        Samples = os.listdir(captures_dir)

    out_file = h5_name if h5_name.endswith(".hdf5") or h5_name.endswith(".h5") else h5_name + ".hdf5"
    with h5py.File(out_file, "w") as h5_file:
        h5_file.create_group("/data")
        h5_file.create_group("/gt")
        count = 0
        start = time.time()
        for f in Samples:
            matches = glob(os.path.abspath(os.path.join(captures_dir, f + "*")))
            if not matches:
                continue
            capture_name = str(matches[0]).replace(captures_dir.rstrip("/") + "/", "")

            try:
                hyperspectral_hdr = str(
                    glob(
                        os.path.abspath(
                            os.path.join(
                                captures_dir, capture_name, "capture", prefix + capture_name + "*.hdr"
                            )
                        )
                    )[0]
                )
            except Exception:
                hyperspectral_hdr = os.path.abspath(
                    os.path.join(captures_dir, capture_name, "capture", prefix + capture_name + ".hdr")
                )

            try:
                hyperspectral_json = str(
                    glob(
                        os.path.abspath(
                            os.path.join(captures_dir, capture_name, capture_name + "*.json")
                        )
                    )[0]
                )
            except Exception:
                hyperspectral_json = os.path.abspath(
                    os.path.join(captures_dir, capture_name, capture_name + ".json")
                )

            try:
                hyperspectral_npy = str(
                    glob(
                        os.path.abspath(
                            os.path.join(captures_dir, capture_name, capture_name + "*.npy")
                        )
                    )[0]
                )
            except Exception:
                hyperspectral_npy = os.path.abspath(
                    os.path.join(captures_dir, capture_name, capture_name + ".npy")
                )

            string = str(count) + ": " + f
            if os.path.exists(hyperspectral_hdr) and (
                os.path.exists(hyperspectral_json) or os.path.exists(hyperspectral_npy)
            ):
                img = spectral.open_image(hyperspectral_hdr).open_memmap()
                if os.path.exists(hyperspectral_npy):
                    gt = np.load(hyperspectral_npy, mmap_mode="r")
                else:
                    gt = multi_labelme(hyperspectral_json)

                h5_file.create_dataset("/data/" + f, data=img)
                h5_file.create_dataset("/gt/" + f, data=gt)
                string += " OK"
                count += 1
            print(string)

    print("TIME: " + str(time.time() - start))


def multi_labelme(labelme_file, convert_unkowns=True, en=False):
    """Read LabelMe annotation JSON file and convert it to 10-channel multi-label GT map.

    Returns numpy array of shape (height, width, n_classes).
    """
    if labelme_file.endswith(".npy"):
        return np.load(labelme_file)
    else:
        with open(labelme_file, "r") as f:
            labelme_data = json.load(f)

        classes = read_categories(en=en)
        category_id_map = {v: k for k, v in classes.items()}

        height = labelme_data["imageHeight"]
        width = labelme_data["imageWidth"]
        np_label = np.zeros((height, width, len(category_id_map)), dtype=np.float16)

        for shape in labelme_data["shapes"]:
            multi_label = shape["label"]
            labels = multi_label.split("_")
            description = shape.get("description", "") or "100"
            percentages = description.split("_")
            if percentages[0] == "":
                percentages[0] = "100"

            np_label_tmp = np.zeros((len(classes)), dtype=np.float16)
            mask = np.zeros((height, width), dtype=np.uint8)

            total_perc = 0.0
            for label, perc in zip(labels, percentages):
                try:
                    if label not in category_id_map and convert_unkowns:
                        np_label_tmp[0] = float(perc) / 100  # Unknown Class
                        print(
                            f"Capture {labelme_file} with Label {label} not in class subset. Defaulted to Unknown (0)"
                        )
                    else:
                        np_label_tmp[category_id_map[label] - 1] = float(perc) / 100
                    total_perc += float(perc) / 100
                except Exception as e:
                    print(f"Error reading annotated data: {labelme_file} ({label}): {e}")

            if not np.isclose(total_perc, 1.0, rtol=1e-3, atol=1e-4):
                print(f"Percentage does not equal 1: {labelme_file} - {total_perc}")

            polygon = shape["points"]
            segmentation = [np.array(polygon).flatten().tolist()]
            mask_label = segmentation[0]
            mask_points = np.array(mask_label).reshape((-1, 2)).astype(int)
            cv2.fillPoly(mask, [mask_points], 255)
            np_label[mask == 255] = np_label_tmp

        return np_label


def read_categories(en=True):
    """Load dictionary mapping class indices (1..10) to class names."""
    base_dir = os.path.dirname(__file__)
    categories_file = (
        os.path.join(base_dir, "classes_en.txt")
        if en and os.path.exists(os.path.join(base_dir, "classes_en.txt"))
        else os.path.join(base_dir, "classes.txt")
    )

    cat = {}
    if not os.path.exists(categories_file):
        return cat

    with open(categories_file, "r") as file:
        for line in file.read().splitlines():
            id_cat = line.split(" ")
            if len(id_cat) >= 2 and id_cat[0] != "":
                cat[int(id_cat[0])] = id_cat[1]

    return cat


def hdf5_directory_loader(h5_file_path):
    """Enumerate samples available in an HDF5 dataset file.

    Returns list of tuples: [(sample_name, hdr, labeled), ...]
    """
    captures = []
    if not os.path.exists(h5_file_path):
        return captures
    try:
        with h5py.File(h5_file_path, "r") as h5_file:
            if "/data" in h5_file:
                data_group = h5_file["/data"]
                gt_group = h5_file["/gt"] if "/gt" in h5_file else {}
                for sample_name in data_group.keys():
                    hdr = "Yes"
                    labeled = "Yes" if sample_name in gt_group else "No"
                    captures.append((sample_name, hdr, labeled))
            else:
                for key in h5_file.keys():
                    if isinstance(h5_file[key], h5py.Dataset):
                        captures.append((key, "Yes", "No"))
    except Exception as e:
        print(f"Error loading HDF5 file {h5_file_path}: {e}")
    return sorted(captures)


def load_hdf5_sample(h5_file_path, sample_name):
    """Load HSI data and optional GT data from an HDF5 dataset file.

    Returns (hsi_data, gt_data)
    """
    if not os.path.exists(h5_file_path):
        return None, None
    try:
        with h5py.File(h5_file_path, "r") as h5_file:
            hsi_data = None
            gt_data = None
            if "/data/" + sample_name in h5_file:
                hsi_data = h5_file["/data/" + sample_name][()]
            elif sample_name in h5_file:
                hsi_data = h5_file[sample_name][()]

            if "/gt/" + sample_name in h5_file:
                gt_data = h5_file["/gt/" + sample_name][()]
            return hsi_data, gt_data
    except Exception as e:
        print(f"Error reading sample {sample_name} from HDF5 {h5_file_path}: {e}")
        return None, None


def load_hdf5_sample_data(h5_file_path, sample_name):
    """Load reflectance, optional raw signal, and GT for an HDF5 sample."""
    if not os.path.exists(h5_file_path):
        return None, None, None
    try:
        with h5py.File(h5_file_path, "r") as h5_file:
            reflectance = None
            raw_signal = None
            gt_data = None
            if "/data/" + sample_name in h5_file:
                reflectance = h5_file["/data/" + sample_name][()]
            elif sample_name in h5_file and isinstance(h5_file[sample_name], h5py.Dataset):
                reflectance = h5_file[sample_name][()]
            if "/raw/" + sample_name in h5_file:
                raw_signal = h5_file["/raw/" + sample_name][()]
            if "/gt/" + sample_name in h5_file:
                gt_data = h5_file["/gt/" + sample_name][()]
            return reflectance, raw_signal, gt_data
    except Exception as e:
        print(f"Error reading sample {sample_name} from HDF5 {h5_file_path}: {e}")
        return None, None, None


def detect_sample_gt(captures_dir, sample_name):
    """Detect and load GT mask associated with a sample in a directory.

    Returns GT numpy array or None.
    """
    sample_folder = os.path.join(captures_dir, sample_name)
    if not os.path.exists(sample_folder):
        matches = glob(os.path.join(captures_dir, sample_name + "*"))
        if matches:
            sample_folder = matches[0]

    if not os.path.exists(sample_folder):
        return None

    npy_files = glob(os.path.join(sample_folder, "*.npy"))
    if npy_files:
        try:
            return np.load(npy_files[0])
        except Exception as e:
            print(f"Error loading NPY GT file {npy_files[0]}: {e}")

    jf = find_sample_labelme_json(captures_dir, sample_name)
    if jf:
        try:
            return multi_labelme(jf)
        except Exception as e:
            print(f"Error loading JSON GT file {jf}: {e}")
    return None


def find_sample_labelme_json(captures_dir, sample_name):
    """Return the original LabelMe annotation path for a sample, if any."""
    sample_folder = os.path.join(captures_dir, sample_name)
    if not os.path.isdir(sample_folder):
        matches = glob(os.path.join(captures_dir, sample_name + "*"))
        sample_folder = matches[0] if matches else sample_folder
    for json_path in glob(os.path.join(sample_folder, "*.json")):
        if not json_path.endswith("coco.json"):
            return json_path
    return None


def labelme_pixel_annotation(json_path, x, y):
    """Return every original LabelMe label/percentage covering (x, y)."""
    if not json_path or not os.path.exists(json_path):
        return []
    try:
        with open(json_path, "r") as annotation_file:
            data = json.load(annotation_file)
        annotations = []
        for shape in data.get("shapes", []):
            points = np.asarray(shape.get("points", []), dtype=np.float32)
            contour = points.reshape((-1, 1, 2)) if len(points) >= 3 else points
            if len(points) < 3 or cv2.pointPolygonTest(contour, (float(x), float(y)), False) < 0:
                continue
            labels = str(shape.get("label", "")).split("_")
            percentages = str(shape.get("description", "") or "100").split("_")
            for index, label in enumerate(labels):
                percentage = percentages[index] if index < len(percentages) else "100"
                try:
                    percentage = float(percentage)
                except (TypeError, ValueError):
                    percentage = 100.0
                annotations.append((label, percentage))
        return annotations
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Error reading LabelMe annotation {json_path}: {error}")
        return []


def gt_to_colored_rgb(gt_mask):
    """Convert GT mask (2D index array or 3D multi-label array) into an RGB overlay image."""
    if gt_mask is None:
        return None

    if gt_mask.ndim == 3:
        gt_indices = gt_ml_to_imgidx(gt_mask)
    else:
        gt_indices = gt_mask.copy()

    h, w = gt_indices.shape[:2]
    rgb_img = np.zeros((h, w, 3), dtype=np.uint8)

    cmap = plt.get_cmap("Set1")
    num_classes = max(10, int(np.max(gt_indices)) + 1)
    colors = (cmap(np.linspace(0, 1, 10)) * 255).astype(np.uint8)[:, :3]

    for cls_id in range(1, num_classes):
        mask = gt_indices == cls_id
        if np.any(mask):
            color = colors[(cls_id - 1) % len(colors)]
            rgb_img[mask] = color

    return rgb_img


def compute_mean_gt_spectrum(hsi_cube, gt_mask, class_id=None):
    """Compute average spectral curve for pixels belonging to class_id or all GT pixels."""
    if hsi_cube is None or gt_mask is None:
        return None

    if hasattr(hsi_cube, "load"):
        hsi_cube = hsi_cube.load()
    hsi_cube = np.asarray(hsi_cube)
    gt_mask = np.asarray(gt_mask)

    h, w = hsi_cube.shape[:2]
    gh, gw = gt_mask.shape[:2]
    if (h != gh) or (w != gw):
        return None

    if gt_mask.ndim == 3:
        num_classes = gt_mask.shape[2]
        if class_id is not None and 1 <= class_id <= num_classes:
            spatial_mask = gt_mask[:, :, class_id - 1] > 0
        else:
            spatial_mask = np.any(gt_mask > 0, axis=-1)
    else:
        if class_id is not None:
            spatial_mask = gt_mask == class_id
        else:
            spatial_mask = gt_mask > 0

    if not np.any(spatial_mask):
        return None

    pixels = hsi_cube[spatial_mask]
    return np.mean(pixels, axis=0)


def get_pixel_gt_info(gt_mask, y, x):
    """Extract GT information for pixel at (y, x).

    Returns (class_id, class_name, gt_vector).
    """
    if gt_mask is None:
        return 0, "None", None

    h, w = gt_mask.shape[:2]
    if y < 0 or y >= h or x < 0 or x >= w:
        return 0, "Out of bounds", None

    categories = read_categories()
    num_categories = len(categories)

    if gt_mask.ndim == 3:
        gt_vec = gt_mask[y, x]
        num_cls = len(gt_vec)
        full_vec = np.zeros(num_categories, dtype=np.float32)
        full_vec[: min(num_cls, num_categories)] = gt_vec[: min(num_cls, num_categories)]
        top_idx = int(np.argmax(gt_vec)) + 1 if np.max(gt_vec) > 0 else 0
        cls_name = categories.get(top_idx, "Unknown") if top_idx > 0 else "Background"
        return top_idx, cls_name, full_vec
    else:
        cls_id = int(gt_mask[y, x])
        full_vec = np.zeros(num_categories, dtype=np.float32)
        if 1 <= cls_id <= num_categories:
            full_vec[cls_id - 1] = 1.0
            cls_name = categories.get(cls_id, "Unknown")
        else:
            cls_name = "Background" if cls_id == 0 else f"Class {cls_id}"
        return cls_id, cls_name, full_vec

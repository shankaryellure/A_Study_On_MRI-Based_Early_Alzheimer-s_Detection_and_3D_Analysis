import os
import re
import math
import numpy as np
import scipy.ndimage as ndimage
import SimpleITK as sitk
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from datetime import datetime
import random
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", message="Could not find the number of physical cores")


def extract_date(filename):
    """
    Extract the date from the filename.
    Assumes the date is in the format YYYY-MM-DD.
    Example:
      AD_002_S_0816_MR_MPRAGE_2007-03-28_I47406_skull_scripted.nii.gz
    Returns a datetime object if found, otherwise returns datetime.max.
    """
    match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
    if match:
        return datetime.strptime(match.group(1), "%Y-%m-%d")
    else:
        return datetime.max

def extract_subject_id(filename):
    """
    Extract the subject ID from the filename.
    Handles prefixes such as 'EMCI_', 'AD_', 'LMCI_', and 'MCI_'.
    Example:
      AD_002_S_0816_MR_MPRAGE_2007-03-28_I47406_skull_scripted.nii.gz
    Returns '002_S_0816' if matched.
    """
    match = re.search(r'(?:EMCI|AD|LMCI|MCI)_([^_]+_[^_]+_[^_]+)', filename)
    if match:
        return match.group(1)
    else:
        return "unknown"

def extract_category(filename):
    """
    Extract the category (AD, MCI, LMCI, EMCI) from the filename prefix.
    Example:
      AD_002_S_0816_MR_MPRAGE_2007-03-28_I47406_skull_scripted.nii.gz -> 'AD'
    """
    match = re.search(r'^(AD|MCI|LMCI|EMCI)_', filename)
    if match:
        return match.group(1)
    else:
        return "Unknown"

def compute_brain_volume(sitk_image):
    """
    Compute the brain volume from an input SimpleITK image.
    Steps:
      1. Normalize intensity to [0,1].
      2. Convert to NumPy array and apply Gaussian smoothing.
      3. Reshape data and perform K-Means segmentation (2 clusters: brain vs. background).
      4. Determine the cluster with higher mean intensity (assumed brain tissue).
      5. Apply morphological closing and keep only the largest connected component.
      6. Use voxel spacing to compute the brain volume (in mm³).
    Returns:
      - brain_volume: Computed volume in mm³.
      - brain_mask: Binary NumPy array of the segmented brain.
      - image_normalized: The intensity-normalized SimpleITK image.
    """
    # Normalize the image intensity to [0, 1]
    rescale_filter = sitk.RescaleIntensityImageFilter()
    rescale_filter.SetOutputMinimum(0.0)
    rescale_filter.SetOutputMaximum(1.0)
    image_normalized = rescale_filter.Execute(sitk_image)
    
    # Convert image to numpy array and smooth it (to reduce noise)
    data = sitk.GetArrayFromImage(image_normalized)
    data_smoothed = ndimage.gaussian_filter(data, sigma=1)
    
    # Prepare data for K-Means clustering by flattening
    flat_data = data_smoothed.reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=0)
    kmeans.fit(flat_data)
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    
    # Assume brain tissue corresponds to cluster with higher mean intensity
    brain_cluster = np.argmax(cluster_centers)
    brain_mask = (labels == brain_cluster).reshape(data.shape)
    
    # Apply morphological closing to smooth the segmentation mask
    brain_mask = ndimage.binary_closing(brain_mask, structure=np.ones((3, 3, 3)))
    
    # Keep only the largest connected component (remove noise/small regions)
    labeled_mask, num_features = ndimage.label(brain_mask)
    if num_features > 0:
        sizes = ndimage.sum(brain_mask, labeled_mask, range(1, num_features + 1))
        largest_component = (sizes.argmax() + 1)
        brain_mask = (labeled_mask == largest_component)
    
    # Use the image spacing to compute the volume (voxel_volume in mm³)
    spacing = image_normalized.GetSpacing()  # (x, y, z) spacing in mm
    voxel_volume = spacing[0] * spacing[1] * spacing[2]
    brain_volume = np.sum(brain_mask) * voxel_volume
    
    return brain_volume, brain_mask, image_normalized

def register_to_baseline(baseline_image, moving_image):
    """
    Registers the moving_image to the baseline_image using SimpleITK.
    Steps:
      1. Set up registration using Mean Squares metric and gradient descent.
      2. Initialize the transform based on image geometry.
      3. Perform registration and execute the final transform.
      4. Resample the moving_image according to the computed transform.
    Returns:
      - moving_resampled: The registered (resampled) moving image.
    """
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMeanSquares()
    registration_method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0,
        minStep=1e-4,
        numberOfIterations=1,
        gradientMagnitudeTolerance=1e-8)
    registration_method.SetInterpolator(sitk.sitkLinear)
    
    # Initialize transform using image geometry (centered initializer)
    initial_transform = sitk.CenteredTransformInitializer(
        baseline_image,
        moving_image,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY)
    
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    final_transform = registration_method.Execute(baseline_image, moving_image)
    
    # Resample the moving image according to the computed transform
    moving_resampled = sitk.Resample(
        moving_image,
        baseline_image,
        final_transform,
        sitk.sitkLinear,
        0.0,
        moving_image.GetPixelID())
    
    return moving_resampled

def main():
    """
    Main function to:
      1. Process all files from the four categories (AD, MCI, LMCI, EMCI).
      2. For each subject:
           - Group files by subject.
           - Consider the earliest scan as baseline.
           - If not the baseline, register current image to the baseline.
           - Compute the brain volume.
           - Record days since first scan (baseline) and actual volume.
      3. Produce multiple plot sets:
           A. With connecting lines (line plots):
              - Combined figure for all subjects.
              - Separate figures per category.
              - Combined multi-panel (subplots) for each category.
           B. Without connecting lines (scatter plots only):
              - Combined figure for all subjects.
              - Separate figures per category.
              - Combined multi-panel (subplots) for each category.
    """
    
    # --- Define directories for the four categories (adjust paths as needed) ---
    root_dirs = [
        r"C:\Users\jaiswal\Desktop\Shankar\AD",
        r"C:\Users\jaiswal\Desktop\Shankar\MCI",
        r"C:\Users\jaiswal\Desktop\Shankar\LMCI",
        r"C:\Users\jaiswal\Desktop\Shankar\EMCI"
    ]
    
    # --- Collect all matching files across the directories ---
    file_paths = []
    for root_dir in root_dirs:
        if not os.path.isdir(root_dir):
            print(f"Warning: The directory '{root_dir}' does not exist or is invalid.")
            continue
        for entry in os.listdir(root_dir):
            full_path = os.path.join(root_dir, entry)
            if os.path.isdir(full_path):
                # Search inside subdirectories
                for f in os.listdir(full_path):
                    if f.endswith("skull_scripted.nii.gz"):
                        file_paths.append(os.path.join(full_path, f))
            else:
                if entry.endswith("skull_scripted.nii.gz"):
                    file_paths.append(full_path)
    
    if not file_paths:
        print("No matching files found across the four categories.")
        return
    
    # --- Group files by subject and record each subject's category ---
    grouped_files = {}      # key: subject id, value: list of (date, file_path)
    subject_categories = {} # key: subject id, value: category (AD, MCI, LMCI, EMCI)
    
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        date = extract_date(filename)
        subj_id = extract_subject_id(filename)
        category = extract_category(filename)
        if subj_id not in grouped_files:
            grouped_files[subj_id] = []
            subject_categories[subj_id] = category
        grouped_files[subj_id].append((date, file_path))
    
    # Sort each subject's files by date (earliest scan first)
    for subj_id in grouped_files:
        grouped_files[subj_id].sort(key=lambda x: x[0])
    
    # --- Process each subject: registration and volume computation ---
    # Dictionary to store processed data: subject_data[subj_id] = list of (days_since_first, date, brain_volume)
    subject_data = {}
    print("Starting volume computation and registration...")
    
    for subj_id, items in grouped_files.items():
        # Baseline (first scan)
        baseline_date, baseline_file_path = items[0]
        try:
            baseline_image = sitk.ReadImage(baseline_file_path, sitk.sitkFloat32)
            baseline_volume, _, _ = compute_brain_volume(baseline_image)
        except Exception as e:
            print(f"Error processing baseline for subject {subj_id}: {e}")
            continue
        
        subject_data[subj_id] = []
        for i, (date, file_path) in enumerate(items):
            filename = os.path.basename(file_path)
            print(f"Subject {subj_id}: Processing {filename} dated {date.strftime('%Y-%m-%d')}")
            try:
                current_image = sitk.ReadImage(file_path, sitk.sitkFloat32)
            except Exception as e:
                print(f"Error reading image {filename}: {e}")
                continue
            
            # Register current image to the baseline if needed
            if i == 0:
                registered_image = current_image
            else:
                try:
                    registered_image = register_to_baseline(baseline_image, current_image)
                except Exception as e:
                    print(f"Error registering image {filename}: {e}")
                    continue
            
            try:
                brain_volume, _, _ = compute_brain_volume(registered_image)
            except Exception as e:
                print(f"Error computing volume for {filename}: {e}")
                continue
            
            days_since_first = (date - baseline_date).days
            subject_data[subj_id].append((days_since_first, date, brain_volume))
    
    if not subject_data:
        print("No subject data processed. Please check your directories and file patterns.")
        return
    
    # --- Define a color mapping for each category ---
    category_colors = {
        "AD": "red",
        "MCI": "blue",
        "LMCI": "green",
        "EMCI": "orange",
        "Unknown": "gray"
    }
    
    # A. WITH CONNECTING LINES
    # A1. Combined Figure (All subjects, connected lines)
    plt.figure("Combined - With Lines", figsize=(12, 6))
    used_labels = {}  # to ensure one legend entry per category
    for subj, records in subject_data.items():
        xs = [r[0] / 365.0 for r in records]  # converting days to years
        ys = [r[2] for r in records]
        cat = subject_categories[subj]
        color = category_colors.get(cat, "gray")
        if cat not in used_labels:
            plt.plot(xs, ys, marker='o', color=color, label=cat, alpha=0.7)
            used_labels[cat] = True
        else:
            plt.plot(xs, ys, marker='o', color=color, alpha=0.7)
    plt.xlabel("Years Since First Scan")
    plt.ylabel("Brain Volume (mm³)")
    plt.title("Combined All Subjects - With Connecting Lines")
    plt.legend(title="Category", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # A2. Separate Figures for Each Category (Connected Lines)
    for cat in ["AD", "MCI", "LMCI", "EMCI"]:
        plt.figure(f"{cat} - With Lines", figsize=(12, 6))
        for subj, records in subject_data.items():
            if subject_categories[subj] == cat:
                xs = [r[0] / 365.0 for r in records]
                ys = [r[2] for r in records]
                # Use the category's color for plotting
                plt.plot(xs, ys, marker='o', color=category_colors[cat], alpha=0.7, label=subj)
        plt.xlabel("Years Since First Scan")
        plt.ylabel("Brain Volume (mm³)")
        plt.title(f"{cat} Subjects - With Connecting Lines")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(title="Subject", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
    
    # A3. Combined Multi-Panel Figure (Connected Lines)
    fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    fig.suptitle("Brain Volume Over Time by Category (With Lines)", fontsize=30)
    cat_list = ["AD", "MCI", "LMCI", "EMCI"]
    for i, cat in enumerate(cat_list):
        ax = axs[i // 2, i % 2]
        for subj, records in subject_data.items():
            if subject_categories[subj] == cat:
                xs = [r[0] / 365.0 for r in records]
                ys = [r[2] for r in records]
                ax.plot(xs, ys, marker='o', color=category_colors[cat], alpha=0.7)
        ax.set_title(cat)
        ax.set_xlabel("Years")
        ax.set_ylabel("Brain Volume (mm³)")
        ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # B. WITHOUT CONNECTING LINES (Points Only)
    # B1. Combined Figure (All Subjects, Points Only)
    plt.figure("Combined - Points Only", figsize=(12, 6))
    used_labels = {}
    for subj, records in subject_data.items():
        xs = [r[0] / 365.0 for r in records]
        ys = [r[2] for r in records]
        cat = subject_categories[subj]
        color = category_colors.get(cat, "gray")
        if cat not in used_labels:
            plt.scatter(xs, ys, color=color, label=cat, alpha=0.7)
            used_labels[cat] = True
        else:
            plt.scatter(xs, ys, color=color, alpha=0.7)
    plt.xlabel("Years Since First Scan")
    plt.ylabel("Brain Volume (mm³)")
    plt.title("Combined All Subjects - Points Only (No Connecting Lines)")
    plt.legend(title="Category", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # B2. Separate Figures for Each Category (Points Only)
    for cat in ["AD", "MCI", "LMCI", "EMCI"]:
        plt.figure(f"{cat} - Points Only", figsize=(12, 6))
        for subj, records in subject_data.items():
            if subject_categories[subj] == cat:
                xs = [r[0] / 365.0 for r in records]
                ys = [r[2] for r in records]
                plt.scatter(xs, ys, color=category_colors[cat], alpha=0.7, label=subj)
        plt.xlabel("Years Since First Scan")
        plt.ylabel("Brain Volume (mm³)")
        plt.title(f"{cat} Subjects - Points Only (No Connecting Lines)")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(title="Subject", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
    
    # B3. Combined Multi-Panel Figure (Points Only)
    fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    fig.suptitle("Brain Volume Over Time by Category (Points Only)", fontsize=30)
    for i, cat in enumerate(cat_list):
        ax = axs[i // 2, i % 2]
        for subj, records in subject_data.items():
            if subject_categories[subj] == cat:
                xs = [r[0] / 365.0 for r in records]
                ys = [r[2] for r in records]
                ax.scatter(xs, ys, color=category_colors[cat], alpha=0.7)
        ax.set_title(cat)
        ax.set_xlabel("Years")
        ax.set_ylabel("Brain Volume (mm³)")
        ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Show all plots
    plt.show()
    
    # --- Optionally, print a summary of the computed volumes ---
    print("\nSummary of Computed Results:")
    for subj, records in subject_data.items():
        print(f"\nSubject: {subj} (Category: {subject_categories[subj]})")
        print("Days_Since_First\tDate\t\t\tBrain_Volume (mm³)")
        for (days, date, volume) in records:
            print(f"{days}\t\t\t{date.strftime('%Y-%m-%d')}\t\t{volume:.2f}")

if __name__ == "__main__":
    main()

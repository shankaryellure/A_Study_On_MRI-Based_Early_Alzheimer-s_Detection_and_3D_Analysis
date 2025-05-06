import os
import numpy as np
import nibabel as nib
from PIL import Image
from scipy.ndimage import label
import pandas as pd
import vtk
from collections import defaultdict

# Global dictionaries to store ROI offsets and bounding boxes.
roi_offsets = {}
cluster_bboxes = {}
cluster_measurements = {}  
pixdim = None              

# Step 1: Slice the 3D NIfTI Image
def process_nifti_file(input_file, output_dir, selected_slices):
    img = nib.load(input_file)
    data = img.get_fdata()

    # Normalize to 0–65535 and convert to 16-bit.
    data_min = np.min(data)
    data_max = np.max(data)
    normalized_data = 65535 * (data - data_min) / (data_max - data_min)
    normalized_data = normalized_data.astype(np.uint16)

    os.makedirs(output_dir, exist_ok=True)

    for i in selected_slices:
        slice_data = normalized_data[:, :, i]
        output_file_tiff = os.path.join(output_dir, f"slice_{i:03d}.tiff")
        output_file_png = os.path.join(output_dir, f"slice_{i:03d}.png")
        
        image = Image.fromarray(slice_data)
        image.save(output_file_tiff, format="TIFF")
        image.save(output_file_png, format="PNG")
        
        print(f"Saved slice {i} to {output_file_tiff} and {output_file_png}")
    
    return output_dir

# Step 2: Analyze Slices for Clusters & Metadata
def analyze_slices_for_clusters(slice_dir, selected_slices, output_excel_file):
    cluster_data = []
    
    for slice_no in selected_slices:
        slice_path = os.path.join(slice_dir, f"slice_{slice_no:03d}.tiff")
        if not os.path.exists(slice_path):
            continue

        image = Image.open(slice_path).convert("I")
        image_array = np.array(image)
        # Adjust thresholds as needed (example range shown).
        gray_mask = (image_array >= 23170) & (image_array <= 30840)
        labeled_array, num_clusters = label(gray_mask)

        for cluster_index in range(1, num_clusters + 1):
            cluster_mask = (labeled_array == cluster_index)
            cluster_size = cluster_mask.sum()
            if cluster_size < 10:
                continue  

            # Collect each voxel in that cluster
            y_coords, x_coords = np.where(cluster_mask)
            for x, y in zip(x_coords, y_coords):
                cluster_data.append({
                    'FileName': os.path.basename(slice_path),
                    'Slice': slice_no,   
                    'X': x,
                    'Y': y,
                    'ColorType': 'Gray',
                    'ClusterID': cluster_index,
                    'ClusterSize': cluster_size
                })
    
    df = pd.DataFrame(cluster_data)
    df.to_excel(output_excel_file, index=False)
    print(f"Saved cluster metadata to {output_excel_file}")
    
    return cluster_data

# Step 3: Extract 3D Images for Clusters (ROIs)
def extract_3d_clusters(cluster_data, slice_dir, output_dir, selected_slices, margin=10):
    os.makedirs(output_dir, exist_ok=True)
    
    # Group cluster data by (ClusterID, ColorType)
    cluster_groups = {}
    for entry in cluster_data:
        cluster_key = (entry['ClusterID'], entry['ColorType'])
        if cluster_key not in cluster_groups:
            cluster_groups[cluster_key] = []
        cluster_groups[cluster_key].append(entry)
    
    for (cluster_id, color_type), pixels in cluster_groups.items():
        all_x_coords = [entry['X'] for entry in pixels]
        all_y_coords = [entry['Y'] for entry in pixels]
        # Compute bounding box with margin.
        x_min = max(0, min(all_x_coords) - margin)
        x_max = min(255, max(all_x_coords) + margin)
        y_min = max(0, min(all_y_coords) - margin)
        y_max = min(255, max(all_y_coords) + margin)
        z_min = min([entry['Slice'] for entry in pixels])
        z_max = max([entry['Slice'] for entry in pixels])
        roi_slices = []

        for slice_no in selected_slices:
            slice_path = os.path.join(slice_dir, f"slice_{slice_no:03d}.tiff")
            if os.path.exists(slice_path):
                image = Image.open(slice_path).convert("I")
                image_array = np.array(image)
                cropped_slice = image_array[y_min:y_max, x_min:x_max]
                roi_slices.append(cropped_slice)
        
        if roi_slices:
            reconstructed_3d = np.stack(roi_slices, axis=-1)

            # Create an affine that maps the ROI into the original space.
            affine = np.array([[1, 0, 0, x_min],
                               [0, 1, 0, y_min],
                               [0, 0, 1, z_min],
                               [0, 0, 0, 1]])
            
            # Save the extracted 3D image directly without re-normalizing.
            nifti_img = nib.Nifti1Image(reconstructed_3d, affine=affine)
            output_file = os.path.join(output_dir, f"Cluster_{cluster_id}_3D.nii.gz")
            nib.save(nifti_img, output_file)
            print(f"Saved reconstructed 3D image for Cluster {cluster_id} to {output_file}")
            
            # Store bounding box info in global dictionaries.
            roi_offsets[(cluster_id, color_type)] = (x_min, y_min, z_min)
            cluster_bboxes[(cluster_id, color_type)] = (x_min, x_max, y_min, y_max, z_min, z_max)

# ---- NEW FUNCTION: Measure the 3D bounding-box dimensions and approximate volume ----
def measure_cluster_sizes(nifti_file, cluster_data, cluster_bboxes):
    """
    Measures bounding-box dimensions and approximate volumes for each cluster 
    using the original voxel size from the NIfTI header.
    """
    # Load the original NIfTI to get voxel spacing (dx, dy, dz).
    nii_img = nib.load(nifti_file)
    dx, dy, dz = nii_img.header.get_zooms()[:3]
    voxel_volume = dx * dy * dz  

    # Summation of voxel counts for each (ClusterID, ColorType).
    cluster_voxel_counts = defaultdict(int)
    for entry in cluster_data:
        cid = (entry['ClusterID'], entry['ColorType'])
        cluster_voxel_counts[cid] += 1

    measurements = []

    for (cluster_id, color_type), bbox in cluster_bboxes.items():
        (x_min, x_max, y_min, y_max, z_min, z_max) = bbox
        
        # Dimensions in voxel space.
        width_vox  = (x_max - x_min)
        height_vox = (y_max - y_min)
        depth_vox  = (z_max - z_min + 1)  

        # Convert bounding-box dimensions to mm.
        width_mm  = width_vox  * dx
        height_mm = height_vox * dy
        depth_mm  = depth_vox  * dz
        bounding_box_volume_mm3 = width_mm * height_mm * depth_mm

        # Actual voxel count from cluster_data.
        voxel_count = cluster_voxel_counts.get((cluster_id, color_type), 0)
        approximate_volume_mm3 = voxel_count * voxel_volume

        measurements.append({
            'ClusterID': cluster_id,
            'ColorType': color_type,
            'BoundingBox(mm)': (round(width_mm, 2), round(height_mm, 2), round(depth_mm, 2)),
            'BoundingBoxVolume(mm^3)': round(bounding_box_volume_mm3, 2),
            'VoxelCount': voxel_count,
            'ApproxVolume(mm^3)': round(approximate_volume_mm3, 2)
        })

    # Return both measurements and the voxel dimensions (pixdim)
    return measurements, (dx, dy, dz)


# Step 4: Visualize 3D Image with Orientation Labels and Dynamic Interactivity
def get_3d_images(directory, extensions=[".nii", ".nii.gz"]):
    return [os.path.join(directory, f) for f in os.listdir(directory)
            if any(f.lower().endswith(ext) for ext in extensions)]

def visualize_3d_image_with_labels(image_path):
    global slice_output_dir, cluster_data, cluster_measurements, pixdim  

    base_name = os.path.basename(image_path)
    try:
        cluster_id_str = base_name.split('_')[1]
        cluster_id = int(cluster_id_str)
    except Exception as e:
        print("Error extracting cluster ID:", e)
        cluster_id = None

    color_type = 'Gray'
    offset_roi = roi_offsets.get((cluster_id, color_type), (0, 0, 0))
    bbox = cluster_bboxes.get((cluster_id, color_type), (0, 0, 0, 0, 0, 0))

    # ========== STEP A: Build the 3D Volume in the Right Renderer ==========
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(image_path)
    reader.Update()

    initial_level = 50
    initial_offset = (initial_level - 50) * (65535.0 / 50.0)
    shiftScale = vtk.vtkImageShiftScale()
    shiftScale.SetInputConnection(reader.GetOutputPort())
    shiftScale.SetScale(1.0)
    shiftScale.SetShift(initial_offset)
    shiftScale.SetOutputScalarTypeToUnsignedShort()
    shiftScale.ClampOverflowOn()
    shiftScale.Update()

    volume_mapper = vtk.vtkSmartVolumeMapper()
    volume_mapper.SetInputConnection(shiftScale.GetOutputPort())
    volume_mapper.SetBlendModeToMaximumIntensity()

    volume_property = vtk.vtkVolumeProperty()
    volume_property.ShadeOff()
    volume_property.SetInterpolationTypeToLinear()

    opacity_function = vtk.vtkPiecewiseFunction()
    opacity_function.AddPoint(0, 1.0)
    opacity_function.AddPoint(65535, 1.0)
    volume_property.SetScalarOpacity(opacity_function)

    color_function = vtk.vtkColorTransferFunction()
    color_function.AddRGBPoint(0, 0.0, 0.0, 0.0)
    color_function.AddRGBPoint(65535, 1.0, 1.0, 1.0)
    volume_property.SetColor(color_function)

    volume = vtk.vtkVolume()
    volume.SetMapper(volume_mapper)
    volume.SetProperty(volume_property)

    # ========== STEP B: Create Two Renderers ==========
    left_renderer = vtk.vtkRenderer()
    left_renderer.SetViewport(0.0, 0.0, 0.3, 1.0)
    left_renderer.SetBackground(0.2, 0.2, 0.2)

    right_renderer = vtk.vtkRenderer()
    right_renderer.SetViewport(0.3, 0.0, 1.0, 1.0)
    right_renderer.SetBackground(0.0, 0.0, 0.0)
    right_renderer.AddVolume(volume)

    # Filter cluster_data for current cluster_id & color_type.
    cluster_rows = [
        row for row in cluster_data
        if int(row["ClusterID"]) == cluster_id and row["ColorType"] == color_type
    ]
    if not cluster_rows:
        print(f"No metadata for cluster {cluster_id}. Skipping visualization.")
        return

    # ========== STEP C: Left Renderer - Display the 2D Slices for This Cluster ==========
    filenames = sorted({ row["FileName"] for row in cluster_rows })
    image_actors = []
    max_width = 0
    total_height = 0
    spacing = 10  # gap in pixels between images

    filename_to_slice = {}
    for row in cluster_rows:
        filename_to_slice[row["FileName"]] = row["Slice"]

    x_min, x_max, y_min, y_max, z_min, z_max = bbox
    radius = min((x_max - x_min), (y_max - y_min)) / 2.0
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0

    circle_source = vtk.vtkRegularPolygonSource()
    circle_source.SetNumberOfSides(50)
    circle_source.SetRadius(radius)
    circle_source.SetCenter(0, 0, 0)

    circle_mapper = vtk.vtkPolyDataMapper()
    circle_mapper.SetInputConnection(circle_source.GetOutputPort())

    circle_actor_template = vtk.vtkActor()
    circle_actor_template.SetMapper(circle_mapper)
    circle_actor_template.GetProperty().SetColor(1, 1, 0)  # Yellow
    circle_actor_template.GetProperty().SetLineWidth(2.0)
    circle_actor_template.GetProperty().SetOpacity(0.8)
    circle_actor_template.GetProperty().SetRepresentationToWireframe()

    for idx, fname in enumerate(filenames):
        slice_path = os.path.join(slice_output_dir, fname)
        if not os.path.exists(slice_path):
            print(f"File not found: {slice_path}")
            continue

        ext = os.path.splitext(fname)[1].lower()
        if ext == ".png":
            img_reader = vtk.vtkPNGReader()
        else:
            img_reader = vtk.vtkTIFFReader()

        img_reader.SetFileName(slice_path)
        img_reader.Update()

        image_actor = vtk.vtkImageActor()
        image_actor.GetMapper().SetInputConnection(img_reader.GetOutputPort())
        image_actor.GetProperty().SetColorWindow(65535)
        image_actor.GetProperty().SetColorLevel(32767)

        dims = img_reader.GetOutput().GetDimensions()
        width, height = dims[0], dims[1]
        if width > max_width:
            max_width = width

        pos_y = idx * (height + spacing)
        image_actor.SetPosition(0, pos_y, 0)
        total_height = pos_y + height

        left_renderer.AddActor(image_actor)
        image_actors.append(image_actor)

        # Add yellow circle if this slice is in cluster's z-range
        this_slice = filename_to_slice.get(fname, None)
        if this_slice is not None and (z_min <= this_slice <= z_max):
            circle_actor = vtk.vtkActor()
            circle_actor.ShallowCopy(circle_actor_template)
            circle_x = cx
            circle_y = (height - 1 - cy)
            circle_y_final = pos_y + circle_y
            circle_actor.SetPosition(circle_x, circle_y_final, 0)
            left_renderer.AddActor(circle_actor)

    if image_actors:
        left_cam = left_renderer.GetActiveCamera()
        left_cam.ParallelProjectionOn()
        center_x = max_width / 2.0
        center_y = total_height / 2.0
        left_cam.SetFocalPoint(center_x, center_y, 0)
        left_cam.SetPosition(center_x, center_y, 1000)
        left_cam.SetViewUp(0, 1, 0)
        left_cam.SetParallelScale(total_height / 2.0)

    # ========== STEP D: Right Renderer - 3D Volume Setup & Overlays ==========
    right_cam = right_renderer.GetActiveCamera()
    vol_bounds = volume.GetBounds()
    center_3d = [
        (vol_bounds[0] + vol_bounds[1]) / 2.0,
        (vol_bounds[2] + vol_bounds[3]) / 2.0,
        (vol_bounds[4] + vol_bounds[5]) / 2.0
    ]
    right_cam.SetFocalPoint(center_3d)
    right_cam.SetPosition(center_3d[0], center_3d[1], center_3d[2] + 500)
    right_cam.SetViewUp(0, 1, 0)

    roi_img = nib.load(image_path)
    affine = roi_img.affine
    label_offset = 15

    def safe_norm(v):
        norm = np.linalg.norm(v)
        return v if norm == 0 else v / norm

    right_vector = safe_norm(affine[:3, 0])
    front_vector = safe_norm(affine[:3, 1])
    top_vector   = safe_norm(affine[:3, 2])
    r_bounds = volume.GetBounds()
    center3 = [
        (r_bounds[0] + r_bounds[1]) / 2.0,
        (r_bounds[2] + r_bounds[3]) / 2.0,
        (r_bounds[4] + r_bounds[5]) / 2.0
    ]
    right_pos = np.array(center3) + label_offset * right_vector * 1.5
    left_pos  = np.array(center3) - label_offset * right_vector * 1.5
    front_pos = np.array(center3) - label_offset * front_vector * 1.5
    back_pos  = np.array(center3) + label_offset * front_vector * 1.5
    top_pos   = np.array(center3) + label_offset * top_vector * 1.5
    bottom_pos= np.array(center3) - label_offset * top_vector * 1.5

    def create_3d_text(text, pos):
        text_actor = vtk.vtkBillboardTextActor3D()
        text_actor.SetPosition(pos)
        text_actor.SetInput(text)
        text_actor.GetTextProperty().SetColor(1.0, 1.0, 1.0)
        text_actor.GetTextProperty().SetFontSize(28)
        text_actor.GetTextProperty().BoldOn()
        return text_actor

    right_renderer.AddActor(create_3d_text("Front", front_pos))
    right_renderer.AddActor(create_3d_text("Back", back_pos))
    right_renderer.AddActor(create_3d_text("Left", left_pos))
    right_renderer.AddActor(create_3d_text("Right", right_pos))
    right_renderer.AddActor(create_3d_text("Top", top_pos))
    right_renderer.AddActor(create_3d_text("Bottom", bottom_pos))

    title_text_actor = vtk.vtkTextActor()
    title_text_actor.SetInput(f"Cluster {cluster_id}")
    title_text_actor.GetTextProperty().SetColor(1.0, 1.0, 1.0)
    title_text_actor.GetTextProperty().SetFontSize(28)
    title_text_actor.GetTextProperty().BoldOn()
    title_text_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    title_text_actor.SetPosition(0.5, 0.95)
    title_text_actor.GetTextProperty().SetJustificationToCentered()
    title_text_actor.GetTextProperty().SetVerticalJustificationToTop()
    right_renderer.AddActor2D(title_text_actor)

    overlay_text_actor = vtk.vtkTextActor()
    overlay_text_actor.SetInput("")
    overlay_text_actor.GetTextProperty().SetColor(1.0, 1.0, 0.0)
    overlay_text_actor.GetTextProperty().SetFontSize(28)
    overlay_text_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    overlay_text_actor.SetPosition(0.95, 0.95)
    overlay_text_actor.GetTextProperty().SetJustificationToRight()
    overlay_text_actor.GetTextProperty().SetVerticalJustificationToTop()
    right_renderer.AddActor2D(overlay_text_actor)

    cubeAxesActor = vtk.vtkCubeAxesActor()
    cubeAxesActor.SetBounds(volume.GetBounds())
    cubeAxesActor.SetCamera(right_renderer.GetActiveCamera())
    cubeAxesActor.SetXTitle("X (mm)")
    cubeAxesActor.SetYTitle("Y (mm)")
    cubeAxesActor.SetZTitle("Z (mm)")
    for axis in range(3):
        cubeAxesActor.GetTitleTextProperty(axis).SetColor(1, 1, 1)
        cubeAxesActor.GetLabelTextProperty(axis).SetColor(1, 1, 1)
    cubeAxesActor.SetFlyModeToOuterEdges()
    right_renderer.AddActor(cubeAxesActor)

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(left_renderer)
    render_window.AddRenderer(right_renderer)
    render_window.SetSize(1200, 800)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    interactor_style = vtk.vtkInteractorStyleTrackballCamera()
    interactor_style.SetMotionFactor(3.0)
    interactor.SetInteractorStyle(interactor_style)

    sliderRep = vtk.vtkSliderRepresentation2D()
    sliderRep.SetMinimumValue(0)
    sliderRep.SetMaximumValue(100)
    sliderRep.SetValue(initial_level)
    sliderRep.SetTitleText("Intensity Level")
    sliderRep.SetTitleHeight(0.02)
    sliderRep.SetLabelHeight(0.02)
    sliderRep.SetSliderLength(0.02)
    sliderRep.SetSliderWidth(0.02)
    sliderRep.SetEndCapLength(0.01)
    sliderRep.SetEndCapWidth(0.02)
    sliderRep.GetTitleProperty().SetColor(1.0, 1.0, 1.0)
    sliderRep.GetLabelProperty().SetColor(1.0, 1.0, 1.0)
    sliderRep.GetSliderProperty().SetColor(1.0, 1.0, 1.0)
    sliderRep.ShowSliderLabelOff()
    sliderRep.GetPoint1Coordinate().SetCoordinateSystemToNormalizedViewport()
    sliderRep.GetPoint1Coordinate().SetValue(-0.3, 0.12)
    sliderRep.GetPoint2Coordinate().SetCoordinateSystemToNormalizedViewport()
    sliderRep.GetPoint2Coordinate().SetValue(0.50, 0.12)
    sliderWidget = vtk.vtkSliderWidget()
    sliderWidget.SetInteractor(interactor)
    sliderWidget.SetRepresentation(sliderRep)
    sliderWidget.SetCurrentRenderer(right_renderer)
    sliderWidget.SetAnimationModeToAnimate()
    sliderWidget.EnabledOn()

    def slider_callback(obj, event):
        rep = obj.GetRepresentation()
        level = rep.GetValue()
        new_offset = (level - 50) * (65535.0 / 50.0)
        shiftScale.SetShift(new_offset)
        shiftScale.Update()
        print(f"Slider level: {level} => offset: {new_offset:.2f}")
        render_window.Render()

    sliderWidget.AddObserver("InteractionEvent", slider_callback)

    picker = vtk.vtkCellPicker()
    picker.SetTolerance(0.005)

    def click_callback(obj, event):
        click_pos = interactor.GetEventPosition()
        picker.Pick(click_pos[0], click_pos[1], 0, right_renderer)
        picked = np.array(picker.GetPickPosition())
        print("Clicked world position (ROI coords):", picked)
        vol_bounds = volume.GetBounds()
        if (picked[0] < vol_bounds[0] or picked[0] > vol_bounds[1] or
            picked[1] < vol_bounds[2] or picked[1] > vol_bounds[3] or
            picked[2] < vol_bounds[4] or picked[2] > vol_bounds[5]):
            overlay_text_actor.SetInput("")
            render_window.Render()
            return

        local_coord = (
            int(round(picked[0])),
            int(round(picked[1])),
            int(round(picked[2]))
        )
        dynamic_coord = (
            int(round(picked[0] + offset_roi[0])),
            int(round(picked[1] + offset_roi[1])),
            int(round(picked[2] + offset_roi[2]))
        )
        range_text = (
            f"Range: X: {bbox[0]}-{bbox[1]}, "
            f"Y: {bbox[2]}-{bbox[3]}, "
            f"Z: {bbox[4]}-{bbox[5]}"
        )
        info_text = (
            f"Local: {local_coord}\n"
            f"Original: {dynamic_coord}\n"
            f"{range_text}"
        )
        # Append pixdim and cluster measurement info if available.
        key = (cluster_id, color_type)
        if key in cluster_measurements:
            meas = cluster_measurements[key]
            additional_info = (
                f"\nPixDim: {pixdim}"
                f"\nBoundingBoxVol: {meas['BoundingBoxVolume(mm^3)']} mm³"
                f"\nApproxVolume: {meas['ApproxVolume(mm^3)']} mm³"
            )
            info_text += additional_info

        overlay_text_actor.SetInput(info_text)
        print("Displaying info:", info_text)
        render_window.Render()

    interactor.AddObserver("LeftButtonPressEvent", click_callback)

    render_window.Render()
    interactor.Start()


# Main Execution
if __name__ == "__main__":
    input_file = "/Users/shankaryellure/Library/CloudStorage/OneDrive-QuinnipiacUniversity/Thesis_Research_papers/ADNI/002_S_2073/Axial_T2-FLAIR/2014-09-17_14_33_48.0/I444159/Axial_T2-FLAIR_SENSE_20140917143348_701_skull_scripted.nii.gz"
    slice_output_dir = "/Users/shankaryellure/Library/CloudStorage/OneDrive-QuinnipiacUniversity/Thesis_Research_papers/ADNI/002_S_2073/Axial_T2-FLAIR/2014-09-17_14_33_48.0/I444159/sliced_images"
    metadata_output_file = "/Users/shankaryellure/Library/CloudStorage/OneDrive-QuinnipiacUniversity/Thesis_Research_papers/ADNI/002_S_2073/Axial_T2-FLAIR/2014-09-17_14_33_48.0/I444159/cluster_metadata.xlsx"
    cluster_output_dir = "/Users/shankaryellure/Library/CloudStorage/OneDrive-QuinnipiacUniversity/Thesis_Research_papers/ADNI/002_S_2073/Axial_T2-FLAIR/2014-09-17_14_33_48.0/I444159/cluster_3d_images"
    selected_slices = range(16, 25)
    
    # 1) Process & slice the NIfTI
    process_nifti_file(input_file, slice_output_dir, selected_slices)

    # 2) Analyze slices for clusters
    cluster_data = analyze_slices_for_clusters(
        slice_output_dir, selected_slices, metadata_output_file
    )

    # 3) Extract each cluster in 3D
    extract_3d_clusters(cluster_data, slice_output_dir, cluster_output_dir, selected_slices)

    # 4) Measure cluster dimensions and volumes along with pixdim.
    measurements, pixdim = measure_cluster_sizes(input_file, cluster_data, cluster_bboxes)
    
    print("\nPixel Dimensions (from header):", pixdim)
    print("\n=== Cluster Size Measurements ===")
    for m in measurements:
        print(m)
    print("=================================")
    
    # Build a lookup dictionary for cluster measurements.
    for m in measurements:
        key = (m['ClusterID'], m['ColorType'])
        cluster_measurements[key] = m

    # 5) Visualize each 3D cluster with VTK.
    for img in get_3d_images(cluster_output_dir):
        visualize_3d_image_with_labels(img)

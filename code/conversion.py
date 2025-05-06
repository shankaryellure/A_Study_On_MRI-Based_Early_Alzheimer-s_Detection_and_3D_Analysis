import os
import pydicom
import numpy as np
import nibabel as nib

# Input directory containing subdirectories with .dcm files
input_root_dir = "/Users/shankaryellure/Library/CloudStorage/OneDrive-QuinnipiacUniversity/Thesis_Research_papers/ADNI/005_S_4185/Axial_FLAIR/2011-09-12_10_38_27.0/I255623"
output_root_dir = "/Users/shankaryellure/Library/CloudStorage/OneDrive-QuinnipiacUniversity/Thesis_Research_papers/ADNI/005_S_4185/Axial_FLAIR/2011-09-12_10_38_27.0/I255623"

# Function to process a directory of .dcm files and save as a 3D NIfTI file
def convert_dicom_to_nifti(input_dir, output_file):
    # Gather all DICOM files in the directory
    dicom_files = [os.path.join(input_dir, f) for f in sorted(os.listdir(input_dir)) if f.endswith('.dcm')]
    
    if not dicom_files:
        print(f"No DICOM files found in {input_dir}")
        return
    
    # Load DICOM slices and stack them into a 3D array
    slices = [pydicom.dcmread(f) for f in dicom_files]
    slices.sort(key=lambda x: int(x.InstanceNumber))  # Sort by InstanceNumber
    pixel_arrays = [s.pixel_array for s in slices]
    image_3d = np.stack(pixel_arrays, axis=-1)  # Create 3D array along the depth axis

    # Normalize the 3D array to [0, 1] for saving as a NIfTI file
    image_3d_normalized = (image_3d - np.min(image_3d)) / (np.max(image_3d) - np.min(image_3d))

    # Create a NIfTI image
    nifti_img = nib.Nifti1Image(image_3d_normalized, affine=np.eye(4))

    # Save the NIfTI file
    nib.save(nifti_img, output_file)
    print(f"Saved NIfTI file: {output_file}")

# Walk through each subdirectory in the input root directory
for root, dirs, files in os.walk(input_root_dir):
    for subdir in dirs:
        input_dir = os.path.join(root, subdir)
        output_file = os.path.join(output_root_dir, subdir, f"{subdir}_3d.nii.gz")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Convert DICOM to NIfTI
        convert_dicom_to_nifti(input_dir, output_file)

print("Conversion to 3D NIfTI files complete.")

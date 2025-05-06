import os
import subprocess

# Define the root directory containing subdirectories with .nii.gz files (macOS path style)
root_dir = "/Users/shankaryellure/Downloads/CN"

# Set your BET parameters (adjust -f and -g as needed)
bet_fraction = "0.5"  # fractional intensity threshold
bet_g = "0"           # vertical gradient (optional)

# Walk through the directory recursively
for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename.endswith(".nii.gz"):
            input_path = os.path.join(dirpath, filename)
            # Create the output file name with _skull_scripted added before the extension
            base_name = filename[:-7]  # removes the ".nii.gz" part
            output_filename = f"{base_name}_skull_scripted.nii.gz"
            output_path = os.path.join(dirpath, output_filename)
            
            # Construct the BET command (FSL's BET is typically installed and available on macOS)
            cmd = ["bet", input_path, output_path, "-f", bet_fraction, "-g", bet_g]
            print(f"Processing: {input_path}")
            try:
                # Run the skull stripping command
                subprocess.run(cmd, check=True)
                print(f"Saved skull stripped image as: {output_path}\n")
            except subprocess.CalledProcessError as e:
                print(f"Error processing {input_path}: {e}")

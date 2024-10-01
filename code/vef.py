import os
from pathlib import Path

# Path to the input directory containing images
input_dir = 'nums/'
start_number = 50  # Starting number for renaming
file_extension = '.jpg'  # Assuming all images are .png

# Get all image paths in the directory
image_paths = list(Path(input_dir).glob(f'*{file_extension}'))

# Loop through all images and rename them
for idx, img_path in enumerate(sorted(image_paths)):
    # New file name with zero padding (e.g., 000400, 000401, etc.)
    new_file_name = f"{str(start_number + idx).zfill(6)}{file_extension}"

    # Full path for the new file
    new_file_path = os.path.join(input_dir, new_file_name)

    # Rename the file
    os.rename(img_path, new_file_path)

    print(f"Renamed: {img_path.name} -> {new_file_name}")

print("Renaming complete!")

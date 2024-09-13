import os

# Define the folder containing the images
folder_path = './'

# Get a list of all .png files in the folder
images = [f for f in os.listdir(folder_path) if f.endswith('.png')]

# Sort the list of images (if needed)
images.sort()

# Rename the images
for i, image in enumerate(images, start=39):
    # Format the new name with 6 digits and .png extension
    new_name = f"{i:06}.png"
    
    # Get the full path of the current and new names
    old_path = os.path.join(folder_path, image)
    new_path = os.path.join(folder_path, new_name)
    
    # Rename the file
    os.rename(old_path, new_path)

print("Renaming completed!")

import os
import csv

# Define the folder containing the images
folder_path = './'

# Define the name of the CSV file
csv_file = 'valid.csv'

# Get a list of all image files in the folder (change if you have other formats)
images = [f for f in os.listdir(folder_path) if f.endswith('.png')]

# Create and write to the CSV file
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Write the header
    writer.writerow(['file_name', 'text'])
    
    # Write image names and default text "0"
    for image in images:
        writer.writerow([image, '0'])

print(f"CSV file '{csv_file}' created successfully!")

import pandas as pd
import shutil
import os

# Load the CSV file
csv_file = 'train.csv'
df = pd.read_csv(csv_file)

# Define source and destination directories
source_dir = 'fr/data'
destination_dir = 'fr/train'

# Create the destination directory if it does not exist
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# Iterate over the rows of the dataframe and move the files
for index, row in df.iterrows():
    file_name = row['file_name']
    src_path = os.path.join(source_dir, file_name)
    dst_path = os.path.join(destination_dir, file_name)

    # Check if the file exists before moving
    if os.path.exists(src_path):
        shutil.move(src_path, dst_path)
        print(f'Moved {file_name} to {destination_dir}')
    else:
        print(f'{file_name} does not exist in the source directory')

print('File moving complete.')

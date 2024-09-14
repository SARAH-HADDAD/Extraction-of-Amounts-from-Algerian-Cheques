import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os

# Load the CSV file
csv_file = './fr/train/train.csv'  # Update with the actual CSV file path
df = pd.read_csv(csv_file)

# Folder containing images
image_folder = './fr/train'  # Update with the actual folder path where images are stored

# Iterate through each row of the CSV
for index, row in df.iterrows():
    image_path = os.path.join(image_folder, row['file_name'])
    
    # Open and display the image
    img = Image.open(image_path)
    plt.imshow(img)
    plt.axis('off')  # Hide axes for better view
    
    # Set the title with the file name and corresponding text
    plt.title(f"File: {row['file_name']}\nText: {row['text']}", fontsize=10)
    
    # Show the figure
    plt.show()

    

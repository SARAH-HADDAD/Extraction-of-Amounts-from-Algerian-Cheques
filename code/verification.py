import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os


csv_file = 'data/fr/valid/valid.csv'  
df = pd.read_csv(csv_file)


image_folder = 'data/fr/valid'  

for index, row in df.iterrows():
    image_path = os.path.join(image_folder, row['file_name'])
    img = Image.open(image_path)
    plt.imshow(img)
    plt.axis('off')  
    plt.title(f"File: {row['file_name']}\nText: {row['text']}", fontsize=10)
    plt.show()
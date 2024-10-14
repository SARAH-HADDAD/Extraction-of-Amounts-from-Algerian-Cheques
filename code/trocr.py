import os
from tqdm import tqdm  # Changed from tqdm.notebook to standard tqdm
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from transformers import VisionEncoderDecoderModel, TrOCRProcessor

def ocr(image, processor, model):
    pixel_values = processor(image, return_tensors='pt').pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

def eval_new_data_with_accuracy(
    data_path,
    csv_path,
    processor,
    model,
    num_samples=100
):
    # Load the CSV file with ground truth labels
    df = pd.read_csv(csv_path)
    df['file_name'] = df['file_name'].apply(lambda x: os.path.join(data_path, x))
    
    correct_predictions = 0
    total_predictions = 0
    
    # Using a context manager for tqdm
    with tqdm(total=min(len(df), num_samples)) as pbar:
        for i, row in df.iterrows():
            if i == num_samples:
                break
            
            image_path = row['file_name']
            ground_truth = row['text']
            
            image = Image.open(image_path).convert('RGB')
            predicted_text = ocr(image, processor, model)
            
            # Compare prediction with ground truth
            if predicted_text.strip().lower() == ground_truth.strip().lower():
                correct_predictions += 1
            total_predictions += 1
            
            
            pbar.update(1)
    
    accuracy = correct_predictions / total_predictions
    print(f"Model Accuracy: {accuracy:.2%}")

# Usage
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-stage1')
model = VisionEncoderDecoderModel.from_pretrained("/Users/sarahhaddad/Documents/GitHub/TrOCR/models/idk")

eval_new_data_with_accuracy(
    data_path="/Users/sarahhaddad/Documents/GitHub/TrOCR/data/fr/valid",
    csv_path="/Users/sarahhaddad/Documents/GitHub/TrOCR/data/fr/valid/valid.csv",
    processor=processor,
    model=model,
    num_samples=100
)
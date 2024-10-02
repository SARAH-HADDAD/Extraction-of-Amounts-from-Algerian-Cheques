# Import necessary libraries
from transformers import VisionEncoderDecoderModel, TrOCRProcessor, BertTokenizer, BertModel
from ultralytics import YOLO
import cv2
import torch
import numpy as np
from PIL import Image 
import certifi
import pymongo
import requests
from pathlib import Path
import os
import re
from datetime import datetime
from word2numberi18n import w2n

# Image preprocessing functions
def estimate_noise(image):
    return np.std(image)

# Estime la luminosité d'une image en calculant la moyenne des niveaux de gris
def estimate_brightness(image):
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    return np.mean(gray_image)

# Applique CLAHE (Contraste adaptatif par histogramme) pour améliorer le contraste d'une image en niveaux de gris.
def apply_clahe(gray_image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Args:
        gray_image (numpy array): Image en niveaux de gris.
        clip_limit (float): Limite du contraste pour éviter une sur-amplification.
        tile_grid_size (tuple): Taille de la grille pour la partition de l'image.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray_image)

# Ajuste la correction gamma pour améliorer la visibilité de l'image.
def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# Corrige l'inclinaison (skew) d'une image de manière adaptative en détectant les lignes via la transformée de Hough.
def adaptive_skew_correction(image, max_angle=2):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    if lines is not None:
        angles = []
        for line in lines[:10]:
            rho, theta = line[0]
            angle = (theta * 180 / np.pi) - 90
            angles.append(angle)

        median_angle = np.median(angles)

        if 0 < abs(median_angle) <= max_angle:
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            corrected_image = cv2.warpAffine(image, M, (w, h),
                                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return corrected_image, median_angle

    return image, 0

def preprocess_cheque_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")

    # Step 1: Noise Reduction (Bilateral Filter)
    noise_level = estimate_noise(image)
    sigmaColor = 50 + (noise_level / 10)
    sigmaSpace = sigmaColor
    denoised_image = cv2.bilateralFilter(image, d=9, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)

    # Step 2: Convert to Grayscale
    gray_image = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2GRAY)

    # Step 3: Apply CLAHE for contrast enhancement
    clahe_image = apply_clahe(gray_image)

    # Step 4: Adjust gamma for better visibility
    brightness = estimate_brightness(clahe_image)
    gamma = 1.0 if brightness > 127 else 1.2
    gamma_corrected = adjust_gamma(clahe_image, gamma)

    # Step 5: Selective Skew Correction
    corrected_image, angle = adaptive_skew_correction(gamma_corrected, max_angle=2)

    return corrected_image

# YOLO model functions

def run_yolo_model(image_path):
    image = preprocess_cheque_image(image_path)
    save_dir='./cropped_images'
    model = YOLO('/Users/sarahhaddad/Documents/GitHub/TrOCR/models/model_yolo/best.pt').to('cpu')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(image)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    img_with_boxes = np.copy(image)
    for r in results:
        for i, box in enumerate(r.boxes):
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            x1, y1, x2, y2 = box.xyxy.cpu().numpy().astype(np.int32)[0]
            cropped_img = image[y1:y2, x1:x2]
            cropped_img_path = save_dir / f"{label}.png"
            cv2.imwrite(str(cropped_img_path), cropped_img)
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_with_boxes, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Text extraction functions
def extract_text_and_draw_boxes(image_path):
    access_token= 'i9cz3vVyX7deJcDOExAfMJYd1drG3Vdo7pCtEzs12R2Kh8EO32usKdT3hLzoV8Km'
    url = f"https://backend.scandocflow.com/v1/api/documents/extract?access_token={access_token}"

    with open(image_path, 'rb') as image_file:
        files = {'files': image_file}
        payload = {
            "type": "ocr",
            "lang": "ara",
            "retain": "false",
        }
        response = requests.post(url, data=payload, files=files)
        response_data = response.json()

    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    if response_data.get('documents'):
        words = response_data['documents'][0]['textAnnotation']['Pages'][0]['Words']
        arabic_texts = []
        bounding_boxes = []

        for word in words:
            if word.get('Lang') == 'ara':
                text = word['Text']
                outline = word['Outline']
                x1, y1 = int(outline[0] * w), int(outline[1] * h)
                x2, y2 = int(outline[4] * w), int(outline[5] * h)

                if not any((x1 >= box[0] and y1 >= box[1] and x2 <= box[2] and y2 <= box[3]) for box in bounding_boxes):
                    arabic_texts.append((text, x1))
                    bounding_boxes.append((x1, y1, x2, y2))

        arabic_texts = sorted(arabic_texts, key=lambda x: x[1], reverse=True)

        for (x1, y1, x2, y2), text in zip(bounding_boxes, arabic_texts):
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, text[0], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return [text for text, _ in arabic_texts]
    else:
        print("No documents found in the response.")
        return []


def ocr(image, model_path):
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-stage1')
    model = VisionEncoderDecoderModel.from_pretrained(model_path).to("cpu")
    pixel_values = processor(image, return_tensors='pt').pixel_values.to('cpu')
    generated_ids = model.generate(pixel_values, output_scores=True, return_dict_in_generate=True)
    generated_text = processor.batch_decode(generated_ids.sequences, skip_special_tokens=True)[0]
    scores = generated_ids.scores
    confidence = torch.mean(torch.stack([torch.max(torch.softmax(score, dim=-1)) for score in scores])).item()
    return generated_text, confidence

def extract_text(image_path, trained_model):
    try:
        image = Image.open(image_path).convert('RGB')
        text, confidence = ocr(image, trained_model)
        return text
    except FileNotFoundError:
        print(f"File {image_path} not found.")
        return ''

# Number correction functions
def correct_number_format(number_text):
    clean_text = re.sub(r'[^\d,\.]', '', number_text)
    clean_text = re.sub(r'[,.](?=.*[,\.])', '', clean_text)
    if '.' in clean_text or ',' in clean_text:
        clean_text = clean_text.replace(',', '.')
        if clean_text.endswith('.00'):
            clean_text = clean_text[:-3]
    try:
        corrected_number = str(int(clean_text))
    except ValueError:
        corrected_number = "Invalid number format"
    return corrected_number

def extract_and_correct_number(image_path):
    number_text = extract_text(image_path, '/Users/sarahhaddad/Documents/GitHub/TrOCR/models/model_num')
    print(f"Le montant en chiffres extrait à partir de l'image : {number_text}")
    corrected_number = correct_number_format(number_text)
    return corrected_number

class WordToNumber:
    def __init__(self):
        self.word_to_num = {
            'صفر': 0, 'واحد': 1, 'اثنان': 2, 'ثلاثة': 3, 'أربعة': 4, 'خمسة': 5,
            'ستة': 6, 'سبعة': 7, 'ثمانية': 8, 'تسعة': 9, 'عشرة': 10,
            'أحد عشر': 11, 'إثنا عشر': 12, 'ثلاثة عشر': 13, 'أربعة عشر': 14,
            'خمسة عشر': 15, 'ستة عشر': 16, 'سبعة عشر': 17, 'ثمانية عشر': 18,
            'تسعة عشر': 19, 'عشرون': 20, 'ثلاثون': 30, 'أربعون': 40,
            'خمسون': 50, 'ستون': 60, 'سبعون': 70, 'ثمانون': 80, 'تسعون': 90,
            'مائة': 100, 'مائتان': 200, 'ثلاثمائة': 300, 'اربعمائة': 400,
            'خمسمائة': 500, 'ستمائة': 600, 'سبعمائة': 700, 'ثمانيمائة': 800,
            'تسعمائة': 900, 'الف': 1000, 'الفان': 2000, 'مليون': 1000000,
            'مليار': 1000000000
        }
        self.multipliers = {'الف': 1000, 'مليون': 1000000, 'مليار': 1000000000}
        self.currencies = {'دينار'}

    # Calcul de la distance de Levenshtein pour l'autocorrection
    def levenshtein_distance(self, s1, s2):
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    # Autocorrection du mot en utilisant la distance de Levenshtein
    def autocorrect(self, word):
        if word in self.word_to_num or word in self.multipliers or word in self.currencies:
            return word
        closest_word = min(list(self.word_to_num.keys()) + list(self.multipliers.keys()) + list(self.currencies),
                           key=lambda x: self.levenshtein_distance(word, x))
        return closest_word

    # Convertir les mots arabes en nombre
    def convert(self, arabic_words):
        words = re.findall(r'\w+|[^\w\s]', arabic_words)
        total = 0
        current_num = 0
        currency = None
        for word in words:
            if word in ['و', 'أ']: # Ignorer les mots de liaison
                continue
            corrected_word = self.autocorrect(word) # Corriger les erreurs dans le mot
            print(corrected_word)
            if corrected_word in self.currencies:
                currency = corrected_word # Gérer la détection de la monnaie
            elif corrected_word in self.multipliers:
                if current_num == 0:
                    current_num = 1
                total += current_num * self.multipliers[corrected_word] # Calculer la valeur du multiplicateur
                current_num = 0
            elif corrected_word in self.word_to_num:
                current_num += self.word_to_num[corrected_word]
            else:
                continue
        total += current_num # Ajouter le dernier nombre traité
        return total

class FrenchAmountConverter:
    def __init__(self):
        self.verifier = FrenchAmountVerifier()
        # Set the language to French for word2number-i18n
        os.environ['w2n.lang'] = 'fr'
        self.w2n_instance = w2n.W2N(lang_param="fr")

    def convert_to_number(self, amount):
        # First, verify and correct the amount
        verified_amount = self.verifier.verify_amount(amount)

        print(verified_amount)

        # Remove the currency word if present
        amount_words = verified_amount.split()
        if amount_words[-1] in ["dinar", "dinars"]:
            amount_words = amount_words[:-1]

        # Join the words back together
        number_words = " ".join(amount_words)

        try:
            # Convert to number using word2number-i18n
            numeric_amount = self.w2n_instance.word_to_num(number_words)
            return numeric_amount
        except ValueError as e:
            print(f"Error converting to number: {e}")
            return None

class FrenchAmountVerifier:
    def __init__(self):
        self.number_words = {
            0: "zéro", 1: "un", 2: "deux", 3: "trois", 4: "quatre", 5: "cinq",
            6: "six", 7: "sept", 8: "huit", 9: "neuf", 10: "dix",
            11: "onze", 12: "douze", 13: "treize", 14: "quatorze", 15: "quinze",
            16: "seize", 20: "vingt", 30: "trente", 40: "quarante", 50: "cinquante",
            60: "soixante", 100: "cent", 1000: "mille", 1000000: "million"
        }
        self.ignore_words = {"et", "de"}
        self.currency_words = {"dinar", "dinars"}

    def levenshtein_distance(self, s1, s2):
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    def find_closest_word(self, word, word_list):
        return min(word_list, key=lambda x: self.levenshtein_distance(word, x))

    def verify_amount(self, amount):
        words = re.findall(r'\b[\w-]+\b', amount.lower())
        corrected_words = []
        for word in words:
            if '-' in word:
                sub_words = word.split('-')
                corrected_sub_words = []
                for sub_word in sub_words:
                    corrected_sub_words.append(self.process_word(sub_word))
                corrected_words.append('-'.join(corrected_sub_words))
            else:
                corrected_words.append(self.process_word(word))
        return " ".join(corrected_words)

    def process_word(self, word):
        if word in self.ignore_words:
            return word
        elif word in self.number_words.values():
            return word
        elif word in self.currency_words:
            return word
        else:
            # Check if it's a misspelled currency word
            closest_currency = self.find_closest_word(word, self.currency_words)
            if self.levenshtein_distance(word, closest_currency) <= 2:  # Arbitrary threshold
                return closest_currency
            else:
                # If not close to a currency word, treat as a number word
                return self.find_closest_word(word, self.number_words.values())
            

# Embedding and similarity calculation
def get_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

def calculate_similarity(text1, text2, tokenizer, model):
    embedding1 = get_embedding(text1, tokenizer, model)
    embedding2 = get_embedding(text2, tokenizer, model)
    cosine_similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2)
    return int(cosine_similarity.item() * 100)

# Main function to process the cheque
def process_cheque(image_path):
    
    run_yolo_model(image_path)
    
    # Extract text
    arabic_texts1 = extract_text_and_draw_boxes("./cropped_images/line1.png")
    arabic_texts= arabic_texts1
    if arabic_texts1:
        if os.path.isfile("./cropped_images/line2.png"):
            arabic_texts2 = extract_text_and_draw_boxes("./cropped_images/line2.png")
            arabic_texts = arabic_texts + arabic_texts2
    
    # Process text based on language
    if arabic_texts:
        converter = WordToNumber()
        concatenated_texts = ' '.join(arabic_texts)
        montant = converter.convert(concatenated_texts)
        language = "arabe"
    else:
        converter = FrenchAmountConverter()
        text_fr1 = extract_text('./cropped_images/line1.png', '/Users/sarahhaddad/Documents/GitHub/TrOCR/models/model_fr')
        text_fr2 = extract_text('./cropped_images/line2.png', '/Users/sarahhaddad/Documents/GitHub/TrOCR/models/model_fr')
        concatenated_texts = f"{text_fr1} {text_fr2}".strip()
        montant = converter.convert_to_number(concatenated_texts)
        language = "français"
    
    # Extract and correct number
    chiffre = extract_and_correct_number('./cropped_images/numbers.png')
    
    # Calculate similarity
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    similarity_percentage = calculate_similarity(str(chiffre), str(montant), tokenizer, model)
    
    
    
    return {
        "montant_en_chiffres": chiffre,
        "montant_en_lettres": montant,
        "langue": language,
        "similarite": similarity_percentage,
    }


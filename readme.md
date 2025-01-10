## **`extraction.py`**  
This file focuses on **extracting information from cheques**. Its key features include:  
- **YOLO**: Used to identify areas of interest such as numeric and written amounts.  
- **TrOCR**: A Hugging Face OCR model used to recognize text from detected regions.  
- **Preprocessing functions**:  
  - Estimation of **noise** and **brightness** to enhance image quality before detection.  
- **word2numberi18n**: Converts and corrects written amounts (e.g., "cent") into numeric values.

---

## **`GUI.py`**  
This file implements the **Graphical User Interface (GUI)** using **PyQt6**. Key functionalities include:  
- **PyQt6 Widgets**: Buttons, labels, and dialog boxes for user interaction.  
- **Image Display**: Enables visualization of cheques and recognition results.  
- **MongoDB Integration**: Stores and retrieves data in a MongoDB database.  
- **User Actions**: Users can **upload cheques**, view results, and validate them.

This module provides the main interface for a **seamless user interaction with the system**.

---

## **`cheque_classifier.py`**  
This file defines a **CNN model** (Convolutional Neural Network) for **cheque classification**. Its main features are:  
- **Bank Identification**: The model detects which bank issued the cheque (e.g., BDL, BNA, CPA).  
- **CNN Model**: Uses convolutional layers to classify cheques by their origin.  

This file is used for **organizing and classifying cheques by their issuing bank**.

---

## **Project Workflow Summary**  
1. **Cheque Upload (GUI)**: The user uploads a cheque image through the interface.  
2. **Detection and Extraction (`extraction.py`)**:  
   - YOLO detects areas containing numeric and written amounts.  
   - TrOCR recognizes text in these areas.  
   - Written text is converted into numeric values.  
3. **Classification (`cheque_classifier.py`)**:  
   - The CNN model identifies the issuing bank.  
4. **Error Verification & Correction**:  
   - The system ensures that the numeric amount matches the written amount.  
5. **Result Display (GUI)**: The interface presents the extracted and classified information to the user.

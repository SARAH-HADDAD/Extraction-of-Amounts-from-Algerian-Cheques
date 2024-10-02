import sys
import certifi
import pymongo
from datetime import datetime
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QFrame, QGridLayout, QMessageBox)
from PyQt6.QtGui import QPixmap, QFont
from PyQt6.QtCore import Qt
from PIL import Image
from cheque_classifier import classify_cheque
from extraction import process_cheque

class ChequeProcessor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Processeur de Chèques")
        self.setGeometry(100, 100, 400, 300)
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #f0fff0;
                color: #006400;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 20px;
                font-size: 14px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.create_widgets()

    def create_widgets(self):
        # Title
        title = QLabel("Processeur de Chèques")
        title.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(title)

        # Upload Button
        upload_btn = QPushButton("Télécharger un chèque")
        upload_btn.clicked.connect(self.upload_file)
        self.layout.addWidget(upload_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        # Upload Area
        self.upload_frame = QLabel("Aucun fichier sélectionné")
        self.upload_frame.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.upload_frame.setStyleSheet("""
            background-color: white;
            border: 2px dashed #4CAF50;
            border-radius: 5px;
            padding: 20px;
        """)
        self.layout.addWidget(self.upload_frame)

    def upload_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Sélectionnez un fichier", "", "Image files (*.png *.jpg *.jpeg *.gif)")
        if file_path:
            self.process_cheque(file_path)

    def process_cheque(self, file_path):
        try:
            pixmap = QPixmap(file_path)
            pixmap = pixmap.scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio)
            self.upload_frame.setPixmap(pixmap)
            
            bank = classify_cheque(file_path)
            result = process_cheque(file_path)
            
            # Insert to MongoDB if similarity is high
            insert_result = self.insert_to_mongodb(result["montant_en_chiffres"], result["similarite"], bank)
            
            # Show result window
            self.show_result_window(file_path, bank, result, insert_result)
        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Une erreur est survenue: {str(e)}")

    def insert_to_mongodb(self, montant, similarity_percentage, bank):
        if similarity_percentage >= 99:
            try:
                ca = certifi.where()
                client = pymongo.MongoClient(
                    "mongodb+srv://stokage15:12345@cluster0.o33im.mongodb.net/xyzdb?retryWrites=true&w=majority",
                    tlsCAFile=ca
                )
                db = client['myFirstDatabase']
                collection = db['transactions']

                document = {
                    "date": datetime.now(),
                    "banque_emetrice": bank,
                    "banque_debitrice": "CPA",
                    "montant": montant
                }
                
                insert_doc = collection.insert_one(document)
                return insert_doc
            except Exception as e:
                print(f"Error inserting to MongoDB: {str(e)}")
                return None
        return None

    def show_result_window(self, file_path, bank, result, insert_result):
        self.result_window = QWidget()
        self.result_window.setWindowTitle("Résultats de l'Analyse")
        self.result_window.setGeometry(200, 200, 600, 400)
        self.result_window.setStyleSheet("""
            QWidget {
                background-color: #f0fff0;
                color: #006400;
            }
            QLabel {
                font-size: 14px;
            }
            QFrame {
                background-color: white;
                border: 1px solid #4CAF50;
                border-radius: 5px;
            }
        """)

        layout = QHBoxLayout(self.result_window)

        # Image frame (left)
        img_frame = QFrame()
        img_layout = QVBoxLayout(img_frame)
        layout.addWidget(img_frame)

        # Load and display image
        pixmap = QPixmap(file_path)
        pixmap = pixmap.scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio)
        img_label = QLabel()
        img_label.setPixmap(pixmap)
        img_layout.addWidget(img_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # Info frame (right)
        info_frame = QFrame()
        info_layout = QGridLayout(info_frame)
        layout.addWidget(info_frame)

        # Title
        title = QLabel("Détails du Chèque")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title.setStyleSheet("color: #4CAF50;")
        info_layout.addWidget(title, 0, 0, 1, 2)

        # Cheque information
        info_data = [
            ("Banque Émettrice", bank),
            ("Montant en lettres", result["montant_en_lettres"]),
            ("Montant en chiffres", str(result["montant_en_chiffres"])),
            ("Langue", result["langue"]),
        ]

        for i, (label, value) in enumerate(info_data, start=1):
            label_widget = QLabel(f"{label}:")
            label_widget.setFont(QFont("Arial", 12, QFont.Weight.Bold))
            value_widget = QLabel(str(value))
            value_widget.setFont(QFont("Arial", 12))
            info_layout.addWidget(label_widget, i, 0)
            info_layout.addWidget(value_widget, i, 1)

        # Close button
        close_btn = QPushButton("Fermer")
        close_btn.clicked.connect(self.result_window.close)
        info_layout.addWidget(close_btn, len(info_data) + 1, 0, 1, 2)

        # Show notification
        self.show_notification(result['similarite'])

        self.result_window.show()

    def show_notification(self, similarity):
        if similarity >= 99:
            QMessageBox.information(self, "Succès","Traitement réussi avec succès!")
        else:
            QMessageBox.warning(self, "Attention", f"Le traitement n'a pas réussi. La similarité est de {similarity:.2f}%")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChequeProcessor()
    window.show()
    sys.exit(app.exec())
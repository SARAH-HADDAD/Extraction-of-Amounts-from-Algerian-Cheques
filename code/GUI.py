import sys
import certifi
import pymongo
import cv2
from datetime import datetime
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QFrame, QGridLayout, QMessageBox,
                             QTableWidget, QTableWidgetItem, QSplitter, QTabWidget)
from PyQt6.QtGui import QPixmap, QFont, QImage, QIcon
from PyQt6.QtCore import Qt, QTimer
from PIL import Image
from bson.decimal128 import Decimal128
from decimal import Decimal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from cheque_classifier import classify_cheque
from extraction import process_cheque

class ChequeProcessor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Processeur de Chèques")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #f5f5f5;
                color: #333;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 20px;
                font-size: 14px;
                border-radius: 5px;
                min-width: 150px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QTableWidget {
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 5px;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QLabel {
                font-size: 14px;
            }
            QTabWidget::pane {
                border: 1px solid #ddd;
                border-radius: 5px;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #e0e0e0;
                padding: 10px 20px;
                margin-right: 2px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom: 2px solid #4CAF50;
            }
        """)

        self.camera = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        # Create main widget and layout
        main_widget = QWidget(self)
        self.main_layout = QVBoxLayout(main_widget)
        self.setCentralWidget(main_widget)

        self.create_header()
        self.create_main_content()
        self.create_footer()

        # Set up a timer to update the balance every 300 seconds
        self.balance_timer = QTimer(self)
        self.balance_timer.timeout.connect(self.update_balance)
        self.balance_timer.start(300000)  # 300 seconds

        # Update balance immediately on startup
        QTimer.singleShot(0, self.update_balance)

    def create_header(self):
        header = QWidget()
        header_layout = QHBoxLayout(header)

        logo = QLabel("Processeur de Chèques")
        logo.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        header_layout.addWidget(logo)

        self.balance_label = QLabel("Solde: Chargement...")
        self.balance_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.balance_label.setFont(QFont("Arial", 16))
        header_layout.addWidget(self.balance_label)

        self.main_layout.addWidget(header)

    def create_main_content(self):
        main_content = QSplitter(Qt.Orientation.Horizontal)
        
        left_panel = self.create_left_panel()
        right_panel = self.create_right_panel()

        main_content.addWidget(left_panel)
        main_content.addWidget(right_panel)
        main_content.setSizes([400, 800])

        self.main_layout.addWidget(main_content)

    def create_footer(self):
        footer = QWidget()
        footer_layout = QHBoxLayout(footer)

        refresh_btn = QPushButton("Rafraîchir les transactions")
        refresh_btn.setIcon(QIcon.fromTheme("view-refresh"))
        refresh_btn.clicked.connect(self.refresh_transactions)
        footer_layout.addWidget(refresh_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        self.main_layout.addWidget(footer)
        
    def update_balance(self):
        try:
            emettrice = "CCP"
            debitrice = "CPA"
            balance = self.calculate_bank_balance(emettrice, debitrice)
            if balance > 0:
                result = f"Solde: {debitrice} doit {abs(balance):.2f} DA à {emettrice}"
            elif balance < 0:
                result = f"Solde: {emettrice} doit {abs(balance):.2f} DA à {debitrice}"
            else:
                result = f"Solde: 0.00 DA entre {emettrice} et {debitrice}"
            self.balance_label.setText(result)
        except Exception as e:
            self.balance_label.setText("Erreur lors du chargement du solde")
            print(f"Error updating balance: {str(e)}")



    def create_left_panel(self):
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Upload Area
        self.upload_frame = QLabel("Aucun fichier sélectionné")
        self.upload_frame.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.upload_frame.setStyleSheet("""
            background-color: white;
            border: 2px dashed #4CAF50;
            border-radius: 5px;
            padding: 20px;
            min-height: 300px;
        """)
        left_layout.addWidget(self.upload_frame)

        # Buttons
        buttons_layout = QHBoxLayout()
        
        upload_btn = QPushButton("Télécharger un chèque")
        upload_btn.setIcon(QIcon.fromTheme("document-open"))
        upload_btn.clicked.connect(self.upload_file)
        buttons_layout.addWidget(upload_btn)

        self.camera_btn = QPushButton("Activer la caméra")
        self.camera_btn.setIcon(QIcon.fromTheme("camera-photo"))
        self.camera_btn.clicked.connect(self.toggle_camera)
        buttons_layout.addWidget(self.camera_btn)

        self.capture_btn = QPushButton("Capturer l'image")
        self.capture_btn.setIcon(QIcon.fromTheme("camera-photo"))
        self.capture_btn.clicked.connect(self.capture_image)
        self.capture_btn.setVisible(False)
        buttons_layout.addWidget(self.capture_btn)

        left_layout.addLayout(buttons_layout)

        return left_panel

    def create_right_panel(self):
        right_panel = QTabWidget()

        # Transactions Tab
        transactions_tab = QWidget()
        transactions_layout = QVBoxLayout(transactions_tab)

        self.transactions_table = QTableWidget()
        self.transactions_table.setColumnCount(5)
        self.transactions_table.setHorizontalHeaderLabels(["Date", "Banque Émettrice", "Banque Débitrice", "Montant", "ID"])
        self.transactions_table.horizontalHeader().setStretchLastSection(True)
        transactions_layout.addWidget(self.transactions_table)

        refresh_btn = QPushButton("Rafraîchir les transactions")
        refresh_btn.setIcon(QIcon.fromTheme("view-refresh"))
        refresh_btn.clicked.connect(self.refresh_transactions)
        transactions_layout.addWidget(refresh_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        right_panel.addTab(transactions_tab, "Transactions")

        # Graph Tab
        graph_tab = QWidget()
        graph_layout = QVBoxLayout(graph_tab)

        self.transaction_history_graph = FigureCanvas(Figure(figsize=(5, 4)))
        graph_layout.addWidget(self.transaction_history_graph)

        right_panel.addTab(graph_tab, "Graphique")

        return right_panel


    def create_transaction_graph(self):
        self.transaction_history_graph.figure.clear()
        ax = self.transaction_history_graph.figure.add_subplot(111)
        ax.set_title('Historique des Transactions')
        ax.set_xlabel('Date')
        ax.set_ylabel('Montant')
        ax.grid(True, linestyle='--', alpha=0.7)
        self.transaction_history_graph.draw()

    def update_transaction_graph(self, dates, amounts):
        ax = self.transaction_history_graph.figure.gca()
        ax.clear()
        ax.plot(dates, amounts, marker='o', linestyle='-', color='#4CAF50')
        ax.set_title('Historique des Transactions')
        ax.set_xlabel('Date')
        ax.set_ylabel('Montant')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, linestyle='--', alpha=0.7)

        # Format y-axis labels as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:,.2f} DA"))

        self.transaction_history_graph.figure.tight_layout()
        self.transaction_history_graph.draw()

    def refresh_transactions(self):
        try:
            ca = certifi.where()
            client = pymongo.MongoClient(
                "mongodb+srv://stokage15:12345@cluster0.o33im.mongodb.net/xyzdb?retryWrites=true&w=majority",
                tlsCAFile=ca
            )
            db = client['myFirstDatabase']
            collection = db['transactions']

            # Fetch the most recent transactions (limit to 10 for example)
            recent_transactions = list(collection.find().sort("date", -1).limit(10))

            self.transactions_table.setRowCount(0)
            dates = []
            amounts = []
            for transaction in recent_transactions:
                row_position = self.transactions_table.rowCount()
                self.transactions_table.insertRow(row_position)
                transaction_date = transaction['date']
                self.transactions_table.setItem(row_position, 0, QTableWidgetItem(transaction_date.strftime("%Y-%m-%d %H:%M")))
                self.transactions_table.setItem(row_position, 1, QTableWidgetItem(transaction['banque_emetrice']))
                self.transactions_table.setItem(row_position, 2, QTableWidgetItem(transaction['banque_debitrice']))
                self.transactions_table.setItem(row_position, 3, QTableWidgetItem(f"{float(transaction['montant']):,.2f} €"))
                self.transactions_table.setItem(row_position, 4, QTableWidgetItem(str(transaction['_id'])))

                # Collect data for the graph
                dates.append(transaction_date)
                amounts.append(float(transaction['montant']))

            # Reverse the order of dates and amounts for chronological display
            dates.reverse()
            amounts.reverse()

            # Update the graph
            self.update_transaction_graph(dates, amounts)

            # Update the balance after refreshing transactions
            self.update_balance()

        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Erreur lors du rafraîchissement des transactions: {str(e)}")

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
            
            # Refresh transactions
            self.refresh_transactions()
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


    def calculate_bank_balance(self, emettrice, debitrice):
        ca = certifi.where()
        client = pymongo.MongoClient(
            "mongodb+srv://stokage15:12345@cluster0.o33im.mongodb.net/xyzdb?retryWrites=true&w=majority",
            tlsCAFile=ca
        )
        db = client['myFirstDatabase']
        collection = db['transactions']

        pipeline = [
            {
                "$match": {
                    "$or": [
                        {"banque_emetrice": emettrice, "banque_debitrice": debitrice},
                        {"banque_emetrice": debitrice, "banque_debitrice": emettrice}
                    ]
                }
            },
            {
                "$group": {
                    "_id": "$banque_emetrice",
                    "total": {
                        "$sum": {
                            "$cond": [
                                {"$eq": ["$montant", {"$toInt": "$montant"}]},
                                {"$toInt": "$montant"},
                                {"$toDouble": "$montant"}
                            ]
                        }
                    }
                }
            }
        ]

        results = list(collection.aggregate(pipeline))

        balance = Decimal('0')
        for result in results:
            if result['_id'] == emettrice:
                balance -= Decimal(str(result['total']))
            else:
                balance += Decimal(str(result['total']))

        return balance

    def get_bank_balance(self, emettrice, debitrice):
        balance = self.calculate_bank_balance(emettrice, debitrice)
        if balance > 0:
            return f"{debitrice} doit {abs(balance)} à {emettrice}"
        elif balance < 0:
            return f"{emettrice} doit {abs(balance)} à {debitrice}"
        else:
            return f"Le solde entre {emettrice} et {debitrice} est zéro."

    def show_balance(self):
        emettrice = "CCP"  
        debitrice = "CPA"  
        result = self.get_bank_balance(emettrice, debitrice)
        self.balance_label.setText(result)

    def toggle_camera(self):
        if self.camera is None:
            self.camera = cv2.VideoCapture(0)
            self.timer.start(30)  # Update every 30 ms
            self.camera_btn.setText("Désactiver la caméra")
            self.capture_btn.setVisible(True)
        else:
            self.timer.stop()
            self.camera.release()
            self.camera = None
            self.upload_frame.setText("Aucun fichier sélectionné")
            self.camera_btn.setText("Activer la caméra")
            self.capture_btn.setVisible(False)

    def update_frame(self):
        ret, frame = self.camera.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            q_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            pixmap = pixmap.scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio)
            self.upload_frame.setPixmap(pixmap)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space and self.camera is not None:
            # Capture the current frame
            ret, frame = self.camera.read()
            if ret:
                # Save the captured frame
                cv2.imwrite("captured_cheque.jpg", frame)
                self.process_cheque("captured_cheque.jpg")
                self.toggle_camera()  # Turn off the camera after capturing

    def capture_image(self):
        if self.camera is not None:
            ret, frame = self.camera.read()
            if ret:
                cv2.imwrite("captured_cheque.jpg", frame)
                self.process_cheque("captured_cheque.jpg")
                self.toggle_camera()  # Turn off the camera after capturing

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChequeProcessor()
    window.show()
    sys.exit(app.exec())


import sys
import certifi
import pymongo
import cv2
from datetime import datetime, timedelta
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QFrame, QGridLayout, QMessageBox,
                             QTableWidget, QTableWidgetItem, QSplitter, QTabWidget)
from PyQt6.QtGui import QPixmap, QFont, QImage, QIcon
from PyQt6.QtCore import Qt, QTimer
import pandas as pd
from matplotlib.ticker import MaxNLocator
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from bson.decimal128 import Decimal128
from decimal import Decimal
from cheque_classifier import classify_cheque
from extraction import process_cheque
import matplotlib.pyplot as plt

class ChequeProcessor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tableau de Bord - Processeur de Chèques")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #f0f0f0;
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
            QLabel {
                font-size: 14px;
            }
            QFrame {
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 5px;
            }
        """)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        self.create_header()
        self.create_dashboard()
        self.create_footer()

        self.camera = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        QTimer.singleShot(0, self.initial_update)

    def initial_update(self):
        self.refresh_transactions()

    def refresh_dashboard(self):
        self.update_transaction_graph()
        self.refresh_performance_graph()
        self.update_cheque_count_graph()

    def create_header(self):
        header = QWidget()
        header_layout = QHBoxLayout(header)

        logo = QLabel("Tableau de Bord - Processeur de Chèques")
        logo.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        header_layout.addWidget(logo)

        self.main_layout.addWidget(header)

    def create_dashboard(self):
        dashboard = QWidget()
        dashboard_layout = QGridLayout(dashboard)

        # Top row: Two graphs
        self.transaction_history_graph = self.create_graph_widget("Historique des Transactions")
        dashboard_layout.addWidget(self.transaction_history_graph, 0, 0)

        self.performance_graph = self.create_graph_widget("Performance du Système")
        dashboard_layout.addWidget(self.performance_graph, 0, 1)

        # Bottom left: Cheque Count Graph
        self.cheque_count_graph = self.create_graph_widget("Nombre de chèques traités")
        dashboard_layout.addWidget(self.cheque_count_graph, 1, 0)

        # Bottom right: Upload and Capture Area
        upload_capture_area = self.create_upload_capture_area()
        dashboard_layout.addWidget(upload_capture_area, 1, 1)

        self.main_layout.addWidget(dashboard)

    def create_upload_capture_area(self):
        frame = QFrame()
        layout = QVBoxLayout(frame)

        title_label = QLabel("Téléchargement et Capture de Chèque")
        title_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        layout.addWidget(title_label)

        self.upload_frame = QLabel("Aucun fichier sélectionné")
        self.upload_frame.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.upload_frame.setStyleSheet("""
            background-color: white;
            border: 2px dashed #4CAF50;
            border-radius: 5px;
            padding: 20px;
            min-height: 200px;
        """)
        layout.addWidget(self.upload_frame)

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

        layout.addLayout(buttons_layout)

        return frame
    
  
    def update_cheque_count_graph(self):
        try:
            ca = certifi.where()
            client = pymongo.MongoClient(
                "mongodb+srv://stokage15:12345@cluster0.o33im.mongodb.net/xyzdb?retryWrites=true&w=majority",
                tlsCAFile=ca
            )
            db = client['myFirstDatabase']
            collection = db['transactions']

            # Get the last 6 months
            end_date = datetime.now()
            start_date = end_date - timedelta(days=180)

            pipeline = [
                {
                    "$match": {
                        "date": {"$gte": start_date, "$lte": end_date}
                    }
                },
                {
                    "$group": {
                        "_id": {
                            "year": {"$year": "$date"},
                            "month": {"$month": "$date"}
                        },
                        "count": {"$sum": 1}
                    }
                },
                {
                    "$sort": {"_id.year": 1, "_id.month": 1}
                }
            ]

            results = list(collection.aggregate(pipeline))

            months = []
            counts = []

            for result in results:
                month_year = f"{result['_id']['year']}-{result['_id']['month']:02d}"
                months.append(month_year)
                counts.append(result['count'])

            ax = self.cheque_count_graph.findChild(FigureCanvas).figure.gca()
            ax.clear()
            ax.bar(months, counts, color='#4CAF50')
            ax.set_title('Nombre de chèques traités')
            ax.set_xlabel('Mois')
            ax.set_ylabel('Nombre de chèques')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, linestyle='--', alpha=0.7)

            # Format y-axis to show whole numbers
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))

            self.cheque_count_graph.findChild(FigureCanvas).figure.tight_layout()
            self.cheque_count_graph.findChild(FigureCanvas).draw()

        except Exception as e:
            print(f"Error updating cheque count graph: {str(e)}")
  

    def create_graph_widget(self, title):
        frame = QFrame()
        layout = QVBoxLayout(frame)
        
        title_label = QLabel(title)
        title_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        layout.addWidget(title_label)

        graph = FigureCanvas(Figure(figsize=(5, 4)))
        layout.addWidget(graph)

        return frame

    def create_table_widget(self, title):
        frame = QFrame()
        layout = QVBoxLayout(frame)
        
        title_label = QLabel(title)
        title_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        layout.addWidget(title_label)

        table = QTableWidget()
        table.setColumnCount(5)
        table.setHorizontalHeaderLabels(["Date", "Banque Émettrice", "Banque Débitrice", "Montant", "ID"])
        table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(table)

        return frame


    def create_upload_area(self):
        frame = QFrame()
        layout = QVBoxLayout(frame)

        title_label = QLabel("Téléchargement de Chèque")
        title_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        layout.addWidget(title_label)

        self.upload_frame = QLabel("Aucun fichier sélectionné")
        self.upload_frame.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.upload_frame.setStyleSheet("""
            background-color: white;
            border: 2px dashed #4CAF50;
            border-radius: 5px;
            padding: 20px;
            min-height: 200px;
        """)
        layout.addWidget(self.upload_frame)

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

        layout.addLayout(buttons_layout)

        return frame
    
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

        refresh_btn = QPushButton("Rafraîchir le tableau de bord")
        refresh_btn.setIcon(QIcon.fromTheme("view-refresh"))
        refresh_btn.clicked.connect(self.refresh_dashboard)
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

        right_panel.addTab(transactions_tab, "Transactions")

        # Graph Tab for Transaction History
        graph_tab = QWidget()
        graph_layout = QVBoxLayout(graph_tab)

        self.transaction_history_graph = FigureCanvas(Figure(figsize=(5, 4)))
        graph_layout.addWidget(self.transaction_history_graph)
        right_panel.addTab(graph_tab, "Graphique")

        # Performance Tab for System Performance
        performance_tab = QWidget()
        performance_layout = QVBoxLayout(performance_tab)

        self.performance_graph = FigureCanvas(Figure(figsize=(5, 4)))
        performance_layout.addWidget(self.performance_graph)

        right_panel.addTab(performance_tab, "Performance du Système")

        return right_panel


    def update_transaction_graph(self):
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

            dates = []
            amounts = []
            for transaction in recent_transactions:
                dates.append(transaction['date'])
                amounts.append(float(transaction['montant']))

            # Reverse the order for chronological display
            dates.reverse()
            amounts.reverse()

            ax = self.transaction_history_graph.findChild(FigureCanvas).figure.gca()
            ax.clear()
            ax.plot(dates, amounts, marker='o', linestyle='-', color='#4CAF50')
            ax.set_title('Historique des Transactions')
            ax.set_xlabel('Date')
            ax.set_ylabel('Montant')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, linestyle='--', alpha=0.7)

            # Format y-axis labels as currency
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:,.2f} DA"))

            self.transaction_history_graph.findChild(FigureCanvas).figure.tight_layout()
            self.transaction_history_graph.findChild(FigureCanvas).draw()

        except Exception as e:
            print(f"Error updating transaction graph: {str(e)}")

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

            table = self.transactions_table.findChild(QTableWidget)
            table.setRowCount(0)
            for transaction in recent_transactions:
                row_position = table.rowCount()
                table.insertRow(row_position)
                table.setItem(row_position, 0, QTableWidgetItem(transaction['date'].strftime("%Y-%m-%d %H:%M")))
                table.setItem(row_position, 1, QTableWidgetItem(transaction['banque_emetrice']))
                table.setItem(row_position, 2, QTableWidgetItem(transaction['banque_debitrice']))
                table.setItem(row_position, 3, QTableWidgetItem(f"{float(transaction['montant']):,.2f} DA"))
                table.setItem(row_position, 4, QTableWidgetItem(str(transaction['_id'])))

        except Exception as e:
            print(f"Error refreshing transactions: {str(e)}")
            
    def refresh_performance_graph(self):
        try:
            ca = certifi.where()
            client = pymongo.MongoClient(
                "mongodb+srv://stokage15:12345@cluster0.o33im.mongodb.net/xyzdb?retryWrites=true&w=majority",
                tlsCAFile=ca
            )
            db = client['myFirstDatabase']
            collection_performance = db['system_performance']

            # Fetch all performance logs
            performance_data = list(collection_performance.find())
            
            if performance_data:
                success_count = sum(1 for log in performance_data if log['status'] == 'success')
                fail_count = sum(1 for log in performance_data if log['status'] == 'failure')

                total = success_count + fail_count
                if total > 0:
                    success_percentage = (success_count / total) * 100
                    fail_percentage = (fail_count / total) * 100

                    ax = self.performance_graph.findChild(FigureCanvas).figure.gca()
                    ax.clear()

                    labels = ['Succès', 'Échec']
                    sizes = [success_percentage, fail_percentage]
                    colors = ['#4CAF50', '#FF5252']

                    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                                      startangle=90, pctdistance=0.85)
                    
                    # Customize text properties
                    for autotext in autotexts:
                        autotext.set_color('white')
                        autotext.set_fontweight('bold')
                        autotext.set_fontsize(10)

                    for text in texts:
                        text.set_fontsize(12)
                    
                    ax.axis('equal')
                    ax.set_title("Performance du Système", fontsize=16, fontweight='bold', pad=20)

                    self.performance_graph.findChild(FigureCanvas).figure.tight_layout()
                    self.performance_graph.findChild(FigureCanvas).draw()
                else:
                    self.performance_graph.findChild(FigureCanvas).figure.clear()
                    self.performance_graph.findChild(FigureCanvas).draw()
            else:
                self.performance_graph.findChild(FigureCanvas).figure.clear()
                self.performance_graph.findChild(FigureCanvas).draw()

        except Exception as e:
            print(f"Error updating performance graph: {str(e)}")

    def create_performance_pie_chart(self, success_percentage, fail_percentage):
        """Create and display a circular pie chart showing system success and failure rates."""
        self.performance_graph.figure.clear()
        ax = self.performance_graph.figure.add_subplot(111)

        labels = ['Succès', 'Échec']
        sizes = [success_percentage, fail_percentage]
        colors = ['#4CAF50', '#FF5252']  # Green for success, Red for failure

        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                          startangle=90, pctdistance=0.85, 
                                          wedgeprops=dict(width=0.5, edgecolor='white'))
        
        # Customize text properties
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)

        for text in texts:
            text.set_fontsize(12)
        
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax.axis('equal')
        
        # Add title
        ax.set_title("Performance du Système", fontsize=16, fontweight='bold', pad=20)

        # Add legend
        ax.legend(wedges, labels, title="Statut", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

        self.performance_graph.figure.tight_layout()
        self.performance_graph.draw()

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

            # Check if the processing was successful
            success = result.get('similarite', 0) >= 99

            # Insert to the performance collection
            performance_log = {
                "date": datetime.now(),
                "image_path": file_path,
                "status": "success" if success else "failure"
            }
            ca = certifi.where()
            client = pymongo.MongoClient("mongodb+srv://stokage15:12345@cluster0.o33im.mongodb.net/xyzdb?retryWrites=true&w=majority",tlsCAFile=ca)
            db = client['myFirstDatabase']
            collection_performance = db['system_performance']
            collection_performance.insert_one(performance_log)
        
            # Insert to MongoDB if similarity is high
            if success:
                insert_result = self.insert_to_mongodb(result["montant_en_chiffres"], result["similarite"], bank)
                # Show result window
                self.show_result_window(file_path, bank, result)
                # Calculate and show balance
                self.show_balance()
            else:
                QMessageBox.warning(self, "Processing Failed", "La similarité est trop faible pour accepter le résultat.")
                self.show_result_window(file_path, bank, result)
        
            # Refresh transactions
            self.refresh_transactions()
        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Une erreur est survenue: {str(e)}")

    def show_balance(self):
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
            QMessageBox.information(self, "Solde Actuel", result)
        except Exception as e:
            QMessageBox.warning(self, "Erreur de Solde", f"Erreur lors du calcul du solde: {str(e)}")

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

    def show_result_window(self, file_path, bank, result):
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


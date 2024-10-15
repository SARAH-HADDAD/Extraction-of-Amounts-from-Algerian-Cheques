import sys
import certifi
import pymongo
import cv2
from datetime import datetime, timedelta
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QFrame, QGridLayout, QMessageBox, QStackedWidget)
from PyQt6.QtGui import QPixmap, QFont, QImage, QIcon
from PyQt6.QtCore import Qt, QTimer
from matplotlib.ticker import MaxNLocator, FuncFormatter
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from decimal import Decimal
from cheque_classifier import classify_cheque
from extraction import process_cheque
import mplcursors
import seaborn as sns

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

        self.stacked_widget = QStackedWidget()
        self.main_layout.addWidget(self.stacked_widget)

        self.create_main_page()
        self.create_graphs_page()

        self.stacked_widget.addWidget(self.main_page)
        self.stacked_widget.addWidget(self.graphs_page)

        self.camera = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

    def create_main_page(self):
        self.main_page = QWidget()
        main_layout = QVBoxLayout(self.main_page)

        title = QLabel("Tableau de Bord - Processeur de Chèques")
        title.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title)

        upload_capture_area = self.create_upload_capture_area()
        main_layout.addWidget(upload_capture_area)

        view_graphs_btn = QPushButton("Voir les Graphiques")
        view_graphs_btn.clicked.connect(self.show_graphs_page)
        main_layout.addWidget(view_graphs_btn, alignment=Qt.AlignmentFlag.AlignCenter)

    def create_graphs_page(self):
        self.graphs_page = QWidget()
        graphs_layout = QGridLayout(self.graphs_page)

        self.transaction_history_graph = self.create_graph_widget("Historique des Transactions")
        graphs_layout.addWidget(self.transaction_history_graph, 0, 0)

        self.performance_graph = self.create_graph_widget("Performance du Système")
        graphs_layout.addWidget(self.performance_graph, 0, 1)

        self.bank_transactions_graph = self.create_graph_widget("Évolution des Transactions par Banque")
        graphs_layout.addWidget(self.bank_transactions_graph, 1, 0)

        self.cheque_count_graph = self.create_graph_widget("Nombre de Chèques Traités par Mois")
        graphs_layout.addWidget(self.cheque_count_graph, 1, 1)

        # Bouton de retour
        back_btn = QPushButton("Retour à la page principale")
        back_btn.clicked.connect(self.show_main_page)
        back_btn.setFixedSize(100, 30)
        back_btn.setStyleSheet("""
        QPushButton {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 5px 10px;
            font-size: 12px;
            border-radius: 3px;
        }
        QPushButton:hover {
            background-color: #45a049;
        }""")
        graphs_layout.addWidget(back_btn, 2, 0, 1, 2, alignment=Qt.AlignmentFlag.AlignCenter)

        # Définir les espacements
        graphs_layout.setVerticalSpacing(20)
        graphs_layout.setHorizontalSpacing(20)
        graphs_layout.setContentsMargins(20, 20, 20, 20)

    def show_graphs_page(self):
        self.update_transaction_graph()
        self.refresh_performance_graph()
        self.update_bank_transactions_graph()  # Mettre à jour le nouveau graphique
        self.update_cheque_count_graph()
        self.stacked_widget.setCurrentWidget(self.graphs_page)

    def update_bank_transactions_graph(self):
        try:
            ca = certifi.where()
            client = pymongo.MongoClient(
                "mongodb+srv://stokage15:12345@cluster0.o33im.mongodb.net/xyzdb?retryWrites=true&w=majority",
                tlsCAFile=ca
            )
            db = client['myFirstDatabase']
            collection = db['transactions']

            # Agréger les données par banque et par mois
            pipeline = [
                {
                    "$group": {
                        "_id": {
                            "banque": "$banque_emetrice",
                            "year": {"$year": "$date"},
                            "month": {"$month": "$date"}
                        },
                        "total": {"$sum": {"$toDouble": "$montant"}},
                    }
                },
                {"$sort": {"_id.year": 1, "_id.month": 1}}
            ]

            results = list(collection.aggregate(pipeline))

            # Préparer les données pour le graphique
            banks = list(set([r['_id']['banque'] for r in results]))
            dates = sorted(set([(r['_id']['year'], r['_id']['month']) for r in results]))
            
            fig = self.bank_transactions_graph.findChild(FigureCanvas).figure
            fig.clear()
            ax = fig.add_subplot(111)

            for bank in banks:
                amounts = []
                for date in dates:
                    result = next((r for r in results if r['_id']['banque'] == bank and 
                               r['_id']['year'] == date[0] and r['_id']['month'] == date[1]), None)
                    amounts.append(result['total'] if result else 0)

                date_strings = [f"{date[0]}-{date[1]:02d}" for date in dates]
                ax.plot(date_strings, amounts, marker='o', label=bank)

            ax.set_xlabel('Date')
            ax.set_ylabel('Montant Total (DA)')
            ax.set_title('Évolution des Transactions par Banque')
            ax.legend()
            ax.tick_params(axis='x', rotation=45)   

            ax.grid(True, linestyle='--', alpha=0.7)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            mplcursors.cursor(ax, hover=True).connect(
            "add", lambda sel: sel.annotation.set_text(
                f'Banque: {sel.artist.get_label()}\nDate: {sel.target[0]}\nMontant: {sel.target[1]:.2f} DA'
            ))

            fig.tight_layout()
            self.bank_transactions_graph.findChild(FigureCanvas).draw()

        except Exception as e:
            print(f"Erreur lors de la mise à jour du graphique des transactions par banque : {str(e)}")

    def show_main_page(self):
        self.stacked_widget.setCurrentWidget(self.main_page)
        
    def setup_graph_style(self, ax, title, xlabel, ylabel):
        ax.set_facecolor('#f0f0f0')
        ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.grid(True, linestyle='--', alpha=0.7, color='#cccccc')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#999999')
        ax.spines['left'].set_color('#999999')

    def update_transaction_graph(self):
        try:
            ca = certifi.where()
            client = pymongo.MongoClient(
                "mongodb+srv://stokage15:12345@cluster0.o33im.mongodb.net/xyzdb?retryWrites=true&w=majority",
                tlsCAFile=ca
            )
            db = client['myFirstDatabase']
            collection = db['transactions']

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
                        "total_amount": {"$sum": {"$toDouble": "$montant"}}
                    }
                },
                {
                    "$sort": {"_id.year": 1, "_id.month": 1}
                }
            ]

            results = list(collection.aggregate(pipeline))

            months = []
            amounts = []

            for result in results:
                month_year = f"{result['_id']['year']}-{result['_id']['month']:02d}"
                months.append(month_year)
                amounts.append(result['total_amount'])

            fig = self.transaction_history_graph.findChild(FigureCanvas).figure
            fig.clear()
            ax = fig.add_subplot(111)

            colors = sns.color_palette("viridis", len(months))
            bars = ax.bar(months, amounts, color=colors, alpha=0.8)

            self.setup_graph_style(ax, 'Montant Total Compensé par Mois', 'Mois', 'Montant Total Compensé')
            ax.tick_params(axis='x', rotation=45)

            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:,.0f} DA"))

            cursor = mplcursors.cursor(bars, hover=True)
            cursor.connect("add", lambda sel: sel.annotation.set_text(
                f'Mois: {months[sel.target.index]}\nMontant: {sel.target[1]:,.0f} DA'))

            fig.tight_layout()
            self.transaction_history_graph.findChild(FigureCanvas).draw()

        except Exception as e:
            print(f"Error updating transaction graph: {str(e)}")

    def refresh_performance_graph(self):
        try:
            ca = certifi.where()
            client = pymongo.MongoClient(
                "mongodb+srv://stokage15:12345@cluster0.o33im.mongodb.net/xyzdb?retryWrites=true&w=majority",
                tlsCAFile=ca
            )
            db = client['myFirstDatabase']
            collection_performance = db['system_performance']

            performance_data = list(collection_performance.find())
            
            if performance_data:
                success_count = sum(1 for log in performance_data if log['status'] == 'success')
                fail_count = sum(1 for log in performance_data if log['status'] == 'failure')

                total = success_count + fail_count
                if total > 0:
                    success_percentage = (success_count / total) * 100
                    fail_percentage = (fail_count / total) * 100

                    fig = self.performance_graph.findChild(FigureCanvas).figure
                    fig.clear()
                    ax = fig.add_subplot(111)

                    labels = ['Succès', 'Échec']
                    sizes = [success_percentage, fail_percentage]
                    colors = sns.color_palette("Set2", 2)

                    wedges, texts, autotexts = ax.pie(
                    sizes, labels=labels, colors=colors, startangle=90,
                    wedgeprops=dict(width=0.5, edgecolor='white'),
                    autopct='%1.1f%%')

                    # Style the percentage text
                    for autotext in autotexts:
                        autotext.set_color('black')
                        autotext.set_fontweight('bold')
                    
                    self.setup_graph_style(ax, "Performance du Système", "", "")
                    ax.axis('equal')

                    legend = ax.legend(wedges, labels, title="Statut", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
                    legend.get_title().set_fontweight('bold')

                    cursor = mplcursors.cursor(wedges, hover=True)
                    cursor.connect("add", lambda sel: sel.annotation.set_text(
                        f'{sel.artist.get_label()}: {sel.target.theta/3.6:.1f}%'))

                    fig.tight_layout()
                    self.performance_graph.findChild(FigureCanvas).draw()
                else:
                    self.performance_graph.findChild(FigureCanvas).figure.clear()
                    self.performance_graph.findChild(FigureCanvas).draw()
            else:
                self.performance_graph.findChild(FigureCanvas).figure.clear()
                self.performance_graph.findChild(FigureCanvas).draw()

        except Exception as e:
            print(f"Error updating performance graph: {str(e)}")

    def update_cheque_count_graph(self):
        try:
            ca = certifi.where()
            client = pymongo.MongoClient(
                "mongodb+srv://stokage15:12345@cluster0.o33im.mongodb.net/xyzdb?retryWrites=true&w=majority",
                tlsCAFile=ca
            )
            db = client['myFirstDatabase']
            collection = db['transactions']

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

            fig = self.cheque_count_graph.findChild(FigureCanvas).figure
            fig.clear()
            ax = fig.add_subplot(111)

            colors = sns.color_palette("rocket", len(months))
            bars = ax.bar(months, counts, color=colors, alpha=0.8)

            self.setup_graph_style(ax, 'Nombre de chèques traités par mois', 'Mois', 'Nombre de chèques')
            ax.tick_params(axis='x', rotation=45)

            ax.yaxis.set_major_locator(MaxNLocator(integer=True))

            cursor = mplcursors.cursor(bars, hover=True)
            cursor.connect("add", lambda sel: sel.annotation.set_text(
                f'Mois: {months[sel.target.index]}\nNombre de chèques: {int(sel.target[1])}'))

            fig.tight_layout()
            self.cheque_count_graph.findChild(FigureCanvas).draw()

        except Exception as e:
            print(f"Error updating cheque count graph: {str(e)}")

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
    
    def create_graph_widget(self, title):
        frame = QFrame()
        layout = QVBoxLayout(frame)
        
        title_label = QLabel(title)
        title_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        layout.addWidget(title_label)

        graph = FigureCanvas(Figure(figsize=(5, 4)))
        layout.addWidget(graph)

        return frame

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

        except Exception as e:
            print(f"Error refreshing transactions: {str(e)}")       

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
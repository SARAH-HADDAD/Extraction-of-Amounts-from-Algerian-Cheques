import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
from cheque_classifier import classify_cheque

class ChequeProcessor(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Processeur de Chèques")
        self.geometry("600x500")
        self.configure(bg="#f5f5f5")  # Light gray background

        # Define color scheme
        self.colors = {
            "primary": "#3498db",  # Blue
            "secondary": "#2ecc71",  # Green
            "text": "#34495e",  # Dark blue
            "bg": "#f5f5f5",  # Light gray
            "accent": "#e74c3c"  # Red (for accents or warnings)
        }

        self.create_widgets()

    def create_widgets(self):
        # Main frame
        main_frame = tk.Frame(self, bg=self.colors["bg"], padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title = tk.Label(main_frame, text="Processeur de Chèques", font=("Helvetica", 24, "bold"),
                         bg=self.colors["bg"], fg=self.colors["primary"])
        title.pack(pady=(0, 10))

        # Subtitle
        subtitle = tk.Label(main_frame, text="Téléchargez ou capturez vos chèques bancaires avec précision",
                            font=("Helvetica", 12), bg=self.colors["bg"], fg=self.colors["text"])
        subtitle.pack(pady=(0, 20))

        # Button Frame
        button_frame = tk.Frame(main_frame, bg=self.colors["bg"])
        button_frame.pack(pady=20)

        # Custom button creation
        self.download_btn = self.create_custom_button(button_frame, "Télécharger", self.colors["primary"], self.upload_file)
        self.download_btn.pack(side=tk.LEFT, padx=10)

        self.capture_btn = self.create_custom_button(button_frame, "Capturer", self.colors["secondary"], lambda: None)
        self.capture_btn.pack(side=tk.LEFT, padx=10)

        # Upload Area
        self.upload_frame = tk.Frame(main_frame, bg="white", height=200, bd=2, relief=tk.GROOVE)
        self.upload_frame.pack(fill=tk.X, pady=20)

        upload_icon = tk.Label(self.upload_frame, text="↑", font=("Helvetica", 48), fg=self.colors["primary"], bg="white")
        upload_icon.pack(pady=(20, 0))

        upload_text = tk.Label(self.upload_frame, text="Téléchargez un chèque ici",
                               font=("Helvetica", 12), bg="white", fg=self.colors["text"])
        upload_text.pack()

        file_types = tk.Label(self.upload_frame, text="PNG, JPG, GIF jusqu'à 10MB",
                              font=("Helvetica", 10), bg="white", fg=self.colors["text"])
        file_types.pack(pady=(0, 20))

    def create_custom_button(self, parent, text, bg_color, command):
        style = ttk.Style()
        style.configure(f"{text}.TButton",
                        font=("Helvetica", 12),
                        background=bg_color,
                        foreground="white")

        style.map(f"{text}.TButton",
                  background=[('active', self.get_hover_color(bg_color))],
                  relief=[('pressed', 'sunken')])

        button = ttk.Button(parent, text=text, style=f"{text}.TButton", command=command)
        return button

    def get_hover_color(self, bg_color):
        # Darken the background color slightly for hover effect
        r, g, b = self.winfo_rgb(bg_color)
        return f'#{int(r//256//1.1):02x}{int(g//256//1.1):02x}{int(b//256//1.1):02x}'

    def upload_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif")])
        if file_path:
            # Ouvrir une nouvelle fenêtre pour afficher l'image et les infos
            self.open_image_info_window(file_path)

    def open_image_info_window(self, file_path):
        # Créer une nouvelle fenêtre
        info_window = tk.Toplevel(self)
        info_window.title("Informations Extraites")
        info_window.geometry("700x400")
        info_window.configure(bg=self.colors["bg"])

        # Image frame (left)
        img_frame = tk.Frame(info_window, bg="white", width=300, height=400)
        img_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Charger l'image
        image = Image.open(file_path)
        image = image.resize((300, 300), Image.ANTIALIAS)
        image_tk = ImageTk.PhotoImage(image)

        img_label = tk.Label(img_frame, image=image_tk, bg="white", relief=tk.SOLID, bd=2)
        img_label.image = image_tk
        img_label.pack(pady=10)

        # Info frame (right)
        info_frame = tk.Frame(info_window, bg=self.colors["bg"], padx=10)
        info_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Titre de la section info
        tk.Label(info_frame, text="Détails du Chèque", font=("Helvetica", 16, "bold"), 
                 bg=self.colors["bg"], fg=self.colors["primary"]).pack(pady=10)

        # Ajouter un séparateur
        ttk.Separator(info_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # Informations sur le chèque
        info_data = [
            ("Banque Emetrice", classify_cheque(file_path)),
            ("Montant en lettres", "Dix mille"),
            ("Montant en chiffres", "10 000,00"),
            ("Date d'émission", "01/09/2024")
        ]

        for label, value in info_data:
            self.create_info_row(info_frame, label, value)

        # Ajouter des boutons d'action
        btn_frame = tk.Frame(info_frame, bg=self.colors["bg"])
        btn_frame.pack(pady=20)
        
        save_btn = self.create_custom_button(btn_frame, "Enregistrer", self.colors["secondary"], lambda: print("Enregistrement..."))
        save_btn.pack(side=tk.LEFT, padx=10)

        close_btn = self.create_custom_button(btn_frame, "Fermer", self.colors["accent"], info_window.destroy)
        close_btn.pack(side=tk.LEFT, padx=10)

    def create_info_row(self, parent, label, value):
        """ Crée une ligne avec un label et une valeur pour l'affichage des informations. """
        row_frame = tk.Frame(parent, bg=self.colors["bg"])
        row_frame.pack(fill=tk.X, pady=5)

        tk.Label(row_frame, text=f"{label} :", font=("Helvetica", 12, "bold"), 
                 bg=self.colors["bg"], fg=self.colors["text"]).pack(side=tk.LEFT)
        tk.Label(row_frame, text=value, font=("Helvetica", 12), 
                 bg=self.colors["bg"], fg=self.colors["text"]).pack(side=tk.LEFT, padx=10)

if __name__ == "__main__":
    app = ChequeProcessor()
    app.mainloop()

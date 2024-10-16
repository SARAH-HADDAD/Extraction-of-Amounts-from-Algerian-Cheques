## **`extraction.py`**  
Ce fichier se concentre sur **l’extraction du contenu des chèques**. Voici les principales fonctionnalités :  
- **YOLO** : Utilisé pour identifier les zones d’intérêt telles que les montants en chiffres et en lettres.  
- **TrOCR** : Un modèle OCR de Hugging Face, utilisé pour reconnaître le texte à partir des zones détectées.  
- **Fonctions de prétraitement** : 
  - Estimation du **bruit** et de la **luminosité** pour améliorer la qualité des images avant la détection.  
- **word2numberi18n** : Convertit et corriger les montants en lettres (par exemple, "cent") en valeurs numériques.

---

## **`GUI.py`**  
Ce fichier implémente **l’interface graphique (GUI)** avec **PyQt6**. Les fonctionnalités importantes :  
- **Widgets PyQt6** : Boutons, étiquettes, boîtes de dialogue pour interagir avec l’utilisateur.  
- **Affichage des images** : Permet de visualiser les chèques et les résultats de reconnaissance.  
- **Intégration MongoDB** : Les données sont stockées ou récupérées dans une base de données MongoDB.  
- **Actions utilisateur** : L’utilisateur peut **télécharger des chèques**, visualiser les résultats et les vérifier.

Ce module constitue l’interface principale permettant une **interaction fluide avec le système**.

---

## **`cheque_classifier.py`**  
Ce fichier définit un **modèle CNN** (réseau de neurones convolutif) pour **classifier les chèques**. Ses principales caractéristiques :  
- **Identification des banques** : Le modèle détecte quelle banque a émis le chèque (ex. : BDL, BNA, CPA).  
- **Modèle CNN** : Utilise des couches convolutives pour classer les chèques selon leur origine.  

Ce fichier est utilisé pour **l’organisation et la classification des chèques selon leur banque émettrice**.

---

## **Résumé du Processus du Projet**  
1. **Téléchargement du chèque (GUI)** : L’utilisateur télécharge une image de chèque via l’interface.  
2. **Détection et extraction (extraction.py)** :  
   - YOLO détecte les zones contenant le montant en chiffres et en lettres.  
   - TrOCR reconnaît le texte dans ces zones.  
   - Le texte en lettres est converti en chiffres.  
3. **Classification (cheque_classifier.py)** :  
   - Le modèle CNN identifie la banque émettrice du chèque.  
4. **Vérification & Correction d’erreurs ** :  
   - Le système compare et vérifie que le montant en chiffres correspond à celui en lettres.  
5. **Affichage des résultats (GUI)** : L’interface montre à l’utilisateur les informations extraites et classifiées.
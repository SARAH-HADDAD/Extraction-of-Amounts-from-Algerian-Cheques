# cheque_classifier.py
import torch
import torch.nn as nn
from PIL import Image  
from torchvision import transforms

banques = [
    "BDL",
    "BNA",
    "BNP",
    "CCP",
    "CPA"
]

# CNN Model for Cheque Classification
target_size = (224, 224)

class ChequeClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super(ChequeClassifier, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc_input_size = 128 * (target_size[0] // 8) * (target_size[1] // 8)
        self.fc = nn.Linear(self.fc_input_size, 256)
        self.output = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return self.output(x)

def classify_cheque(original_path):
    model = ChequeClassifier(num_classes=5)
    model.load_state_dict(torch.load("/Users/sarahhaddad/Documents/GitHub/TrOCR/models/cheque_classifier.pth"))
    model.eval()

    def preprocess_image(image_path, transform):
        image = Image.open(image_path).convert('RGB')
        image = transform(image)
        image = image.unsqueeze(0)
        return image

    test_transforms = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    new_image = preprocess_image(original_path, test_transforms)

    with torch.no_grad():
        output = model(new_image)
        _, predicted = torch.max(output, 1)
        predicted_class = banques[predicted.item()]
        print(f'Predicted class: {predicted_class}')
        return predicted_class

import sys
import torch
import torchvision.transforms as transforms
import os

# Append the MiDaS directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "MiDaS"))

from midas.model_loader import load_model, default_models

class MiDaSModel:
    def __init__(self, model_type="DPT_Large", device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(default_models[model_type], self.device)
        self.model.eval()

    def preprocess(self, image):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform(image).unsqueeze(0).to(self.device)

    def predict(self, image):
        with torch.no_grad():
            input_tensor = self.preprocess(image)
            depth_map = self.model(input_tensor)
            return depth_map.squeeze().cpu().numpy()
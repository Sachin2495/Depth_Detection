import sys
import torch
import numpy as np
import os

# Append the AdaBins directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "AdaBins"))

def get_inference_helper():
    from models.AdaBins.infer import InferenceHelper  # Lazy import inside function
    return InferenceHelper
def load_adabins_model():
    from models.AdaBins.infer import InferenceHelper  # Keep it inside
    return InferenceHelper()

def preprocess_adabins_input():  # Define it here instead of importing
    print("Preprocessing input for AdaBins")



class AdaBinsModel:
    def __init__(self, model_path="pretrained/AdaBins_nyu.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = InferenceHelper(model_path, self.device)

    def preprocess(self, image):
        image = cv2.resize(image, (640, 480))
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return image

    def predict(self, image):
        with torch.no_grad():
            input_tensor = self.preprocess(image)
            depth_map = self.model.predict_pil(input_tensor)
            return depth_map.squeeze().cpu().numpy()
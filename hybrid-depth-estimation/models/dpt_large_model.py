import torch
from midas.model_loader import load_model

class DPTLargeModel:
    def __init__(self, device):
        self.device = device
        self.model = self.load_model()

    def load_model(self):
        model = load_model("DPT_Large", self.device)
        model.eval()
        return model

    def preprocess(self, image):
        # Preprocess the image for DPT-Large
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return image

    def predict(self, image):
        image = self.preprocess(image)
        with torch.no_grad():
            depth_map = self.model(image)
        return depth_map.squeeze().cpu().numpy()
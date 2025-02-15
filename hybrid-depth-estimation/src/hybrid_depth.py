import sys
import torch
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Append paths for both models
sys.path.append(os.path.join(os.path.dirname(__file__), "../models/MiDaS"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../models/MiDaS/midas"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../models/AdaBins"))

# Import MiDaS model
from midas.model_loader import default_models, load_model

# Import AdaBins model
try:
    from infer import InferenceHelper
except ModuleNotFoundError:
    print("❌ ERROR: AdaBins module not found! Check paths.")
    sys.exit(1)

# Append the models directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "../models"))

from models.midas_model import MiDaSModel
from models.adabins_model import AdaBinsModel
from models.dpt_large_model import load_model, predict
from utils import preprocess_image, normalize_depth

# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas_model = MiDaSModel(device=device)
adabins_model = AdaBinsModel()
dpt_large_model = load_dpt_large_model(device)

# Load an image
image_path = "sample.jpg"  # Change to your image path
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Preprocess image
image_resized = preprocess_image(image)

# Get depth maps from all models
depth_midas = midas_model.predict(image_resized)
depth_adabins = adabins_model.predict(image_resized)
depth_dpt_large = predict_dpt_large(dpt_large_model, image_resized)

# Normalize depth maps
depth_midas = normalize_depth(depth_midas)
depth_adabins = normalize_depth(depth_adabins)
depth_dpt_large = normalize_depth(depth_dpt_large)

# Hybrid depth map (weighted average)
alpha = 0.4  # Adjust weight for MiDaS
beta = 0.3   # Adjust weight for AdaBins
gamma = 0.3  # Adjust weight for DPT-Large
hybrid_depth = (alpha * depth_midas) + (beta * depth_adabins) + (gamma * depth_dpt_large)

# Display all depth maps
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
axes[0].imshow(depth_midas, cmap="inferno")
axes[0].set_title("MiDaS Depth Map")
axes[0].axis("off")

axes[1].imshow(depth_adabins, cmap="inferno")
axes[1].set_title("AdaBins Depth Map")
axes[1].axis("off")

axes[2].imshow(depth_dpt_large, cmap="inferno")
axes[2].set_title("DPT-Large Depth Map")
axes[2].axis("off")

axes[3].imshow(hybrid_depth, cmap="inferno")
axes[3].set_title("Hybrid Depth Map")
axes[3].axis("off")

plt.show()
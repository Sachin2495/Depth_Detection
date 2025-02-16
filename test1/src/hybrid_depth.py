import sys
import torch
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Append the root directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Append paths for both models
sys.path.append(os.path.join(os.path.dirname(__file__), "../models/MiDaS/midas"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../models/AdaBins"))

# Import MiDaS model
from models.midas_model import MiDaSModel

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
from src.utils import preprocess_image, normalize_depth_map

# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas_model = MiDaSModel(model_type="MiDaS_large", device=device)
adabins_model = AdaBinsModel()

# Load an image
image_path = "sample.jpg"  # Change to your image path
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Preprocess image
image_resized = preprocess_image(image)

# Get depth maps from all models
depth_midas = midas_model.predict(image_resized)
depth_adabins = adabins_model.predict(image_resized)

# Normalize depth maps
depth_midas = normalize_depth(depth_midas)
depth_adabins = normalize_depth(depth_adabins)

# Hybrid depth map (weighted average)
alpha = 0.5  # Adjust weight for MiDaS
beta = 0.5   # Adjust weight for AdaBins
hybrid_depth = (alpha * depth_midas) + (beta * depth_adabins)

# Display all depth maps
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(depth_midas, cmap="inferno")
axes[0].set_title("MiDaS Depth Map")
axes[0].axis("off")

axes[1].imshow(depth_adabins, cmap="inferno")
axes[1].set_title("AdaBins Depth Map")
axes[1].axis("off")

axes[2].imshow(hybrid_depth, cmap="inferno")
axes[2].set_title("Hybrid Depth Map")
axes[2].axis("off")

plt.show()
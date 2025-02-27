def resize_image(image, size=(384, 384)):
    return cv2.resize(image, size)

def normalize_depth_map(depth_map):
    return (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
# Add this to src/utils.py
import cv2
import numpy as np

def preprocess_image(image, size=(384, 384)):
    """Preprocess an image for depth estimation."""
    image = cv2.resize(image, size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # Convert HWC to CHW
    return image

def display_depth_maps(depth_maps, titles):
    fig, axes = plt.subplots(1, len(depth_maps), figsize=(15, 5))
    for ax, depth_map, title in zip(axes, depth_maps, titles):
        ax.imshow(depth_map, cmap="inferno")
        ax.set_title(title)
        ax.axis("off")
    plt.show()
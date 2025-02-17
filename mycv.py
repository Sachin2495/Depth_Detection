import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import Compose
from threading import Thread

# Load MiDaS model
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
midas.to(device)
midas.eval()

# Load MiDaS transform
transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = transforms.small_transform

def process_frame(frame):
    """Apply MiDaS model to compute depth map for a frame."""
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_input = transform(img).to(device)

    with torch.no_grad():
        depth = midas(img_input)
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1), size=img.shape[:2], mode="bicubic", align_corners=False
        ).squeeze().cpu().numpy()
    
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=255.0 / depth.max()), cv2.COLORMAP_MAGMA)
    return depth_colormap

def start_video():
    """Capture live video and process depth in parallel."""
    cap = cv2.VideoCapture(0)  # Access webcam

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Process depth in a separate thread to avoid lag
        thread = Thread(target=lambda: None)
        depth_map = process_frame(frame)
        
        # Show original video and depth side by side
        combined = np.hstack((frame, depth_map))
        cv2.imshow("Live Depth Estimation (Left: Original | Right: Depth)", combined)

        if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_video()

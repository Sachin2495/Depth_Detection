import torch
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load MiDaS model from torch.hub
print("Loading MiDaS model...")
model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Load transforms
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

# Initialize webcam
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

print("Press 'q' to exit and view the 3D plot.")

# Placeholder for depth map
depth_map = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Convert frame to RGB
    img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Apply MiDaS transform
    input_batch = transform(img).to(device)

    # Make depth prediction
    with torch.no_grad():
        prediction = model(input_batch)

    # Convert prediction to a NumPy array
    depth_map = prediction.squeeze().cpu().numpy()

    # Resize depth map to match frame size
    depth_map = cv.resize(depth_map, (frame.shape[1], frame.shape[0]))

    # Normalize depth map for visualization
    depth_normalized = cv.normalize(depth_map, None, 0, 255, cv.NORM_MINMAX)
    depth_normalized = np.uint8(depth_normalized)

    # Display live camera and depth map
    cv.imshow("Live Camera", frame)
    cv.imshow("Depth Map", depth_normalized)

    # Exit on 'q' key
    if cv.waitKey(1) & 0xFF == ord('q'):
        print("Exiting live view and preparing 3D plot...")
        break

cap.release()
cv.destroyAllWindows()

# Check if a depth map was captured
if depth_map is None:
    print("No depth map captured. Exiting...")
    exit()

# Create 3D point cloud
h, w = depth_map.shape
x = np.arange(0, w)
y = np.arange(0, h)
X, Y = np.meshgrid(x, y)
Z = depth_map

# Plot 3D point cloud
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_title("3D Depth Map")
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Depth')
plt.show()

from ultralytics import YOLO
import torch

# Check if MPS is available
print(f"MPS available: {torch.backends.mps.is_available()}")

# Load an official or custom model

# model = YOLO("yolo11n.pt")  # Load an official Detect model

model = YOLO("yolo11n-seg.pt")  # Load an official Segment model

# model = YOLO("yolo11n-pose.pt")  # Load an official Pose model

# model = YOLO("path/to/best.pt")  # Load a custom trained model

# Perform tracking with the model
# results = model.track("https://youtu.be/LNwODJXcvt4", show=True)  # Tracking with default tracker
# results = model.track("https://youtu.be/lIsfCmPI0cU?si=52C3n2lkSmzqchrM", show=True, tracker="bytetrack.yaml")  # with ByteTrack

# GitHub discussion link showing how to only track humans.
# https://github.com/orgs/ultralytics/discussions/13354#discussioncomment-9665190

results = model.track("../videos/8th-grade-vid.mp4", show=True,
  tracker="bytetrack.yaml", device='mps', classes=[0])
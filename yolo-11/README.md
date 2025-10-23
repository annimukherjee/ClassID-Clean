# YOLO-11 Tracking with ClassID

This directory contains a partial implementation of the ClassID person tracking system using YOLO-11. Each subdirectory builds upon the previous one, progressively adding features from the ClassID paper.

## Main Implementation

**If you want to see the main implementation, go to `4_local_reconsil_yolo/`**. This contains the most complete version of the ClassID tracking system implemented so far.

## Prerequisites

Install the required dependencies:

```bash
pip install -r requirements.txt
```

The main dependencies are:
- opencv-python
- matplotlib
- numpy
- torch
- ultralytics

## Directory Structure

The subdirectories are organized incrementally, with each one building on the previous implementation:

### 0_vanilla-yolo-out-of-the-box
Basic YOLO-11 examples straight from the documentation. Includes detection, segmentation, and pose estimation models. This is the baseline before any ClassID enhancements.

### 1_vanilla-yolo-out-of-the-box-person-only
Same as the baseline but filtered to detect only persons (class 0). This is where person tracking begins.

### 2_vanilla_yolo_num_people_and_ids
Adds tracking visualization with ID counting. Shows how many people are in each frame and tracks their assigned IDs over time with matplotlib plots.

### 3_vanilla_ephemeral_yolo
Implements the first ClassID feature: ephemeral ID filtering. This removes IDs that appear for fewer than a specified number of frames, addressing one of the main problems with vanilla YOLO tracking.

### 4_local_reconsil_yolo
Implements local reconciliation tracking, the next stage of the ClassID system. This is the most advanced implementation currently available.

## Running the Code

Navigate to any subdirectory and run the scripts there. For the main implementation:

```bash
cd 4_local_reconsil_yolo
python tracking_visualization_ephemeral_filtering_local_reconsil.py
```

For the ephemeral filtering stage with custom threshold:

```bash
cd 3_vanilla_ephemeral_yolo
python tracking_visualization_ephemeral_filtering.py --ephemeral_threshold 50
```

For basic YOLO tracking:

```bash
cd 0_vanilla-yolo-out-of-the-box
python 1_detect_model.py
```

## Model Files

The directory includes several YOLO-11 model weights:
- `yolo11n.pt` - Nano model for detection
- `yolo11l.pt` - Large model for better accuracy
- `yolo11m.pt` - Medium model (balance of speed and accuracy)
- `yolo11n-pose.pt` - Nano model for pose estimation
- `yolo11n-seg.pt` - Nano model for segmentation

## Notes

- The code is configured to use Apple Metal Performance Shaders (MPS) if available
- Video files should be placed in the `videos/` directory
- Press 'q' in the video window to quit during processing
- Plots update in real-time during tracking

## Reference

The tracking implementation is based on YOLO-11 documentation: https://docs.ultralytics.com/modes/track/
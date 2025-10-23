"""
Simple test script to demonstrate ClassID ephemeral filtering
Run this to see the difference between raw YOLO and ClassID-filtered tracking
"""

from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def run_classid_demo(video_path="./videos/8th-grade-vid.mp4", min_duration_frames=15, max_duration_seconds=30):
    """
    Demo showing the difference between raw YOLO tracking and ClassID filtering
    """
    print("="*60)
    print("ClassID Ephemeral Filter Demo")
    print("="*60)
    print(f"Video: {video_path}")
    print(f"Processing duration: {max_duration_seconds} seconds")
    print(f"Min ID duration: {min_duration_frames} frames")
    print(f"MPS available: {torch.backends.mps.is_available()}")

    # Initialize YOLO model
    model = YOLO("yolo11l.pt")

    # Track data
    raw_data = []
    frame_count = 0

    # Estimate max frames to process
    estimated_fps = 5  # YOLO processing speed
    max_frames = max_duration_seconds * estimated_fps

    print(f"\nProcessing first {max_frames} frames (~{max_duration_seconds}s)...")

    # Run YOLO tracking
    results = model.track(
        video_path,
        show=False,  # Set to True if you want to see video
        tracker="bytetrack.yaml",
        classes=[0],
        device='mps' if torch.backends.mps.is_available() else 'cpu',
        stream=True,
        verbose=False
    )

    # Collect tracking data
    for result in results:
        frame_count += 1

        # Stop after processing enough frames
        if frame_count > max_frames:
            print(f"Reached {max_duration_seconds}s limit, stopping...")
            break

        if result.boxes is not None and len(result.boxes) > 0:
            if hasattr(result.boxes, 'id') and result.boxes.id is not None:
                track_ids = result.boxes.id.cpu().numpy().astype(int)

                for track_id in track_ids:
                    raw_data.append({
                        'frame': frame_count,
                        'track_id': track_id,
                        'person_count': len(track_ids)
                    })

        if frame_count % 15 == 0:
            print(f"Processed {frame_count}/{max_frames} frames...")

    print(f"Total frames processed: {frame_count}")

    # Analyze raw tracking results
    id_frames = defaultdict(list)
    for data in raw_data:
        id_frames[data['track_id']].append(data['frame'])

    raw_total_ids = len(id_frames)
    print(f"\nRaw YOLO Tracking Results:")
    print(f"  Total unique IDs: {raw_total_ids}")

    # Apply ClassID ephemeral filtering
    print(f"\nApplying ClassID ephemeral filtering...")
    persistent_ids = {}
    ephemeral_ids = []

    for track_id, frames in id_frames.items():
        if len(frames) >= min_duration_frames:
            persistent_ids[track_id] = frames
        else:
            ephemeral_ids.append(track_id)

    filtered_total_ids = len(persistent_ids)
    removed_ids = len(ephemeral_ids)

    print(f"ClassID Filtered Results:")
    print(f"  Persistent IDs: {filtered_total_ids}")
    print(f"  Ephemeral IDs removed: {removed_ids}")
    print(f"  Improvement: {raw_total_ids/max(filtered_total_ids, 1):.2f}x reduction in ID count")

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Raw tracking visualization
    ax1.set_title(f'Raw YOLO Tracking\nTotal IDs: {raw_total_ids}')
    colors1 = plt.cm.Set1(np.linspace(0, 1, min(raw_total_ids, 9)))
    for i, (track_id, frames) in enumerate(id_frames.items()):
        color = colors1[i % len(colors1)]
        ax1.scatter(frames, [track_id] * len(frames),
                   c=[color], s=20, alpha=0.6, label=f'ID {track_id}')
    ax1.set_xlabel('Frame Number')
    ax1.set_ylabel('Track ID')
    ax1.grid(True)

    # Filtered tracking visualization
    ax2.set_title(f'ClassID Filtered Tracking\nPersistent IDs: {filtered_total_ids}')
    colors2 = plt.cm.Set2(np.linspace(0, 1, min(filtered_total_ids, 8)))
    for i, (track_id, frames) in enumerate(persistent_ids.items()):
        color = colors2[i % len(colors2)]
        ax2.scatter(frames, [track_id] * len(frames),
                   c=[color], s=20, alpha=0.8, label=f'ID {track_id}')
    ax2.set_xlabel('Frame Number')
    ax2.set_ylabel('Track ID')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    # Print detailed analysis
    print(f"\nDetailed Analysis:")
    print(f"Ephemeral IDs removed (< {min_duration_frames} frames):")
    for eid in ephemeral_ids:
        print(f"  ID {eid}: {len(id_frames[eid])} frames")

    print(f"\nPersistent IDs kept (>= {min_duration_frames} frames):")
    for pid, frames in persistent_ids.items():
        print(f"  ID {pid}: {len(frames)} frames (frames {min(frames)}-{max(frames)})")

if __name__ == "__main__":
    # Run the demo
    run_classid_demo(
        video_path="./videos/8th-grade-vid.mp4",
        min_duration_frames=15,  # Adjust this threshold as needed
        max_duration_seconds=30   # Process only first 30 seconds
    )
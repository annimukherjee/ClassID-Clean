from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import pandas as pd

class ClassIDTrackingVisualizer:
    def __init__(self, model_path="yolo11n.pt", video_path="./videos/8th-grade-vid.mp4", min_id_duration_frames=30, max_duration_seconds=30):
        self.model = YOLO(model_path)
        self.video_path = video_path
        self.min_id_duration_frames = min_id_duration_frames
        self.max_duration_seconds = max_duration_seconds

        # Storage for raw tracking data
        self.raw_tracking_data = []
        self.raw_frame_counts = []
        self.raw_frame_numbers = []
        self.raw_unique_ids = set()
        self.raw_id_history = defaultdict(list)

        # Storage for filtered tracking data
        self.filtered_frame_counts = []
        self.filtered_frame_numbers = []
        self.filtered_unique_ids = set()
        self.filtered_id_history = defaultdict(list)

        self.current_frame = 0

        # Setup matplotlib for comparison plots
        plt.ion()
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Initialize plots
        self.ax1.set_title('Raw YOLO Tracking - Person Count per Frame')
        self.ax1.set_xlabel('Frame Number')
        self.ax1.set_ylabel('Person Count')
        self.ax1.grid(True)

        self.ax2.set_title('Raw YOLO Tracking - ID Assignments Over Time')
        self.ax2.set_xlabel('Frame Number')
        self.ax2.set_ylabel('Person ID')
        self.ax2.grid(True)

        self.ax3.set_title('ClassID Filtered - Person Count per Frame')
        self.ax3.set_xlabel('Frame Number')
        self.ax3.set_ylabel('Person Count')
        self.ax3.grid(True)

        self.ax4.set_title('ClassID Filtered - ID Assignments Over Time')
        self.ax4.set_xlabel('Frame Number')
        self.ax4.set_ylabel('Person ID')
        self.ax4.grid(True)

        plt.tight_layout()

    def filter_ephemeral_ids(self):
        """Apply ClassID ephemeral filtering to the raw tracking data."""
        print(f"[ClassID] Applying ephemeral ID filtering with threshold: {self.min_id_duration_frames} frames")

        # Convert raw data to DataFrame for processing
        df = pd.DataFrame(self.raw_tracking_data, columns=['frame_id', 'track_id', 'person_count'])

        if df.empty:
            print("[ClassID] No data to filter")
            return

        # Count frame occurrences for each track_id (ClassID methodology)
        id_counts = df.groupby('track_id')['frame_id'].nunique()
        ephemeral_ids = id_counts[id_counts < self.min_id_duration_frames].index.tolist()

        print(f"[ClassID] Found {len(ephemeral_ids)} ephemeral IDs to remove")
        print(f"[ClassID] Total unique IDs before filtering: {df['track_id'].nunique()}")

        # Filter out ephemeral IDs
        if ephemeral_ids:
            df_filtered = df[~df['track_id'].isin(ephemeral_ids)].copy()
        else:
            df_filtered = df.copy()

        print(f"[ClassID] Total unique IDs after filtering: {df_filtered['track_id'].nunique()}")

        # Rebuild filtered data structures
        self.filtered_frame_numbers = []
        self.filtered_frame_counts = []
        self.filtered_unique_ids = set()
        self.filtered_id_history = defaultdict(list)

        # Group by frame and rebuild frame-wise data
        for frame_id in sorted(df_filtered['frame_id'].unique()):
            frame_data = df_filtered[df_filtered['frame_id'] == frame_id]

            self.filtered_frame_numbers.append(frame_id)
            self.filtered_frame_counts.append(len(frame_data))

            # Track ID history for this frame
            for _, row in frame_data.iterrows():
                track_id = row['track_id']
                self.filtered_unique_ids.add(track_id)
                self.filtered_id_history[track_id].append(frame_id)

    def update_plots(self):
        """Update both raw and filtered visualizations."""
        if len(self.raw_frame_numbers) > 0:
            # Update raw tracking plots (top row)
            self.ax1.clear()
            self.ax1.bar(self.raw_frame_numbers, self.raw_frame_counts, width=1.0, alpha=0.7, color='red')
            self.ax1.set_title(f'Raw YOLO Tracking - Person Count\n(Current: {self.raw_frame_counts[-1] if self.raw_frame_counts else 0}, Total IDs: {len(self.raw_unique_ids)})')
            self.ax1.set_xlabel('Frame Number')
            self.ax1.set_ylabel('Person Count')
            self.ax1.grid(True)

            # Raw ID tracking plot
            self.ax2.clear()
            colors_raw = plt.cm.Set1(np.linspace(0, 1, min(len(self.raw_unique_ids), 9)))
            for i, (track_id, frames) in enumerate(self.raw_id_history.items()):
                if frames:
                    color = colors_raw[i % len(colors_raw)]
                    self.ax2.scatter(frames, [track_id] * len(frames),
                                   c=[color], s=10, alpha=0.6)

            self.ax2.set_title(f'Raw YOLO Tracking - ID Assignments\n(Total IDs: {len(self.raw_unique_ids)})')
            self.ax2.set_xlabel('Frame Number')
            self.ax2.set_ylabel('Person ID')
            self.ax2.grid(True)

        # Update filtered plots (bottom row) if filtering has been applied
        if len(self.filtered_frame_numbers) > 0:
            self.ax3.clear()
            self.ax3.bar(self.filtered_frame_numbers, self.filtered_frame_counts, width=1.0, alpha=0.7, color='green')
            self.ax3.set_title(f'ClassID Filtered - Person Count\n(Current: {self.filtered_frame_counts[-1] if self.filtered_frame_counts else 0}, Total IDs: {len(self.filtered_unique_ids)})')
            self.ax3.set_xlabel('Frame Number')
            self.ax3.set_ylabel('Person Count')
            self.ax3.grid(True)

            # Filtered ID tracking plot
            self.ax4.clear()
            colors_filtered = plt.cm.Set2(np.linspace(0, 1, min(len(self.filtered_unique_ids), 8)))
            for i, (track_id, frames) in enumerate(self.filtered_id_history.items()):
                if frames:
                    color = colors_filtered[i % len(colors_filtered)]
                    self.ax4.scatter(frames, [track_id] * len(frames),
                                   c=[color], s=15, alpha=0.8)

            self.ax4.set_title(f'ClassID Filtered - ID Assignments\n(Total IDs: {len(self.filtered_unique_ids)})')
            self.ax4.set_xlabel('Frame Number')
            self.ax4.set_ylabel('Person ID')
            self.ax4.grid(True)

        plt.tight_layout()
        plt.pause(0.01)

    def process_video(self):
        print("Starting ClassID-enhanced YOLO tracking...")
        print(f"Processing first {self.max_duration_seconds} seconds of video")
        print("This will show raw YOLO problems and ClassID improvements side-by-side")
        print("Press 'q' in the video window to quit")

        # Estimate frames to process (assuming ~5 FPS for YOLO processing)
        estimated_fps = 5  # YOLO processing speed, not video FPS
        max_frames = self.max_duration_seconds * estimated_fps

        # Phase 1: Collect raw tracking data
        print(f"\n=== Phase 1: Raw YOLO Tracking (max {max_frames} frames) ===")

        results = self.model.track(
            self.video_path,
            show=True,
            tracker="bytetrack.yaml",
            classes=[0],  # Only detect persons
            device='mps' if torch.backends.mps.is_available() else 'cpu',
            stream=True,
            verbose=False
        )

        for result in results:
            self.current_frame += 1

            # Stop after processing enough frames for the specified duration
            if self.current_frame > max_frames:
                print(f"Reached {self.max_duration_seconds}s limit, stopping video processing...")
                break

            # Count people in current frame
            person_count = 0
            current_ids = set()

            if result.boxes is not None and len(result.boxes) > 0:
                person_count = len(result.boxes)

                # Extract tracking IDs if available
                if hasattr(result.boxes, 'id') and result.boxes.id is not None:
                    track_ids = result.boxes.id.cpu().numpy().astype(int)
                    current_ids = set(track_ids)

                    # Store raw tracking data
                    for track_id in track_ids:
                        self.raw_unique_ids.add(track_id)
                        self.raw_id_history[track_id].append(self.current_frame)
                        self.raw_tracking_data.append([self.current_frame, track_id, person_count])

            # Store raw frame data
            self.raw_frame_numbers.append(self.current_frame)
            self.raw_frame_counts.append(person_count)

            # Update raw plots every 5 frames
            if self.current_frame % 5 == 0:
                self.update_plots()

            # Print progress
            if self.current_frame % 15 == 0:
                print(f"Frame {self.current_frame}/{max_frames}: {person_count} people, "
                      f"Total unique IDs: {len(self.raw_unique_ids)}")

        # Phase 2: Apply ClassID filtering
        print(f"\n=== Phase 2: Applying ClassID Ephemeral Filtering ===")
        self.filter_ephemeral_ids()

        # Final plot update
        self.update_plots()

        # Print comparison statistics
        self.print_comparison_summary()

        # Keep plots open
        print("\nVisualization complete! Close the plot window to exit.")
        print("Top row shows raw YOLO tracking problems.")
        print("Bottom row shows ClassID filtered results.")
        plt.ioff()
        plt.show()

    def print_comparison_summary(self):
        print("\n" + "="*60)
        print("CLASSID TRACKING COMPARISON SUMMARY")
        print("="*60)
        print(f"Total frames processed: {self.current_frame}")
        print(f"Average people per frame: {np.mean(self.raw_frame_counts):.2f}")
        print()
        print("RAW YOLO TRACKING:")
        print(f"  • Total unique IDs assigned: {len(self.raw_unique_ids)}")
        print(f"  • Max people in single frame: {max(self.raw_frame_counts) if self.raw_frame_counts else 0}")
        print(f"  • Min people in single frame: {min(self.raw_frame_counts) if self.raw_frame_counts else 0}")
        print()
        print("CLASSID FILTERED TRACKING:")
        print(f"  • Total unique IDs after filtering: {len(self.filtered_unique_ids)}")
        print(f"  • Ephemeral IDs removed: {len(self.raw_unique_ids) - len(self.filtered_unique_ids)}")
        print(f"  • Improvement ratio: {len(self.raw_unique_ids) / max(len(self.filtered_unique_ids), 1):.2f}x reduction")
        print()
        print("ClassID Ephemeral Filter Settings:")
        print(f"  • Minimum ID duration: {self.min_id_duration_frames} frames")
        print(f"  • Filtering threshold: IDs appearing < {self.min_id_duration_frames} frames removed")

        if len(self.filtered_unique_ids) > 0:
            print(f"\nPersistent ID Analysis:")
            for track_id, frames in self.filtered_id_history.items():
                if len(frames) >= self.min_id_duration_frames:
                    lifespan = len(frames)
                    first_frame = min(frames)
                    last_frame = max(frames)
                    print(f"  ID {track_id}: {lifespan} frames (frames {first_frame}-{last_frame})")


if __name__ == "__main__":
    print(f"MPS available: {torch.backends.mps.is_available()}")

    # Create ClassID-enhanced visualizer
    # min_id_duration_frames: ClassID paper suggests 1 minute, adjust based on your video FPS
    # For 5 FPS video: 30 frames = 6 seconds, 60 frames = 12 seconds
    visualizer = ClassIDTrackingVisualizer(
        model_path="yolo11l.pt",
        video_path="./videos/8th-grade-vid.mp4",
        min_id_duration_frames=30  # Adjust this threshold as needed
    )
    visualizer.process_video()
#!/usr/bin/env python3
"""
YOLO Tracking Visualization with Ephemeral ID Filtering
========================================================

This script processes a video using YOLO-11 tracking and implements ephemeral ID filtering.
IDs that appear for less than the specified threshold are filtered out of the final results.

Features:
- Real-time tracking visualization with matplotlib plots
- Ephemeral ID filtering based on minimum duration threshold
- Color-coded bounding boxes: GREEN for persistent IDs, RED for ephemeral IDs
- Command-line argument for ephemeral threshold (default: 50 frames)
- Enhanced video display with filtered annotations

Usage:
    python tracking_visualization_ephemeral_filtering.py --ephemeral_threshold 50
"""

import argparse
from collections import defaultdict
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics import YOLO


class YOLOTrackingVisualizerWithEphemeralFiltering:
    def __init__(self, model_path="../yolo11n.pt", video_path="../videos/8th-grade-vid.mp4", ephemeral_threshold=50):
        """
        Initialize the YOLO tracking visualizer with ephemeral filtering.

        Args:
            model_path (str): Path to YOLO model weights
            video_path (str): Path to input video file
            ephemeral_threshold (int): Minimum frames an ID must appear to be considered persistent
        """
        self.model = YOLO(model_path)
        self.video_path = video_path
        self.ephemeral_threshold = ephemeral_threshold

        # Data storage for tracking (all IDs including ephemeral)
        self.num_of_people_in_each_frame = []
        self.frame_numbers = []
        self.all_unique_ids = set()  # All IDs (including ephemeral)
        self.all_id_history = defaultdict(list)  # All ID history
        self.current_frame = 0

        # Data storage for filtered tracking (persistent IDs only)
        self.filtered_num_of_people_in_each_frame = []
        self.persistent_ids = set()  # Only persistent IDs
        self.persistent_id_history = defaultdict(list)  # Only persistent ID history

        # Ephemeral filtering data
        self.ephemeral_ids = set()  # IDs that are ephemeral
        self.id_frame_counts = defaultdict(int)  # Count frames per ID

        # Video annotation data
        self.frame_detections = {}  # Store bounding boxes per frame for video annotation

        # Setup matplotlib for live plotting
        plt.ion()
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(16, 10))

        # Initialize plots
        self._setup_plots()

    def _setup_plots(self):
        """Setup the matplotlib subplots."""
        # Top left: All detections (including ephemeral)
        self.ax1.set_title('All People Detected per Frame (Including Ephemeral)')
        self.ax1.set_xlabel('Frame Number')
        self.ax1.set_ylabel('Person Count')
        self.ax1.grid(True)

        # Top right: All ID tracking (including ephemeral)
        self.ax2.set_title('All ID Tracking Over Time (Including Ephemeral)')
        self.ax2.set_xlabel('Frame Number')
        self.ax2.set_ylabel('Person ID')
        self.ax2.grid(True)

        # Bottom left: Filtered detections (persistent only)
        self.ax3.set_title(f'Persistent People (≥{self.ephemeral_threshold} frames)')
        self.ax3.set_xlabel('Frame Number')
        self.ax3.set_ylabel('Person Count')
        self.ax3.grid(True)

        # Bottom right: Filtered ID tracking (persistent only)
        self.ax4.set_title(f'Persistent ID Tracking (≥{self.ephemeral_threshold} frames)')
        self.ax4.set_xlabel('Frame Number')
        self.ax4.set_ylabel('Person ID')
        self.ax4.grid(True)

        plt.tight_layout()

    def _identify_ephemeral_ids(self):
        """Identify which IDs are ephemeral based on the threshold."""
        self.ephemeral_ids = {
            track_id for track_id, count in self.id_frame_counts.items()
            if count < self.ephemeral_threshold
        }
        self.persistent_ids = self.all_unique_ids - self.ephemeral_ids

        print(f"[FILTER] Total unique IDs: {len(self.all_unique_ids)}")
        print(f"[FILTER] Ephemeral IDs (< {self.ephemeral_threshold} frames): {len(self.ephemeral_ids)}")
        print(f"[FILTER] Persistent IDs (≥ {self.ephemeral_threshold} frames): {len(self.persistent_ids)}")

        # Build persistent tracking data
        self._build_persistent_tracking_data()

    def _build_persistent_tracking_data(self):
        """Build tracking data for persistent IDs only."""
        self.persistent_id_history = {
            track_id: frames for track_id, frames in self.all_id_history.items()
            if track_id in self.persistent_ids
        }

        # Rebuild filtered frame counts for all frames processed so far
        self.filtered_num_of_people_in_each_frame = []
        for frame_num in self.frame_numbers:
            persistent_count = sum(
                1 for track_id, frames in self.persistent_id_history.items()
                if frame_num in frames
            )
            self.filtered_num_of_people_in_each_frame.append(persistent_count)

    def _get_box_color_and_label(self, track_id: int) -> Tuple[Tuple[int, int, int], str]:
        """
        Get color and label for bounding box based on whether ID is ephemeral or persistent.

        Args:
            track_id (int): The tracking ID

        Returns:
            Tuple[Tuple[int, int, int], str]: (BGR_color, label_text)
        """
        if track_id in self.ephemeral_ids:
            return (0, 0, 255), f"ID: {track_id} (Removed)"  # Red for ephemeral
        elif track_id in self.persistent_ids:
            return (0, 255, 0), f"ID: {track_id}"  # Green for persistent
        else:
            # ID hasn't been classified yet (still being evaluated)
            return (0, 255, 255), f"ID: {track_id} (Evaluating)"  # Yellow for evaluating

    def _draw_text_with_background(self, frame, text: str, pos: Tuple[int, int],
                                 font_scale=0.6, text_color=(255, 255, 255), bg_color=(0, 0, 0)):
        """Draw text with background for better visibility."""
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        start_point = (pos[0], pos[1] - text_height - 5)
        end_point = (pos[0] + text_width + 5, pos[1])
        cv2.rectangle(frame, start_point, end_point, bg_color, -1)
        cv2.putText(frame, text, (pos[0], pos[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 2)

    def update_plots(self):
        """Update all matplotlib plots."""
        if len(self.frame_numbers) > 0:
            # Top left: All people count
            self.ax1.clear()
            self.ax1.bar(self.frame_numbers, self.num_of_people_in_each_frame, width=1.0, alpha=0.4, color='blue')
            self.ax1.set_title(f'All People Detected (Current: {self.num_of_people_in_each_frame[-1] if self.num_of_people_in_each_frame else 0})')
            self.ax1.set_xlabel('Frame Number')
            self.ax1.set_ylabel('Person Count')
            self.ax1.grid(True)

            # Top right: All ID tracking
            self.ax2.clear()
            if self.all_unique_ids:
                colors = plt.cm.tab20(np.linspace(0, 1, len(self.all_unique_ids)))
                for i, (track_id, frames) in enumerate(self.all_id_history.items()):
                    if frames:
                        color = 'red' if track_id in self.ephemeral_ids else 'green' if track_id in self.persistent_ids else 'orange'
                        alpha = 0.3 if track_id in self.ephemeral_ids else 0.8
                        self.ax2.scatter(frames, [track_id] * len(frames),
                                       c=color, s=15, alpha=alpha, label=f'ID {track_id}')

            self.ax2.set_title(f'All ID Tracking (Total: {len(self.all_unique_ids)}, Ephemeral: {len(self.ephemeral_ids)})')
            self.ax2.set_xlabel('Frame Number')
            self.ax2.set_ylabel('Person ID')
            self.ax2.grid(True)

            # Bottom left: Filtered people count
            if self.filtered_num_of_people_in_each_frame and len(self.filtered_num_of_people_in_each_frame) == len(self.frame_numbers):
                self.ax3.clear()
                self.ax3.bar(self.frame_numbers, self.filtered_num_of_people_in_each_frame, width=1.0, alpha=0.6, color='green')
                self.ax3.set_title(f'Persistent People (Current: {self.filtered_num_of_people_in_each_frame[-1]})')
                self.ax3.set_xlabel('Frame Number')
                self.ax3.set_ylabel('Person Count')
                self.ax3.grid(True)
            else:
                # Clear the plot if no data or mismatch
                self.ax3.clear()
                self.ax3.set_title(f'Persistent People (≥{self.ephemeral_threshold} frames) - No data yet')
                self.ax3.set_xlabel('Frame Number')
                self.ax3.set_ylabel('Person Count')
                self.ax3.grid(True)

            # Bottom right: Filtered ID tracking
            if self.persistent_id_history:
                self.ax4.clear()
                colors = plt.cm.tab10(np.linspace(0, 1, len(self.persistent_ids)))
                for i, (track_id, frames) in enumerate(self.persistent_id_history.items()):
                    if frames:
                        color = colors[i % len(colors)]
                        self.ax4.scatter(frames, [track_id] * len(frames),
                                       c=[color], s=20, alpha=0.8, label=f'ID {track_id}')

                self.ax4.set_title(f'Persistent ID Tracking (Total: {len(self.persistent_ids)})')
                self.ax4.set_xlabel('Frame Number')
                self.ax4.set_ylabel('Person ID')
                self.ax4.grid(True)

                if len(self.persistent_ids) <= 15:
                    self.ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            plt.tight_layout()
            plt.pause(0.01)

    def process_video(self):
        """Process video with YOLO tracking and ephemeral filtering."""
        print("Starting YOLO tracking with ephemeral filtering...")
        print(f"Ephemeral threshold: {self.ephemeral_threshold} frames")
        print("Press 'q' in the video window to quit")

        # Phase 1: Collect all tracking data
        print("\n=== Phase 1: Collecting tracking data ===")
        results = self.model.track(
            self.video_path,
            show=False,  # We'll handle video display manually
            tracker="bytetrack.yaml",
            classes=[0],  # Only detect persons
            device='mps' if torch.backends.mps.is_available() else 'cpu',
            stream=True,
            verbose=False
        )

        # Open video for manual display with annotations
        cap = cv2.VideoCapture(self.video_path)

        for result in results:
            self.current_frame += 1

            # Get the current video frame
            ret, frame = cap.read()
            if not ret:
                break

            # Count people in current frame
            person_count = 0
            frame_detections = []

            if result.boxes is not None and len(result.boxes) > 0:
                person_count = len(result.boxes)

                # Extract tracking IDs and bounding boxes
                if hasattr(result.boxes, 'id') and result.boxes.id is not None:
                    track_ids = result.boxes.id.cpu().numpy().astype(int)
                    xyxy = result.boxes.xyxy.cpu().numpy()  # Bounding boxes

                    # Store detections for this frame
                    for track_id, bbox in zip(track_ids, xyxy):
                        x1, y1, x2, y2 = bbox.astype(int)
                        frame_detections.append({
                            'track_id': track_id,
                            'bbox': (x1, y1, x2, y2)
                        })

                        # Update tracking history and frame counts
                        self.all_unique_ids.add(track_id)
                        self.all_id_history[track_id].append(self.current_frame)
                        self.id_frame_counts[track_id] += 1

            # Store frame detections for video annotation
            self.frame_detections[self.current_frame] = frame_detections

            # Store frame data
            self.frame_numbers.append(self.current_frame)
            self.num_of_people_in_each_frame.append(person_count)

            # Re-evaluate ephemeral status periodically
            if self.current_frame % 30 == 0:
                self._identify_ephemeral_ids()

            # Draw bounding boxes on frame
            frame_copy = frame.copy()
            for detection in frame_detections:
                track_id = detection['track_id']
                x1, y1, x2, y2 = detection['bbox']

                # Get color and label based on ephemeral status
                color, label = self._get_box_color_and_label(track_id)

                # Draw bounding box
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
                self._draw_text_with_background(frame_copy, label, (x1, y1), bg_color=color)

            # Add frame info
            info_text = f"Frame: {self.current_frame} | People: {person_count} | "
            info_text += f"All IDs: {len(self.all_unique_ids)} | Persistent: {len(self.persistent_ids)} | "
            info_text += f"Ephemeral: {len(self.ephemeral_ids)}"
            self._draw_text_with_background(frame_copy, info_text, (10, 30), font_scale=0.7, bg_color=(0, 0, 0))

            # Show annotated frame
            cv2.imshow('YOLO Tracking with Ephemeral Filtering', frame_copy)

            # Update plots every 10 frames for performance
            if self.current_frame % 10 == 0:
                self.update_plots()

            # Print current stats
            if self.current_frame % 50 == 0:
                print(f"Frame {self.current_frame}: {person_count} people, "
                      f"Total IDs: {len(self.all_unique_ids)}, "
                      f"Persistent: {len(self.persistent_ids)}, "
                      f"Ephemeral: {len(self.ephemeral_ids)}")

            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        # Phase 2: Final ephemeral filtering
        print("\n=== Phase 2: Final ephemeral filtering ===")
        self._identify_ephemeral_ids()

        # Final plot update
        self.update_plots()

        # Print summary statistics
        self.print_summary()

        # Keep plots open
        print("\nVisualization complete! Close the plot window to exit.")
        plt.ioff()
        plt.show()

    def print_summary(self):
        """Print detailed summary statistics."""
        print("\n" + "="*70)
        print("TRACKING SUMMARY WITH EPHEMERAL FILTERING")
        print("="*70)
        print(f"Total frames processed: {self.current_frame}")
        print(f"Ephemeral threshold: {self.ephemeral_threshold} frames")
        print()

        print("ALL TRACKING RESULTS:")
        print(f"  • Average people per frame: {np.mean(self.num_of_people_in_each_frame):.2f}")
        print(f"  • Max people in single frame: {max(self.num_of_people_in_each_frame) if self.num_of_people_in_each_frame else 0}")
        print(f"  • Min people in single frame: {min(self.num_of_people_in_each_frame) if self.num_of_people_in_each_frame else 0}")
        print(f"  • Total unique IDs tracked: {len(self.all_unique_ids)}")
        print()

        print("EPHEMERAL FILTERING RESULTS:")
        print(f"  • Persistent IDs (≥{self.ephemeral_threshold} frames): {len(self.persistent_ids)}")
        print(f"  • Ephemeral IDs (<{self.ephemeral_threshold} frames): {len(self.ephemeral_ids)}")
        print(f"  • Filtering efficiency: {len(self.ephemeral_ids)/len(self.all_unique_ids)*100:.1f}% IDs removed")
        print()

        if self.filtered_num_of_people_in_each_frame:
            print(f"FILTERED TRACKING STATS:")
            print(f"  • Average persistent people per frame: {np.mean(self.filtered_num_of_people_in_each_frame):.2f}")
            print()

        if self.persistent_id_history:
            print("PERSISTENT ID ANALYSIS:")
            for track_id, frames in sorted(self.persistent_id_history.items()):
                lifespan = len(frames)
                first_frame = min(frames)
                last_frame = max(frames)
                print(f"  ID {track_id}: appeared in {lifespan} frames (frames {first_frame}-{last_frame})")

        if self.ephemeral_ids:
            print(f"\nEPHEMERAL IDs REMOVED: {sorted(list(self.ephemeral_ids))}")


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description='YOLO Tracking Visualization with Ephemeral Filtering')
    parser.add_argument('--ephemeral_threshold', type=int, default=50,
                       help='Minimum frames an ID must appear to be considered persistent (default: 50)')
    parser.add_argument('--model_path', type=str, default="../yolo11n.pt",
                       help='Path to YOLO model weights (default: ../yolo11n.pt)')
    parser.add_argument('--video_path', type=str, default="../videos/8th-grade-vid.mp4",
                       help='Path to input video (default: ../videos/8th-grade-vid.mp4)')

    args = parser.parse_args()

    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"Using ephemeral threshold: {args.ephemeral_threshold} frames")

    visualizer = YOLOTrackingVisualizerWithEphemeralFiltering(
        model_path=args.model_path,
        video_path=args.video_path,
        ephemeral_threshold=args.ephemeral_threshold
    )
    visualizer.process_video()


if __name__ == "__main__":
    main()
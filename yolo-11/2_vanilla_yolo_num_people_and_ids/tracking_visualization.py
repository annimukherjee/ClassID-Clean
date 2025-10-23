
from ultralytics import YOLO
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, deque
import time

class YOLOTrackingVisualizer:
    def __init__(self, model_path="../yolo11l.pt", video_path="../videos/8th-grade-vid.mp4"):
        self.model = YOLO(model_path)
        self.video_path = video_path

        # Data storage for tracking
        self.num_of_people_in_each_frame = []
        self.frame_numbers = []
        self.unique_ids = set()
        self.id_history = defaultdict(list)  # track_id -> [frame_numbers]
        self.current_frame = 0

        # Setup matplotlib for live plotting
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Initialize plots
        self.ax1.set_title('Number of People Detected per Frame')
        self.ax1.set_xlabel('Frame Number')
        self.ax1.set_ylabel('Person Count')
        self.ax1.grid(True)

        self.ax2.set_title('Person ID Tracking Over Time')
        self.ax2.set_xlabel('Frame Number')
        self.ax2.set_ylabel('Person ID')
        self.ax2.grid(True)

        plt.tight_layout()

    def update_plots(self):
        if len(self.frame_numbers) > 0:
            # Update person count plot as bar chart
            self.ax1.clear()
            self.ax1.bar(self.frame_numbers, self.num_of_people_in_each_frame, width=1.0, alpha=0.4, color='blue')
            self.ax1.set_title(f'People Detected per Frame (Current: {self.num_of_people_in_each_frame[-1] if self.num_of_people_in_each_frame else 0})')
            self.ax1.set_xlabel('Frame Number')
            self.ax1.set_ylabel('Person Count')
            self.ax1.grid(True)

            # Update ID tracking plot
            self.ax2.clear()
            colors = plt.cm.tab20(np.linspace(0, 1, len(self.unique_ids)))

            for i, (track_id, frames) in enumerate(self.id_history.items()):
                if frames:  # Only plot if there are frames for this ID
                    color = colors[i % len(colors)]
                    self.ax2.scatter(frames, [track_id] * len(frames),
                                   c=[color], s=20, alpha=0.7, label=f'ID {track_id}')

            self.ax2.set_title(f'Person ID Tracking Over Time (Total IDs: {len(self.unique_ids)})')
            self.ax2.set_xlabel('Frame Number')
            self.ax2.set_ylabel('Person ID')
            self.ax2.grid(True)

            if len(self.unique_ids) <= 20:  # Only show legend if not too many IDs
                self.ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            plt.tight_layout()
            plt.pause(0.01)

    def process_video(self):
        print("Starting YOLO tracking with live visualization...")
        print("Press 'q' in the video window to quit")

        # Process video with tracking
        results = self.model.track(
            self.video_path,
            show=True,
            tracker="bytetrack.yaml",
            classes=[0],  # Only detect persons
            device='mps' if torch.backends.mps.is_available() else 'cpu',
            stream=True,
            verbose=False
        )
        
        # print(results)

        for result in results:
            self.current_frame += 1

            # if self.current_frame==10:
                # print("==========")
                # print(result.boxes)
                # print("==========")
            # Count people in current frame
            person_count = 0
            current_ids = set()

            if result.boxes is not None and len(result.boxes) > 0:
                person_count = len(result.boxes)

                # Extract tracking IDs if available
                if hasattr(result.boxes, 'id') and result.boxes.id is not None:
                    track_ids = result.boxes.id.cpu().numpy().astype(int)
                    current_ids = set(track_ids)

                    # Update tracking history
                    for track_id in track_ids:
                        self.unique_ids.add(track_id)
                        self.id_history[track_id].append(self.current_frame) # self.id_history = {1: [1], 2: [1], 3: [1]}

            # Store data
            self.frame_numbers.append(self.current_frame)
            self.num_of_people_in_each_frame.append(person_count)

            # Update plots every 5 frames for performance
            if self.current_frame % 10 == 0:
                self.update_plots()

            # Print current stats
            if self.current_frame % 30 == 0:  # Print every 30 frames
                print(f"Frame {self.current_frame}: {person_count} people, "
                      f"Total unique IDs seen: {len(self.unique_ids)}")

        # Final plot update
        self.update_plots()

        # Print summary statistics
        self.print_summary()

        # Keep plot open
        print("\nVisualization complete! Close the plot window to exit.")
        plt.ioff()
        plt.show()

    def print_summary(self):
        print("\n" + "="*50)
        print("TRACKING SUMMARY")
        print("="*50)
        print(f"Total frames processed: {self.current_frame}")
        print(f"Average people per frame: {np.mean(self.num_of_people_in_each_frame):.2f}")
        print(f"Max people in single frame: {max(self.num_of_people_in_each_frame) if self.num_of_people_in_each_frame else 0}")
        print(f"Min people in single frame: {min(self.num_of_people_in_each_frame) if self.num_of_people_in_each_frame else 0}")
        print(f"Total unique person IDs tracked: {len(self.unique_ids)}")

        if self.id_history:
            print(f"\nID Tracking Analysis:")
            for track_id, frames in self.id_history.items():
                lifespan = len(frames)
                first_frame = min(frames)
                last_frame = max(frames)
                print(f"  ID {track_id}: appeared in {lifespan} frames "
                      
    
                      f"(frames {first_frame}-{last_frame})")









if __name__ == "__main__":
    print(f"MPS available: {torch.backends.mps.is_available()}")

    visualizer = YOLOTrackingVisualizer()
    visualizer.process_video()
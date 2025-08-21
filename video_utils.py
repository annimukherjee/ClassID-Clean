# video_utils.py

import cv2
import numpy as np
import pandas as pd
import mmcv
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

# --- Helper Functions ---

def get_color_palette(num_colors):
    """Generates a list of distinct BGR colors."""
    # Use a perceptually distinct color map
    cmap = plt.get_cmap('tab20', num_colors) 
    colors = [tuple(int(c * 255) for c in cmap(i)[:3][::-1]) for i in range(num_colors)]
    return colors

def draw_text(frame, text, pos, font_scale=0.8, text_color=(255, 255, 255), bg_color=(0, 0, 0)):
    """Draws text with a background for better visibility."""
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
    # Ensure the background rectangle is drawn within frame bounds
    start_point = (pos[0], pos[1] - text_height - 10)
    end_point = (pos[0] + text_width + 5, pos[1])
    cv2.rectangle(frame, start_point, end_point, bg_color, -1)
    cv2.putText(frame, text, (pos[0], pos[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 2)


class VideoVisualizer:
    """
    Handles the creation of annotated videos for each step of the ID assignment pipeline.
    Annotations from processed frames are persisted until the next processed frame.
    """
    def __init__(self, video_path: str):
        self.video_reader = mmcv.VideoReader(video_path)
        self.fps = self.video_reader.fps
        self.width = self.video_reader.width
        self.height = self.video_reader.height
        self.color_palette = get_color_palette(100) # Increased palette size

    def _get_video_writer(self, output_path: str):
        """Initializes a VideoWriter object."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        return cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))

    def _get_color_for_id(self, track_id):
        """Assigns a consistent color to a track ID."""
        return self.color_palette[track_id % len(self.color_palette)]

    def generate_step_1_oc_sort_video(self, df_tracks: pd.DataFrame, output_path: str):
        print(f"\n[VIDEO] Generating Step 1: OC-SORT Initial Tracking video...")
        writer = self._get_video_writer(output_path)
        
        # Pre-calculate lifespans and prepare for stateful drawing
        lifespans = df_tracks.groupby('track_id')['frame_id'].agg(['min', 'max']).to_dict('index')
        last_known_annotations = {}

        for frame_id, frame in tqdm(enumerate(self.video_reader), total=len(self.video_reader), desc="Video 1/5"):
            draw_text(frame, "Step 1: OC-SORT Initial IDs", (10, 40))
            
            # Update the state with new data if it exists for this frame
            frame_data = df_tracks[df_tracks['frame_id'] == frame_id]
            for _, row in frame_data.iterrows():
                track_id = int(row['track_id'])
                last_known_annotations[track_id] = {
                    'bbox': [int(c) for c in [row['x1'], row['y1'], row['x2'], row['y2']]]
                }
            
            # Draw all active tracks based on their last known state
            for track_id, data in last_known_annotations.items():
                if lifespans[track_id]['min'] <= frame_id <= lifespans[track_id]['max']:
                    color = self._get_color_for_id(track_id)
                    bbox = data['bbox']
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                    draw_text(frame, f"ID: {track_id}", (bbox[0], bbox[1]), bg_color=color)

            writer.write(frame)
        writer.release()
        print(f"[VIDEO] Saved: {output_path}")

    def generate_step_2_ephemeral_filter_video(self, df_before: pd.DataFrame, df_after: pd.DataFrame, output_path: str):
        print(f"\n[VIDEO] Generating Step 2: Ephemeral ID Filtering video...")
        writer = self._get_video_writer(output_path)
        
        persistent_ids = set(df_after['track_id'].unique())
        lifespans = df_before.groupby('track_id')['frame_id'].agg(['min', 'max']).to_dict('index')
        last_known_annotations = {}

        for frame_id, frame in tqdm(enumerate(self.video_reader), total=len(self.video_reader), desc="Video 2/5"):
            draw_text(frame, "Step 2: Filtering Ephemeral (Short-Lived) IDs", (10, 40))
            
            frame_data_before = df_before[df_before['frame_id'] == frame_id]
            for _, row in frame_data_before.iterrows():
                track_id = int(row['track_id'])
                last_known_annotations[track_id] = {
                    'bbox': [int(c) for c in [row['x1'], row['y1'], row['x2'], row['y2']]]
                }

            for track_id, data in last_known_annotations.items():
                if lifespans[track_id]['min'] <= frame_id <= lifespans[track_id]['max']:
                    bbox = data['bbox']
                    if track_id in persistent_ids:
                        color = (0, 255, 0) # Green for kept
                        label = f"ID: {track_id} (Kept)"
                    else:
                        color = (0, 0, 255) # Red for removed
                        label = f"ID: {track_id} (Removed)"

                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                    draw_text(frame, label, (bbox[0], bbox[1]), bg_color=color)

            writer.write(frame)
        writer.release()
        print(f"[VIDEO] Saved: {output_path}")

    def generate_step_3_local_reconciliation_video(self, df_before: pd.DataFrame, df_after: pd.DataFrame, merge_map: dict, output_path: str):
        print(f"\n[VIDEO] Generating Step 3: Local Reconciliation video...")
        writer = self._get_video_writer(output_path)

        lifespans = df_after.groupby('track_id')['frame_id'].agg(['min', 'max']).to_dict('index')
        last_known_annotations = {}

        for frame_id, frame in tqdm(enumerate(self.video_reader), total=len(self.video_reader), desc="Video 3/5"):
            draw_text(frame, "Step 3: Local Reconciliation (Merging Fragments)", (10, 40))
            
            frame_data_after = df_after[df_after['frame_id'] == frame_id]
            for _, row in frame_data_after.iterrows():
                final_id = int(row['track_id'])
                original_row = df_before[(df_before['frame_id'] == frame_id) & (df_before['x1'] == row['x1'])]
                original_id = int(original_row['track_id'].iloc[0]) if not original_row.empty else final_id
                
                label = f"ID: {original_id} -> {final_id}" if original_id in merge_map else f"ID: {final_id}"
                
                last_known_annotations[original_id] = { # Use original ID as the key to track the object
                    'bbox': [int(c) for c in [row['x1'], row['y1'], row['x2'], row['y2']]],
                    'label': label,
                    'final_id': final_id
                }

            for data in last_known_annotations.values():
                # For this video, we draw all boxes from the final dataframe.
                # The lifespan check must use the final_id.
                if lifespans[data['final_id']]['min'] <= frame_id <= lifespans[data['final_id']]['max']:
                    bbox = data['bbox']
                    color = self._get_color_for_id(data['final_id'])
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                    draw_text(frame, data['label'], (bbox[0], bbox[1]), bg_color=color)

            writer.write(frame)
        writer.release()
        print(f"[VIDEO] Saved: {output_path}")
        
    def generate_step_4_feature_extraction_video(self, df_features: pd.DataFrame, output_path: str):
        print(f"\n[VIDEO] Generating Step 4: Feature Extraction video...")
        writer = self._get_video_writer(output_path)

        lifespans = df_features.groupby('track_id')['frame_id'].agg(['min', 'max']).to_dict('index')
        last_known_annotations = {}

        for frame_id, frame in tqdm(enumerate(self.video_reader), total=len(self.video_reader), desc="Video 4/5"):
            draw_text(frame, "Step 4: Feature Extraction (Pose, Face, Gaze)", (10, 40))

            frame_data = df_features[df_features['frame_id'] == frame_id]
            for _, row in frame_data.iterrows():
                track_id = int(row['track_id'])
                # Update the state for this track_id
                last_known_annotations[track_id] = {
                    'p_bbox': [int(c) for c in [row['x1'], row['y1'], row['x2'], row['y2']]],
                    'face_bbox': row['face_bbox'],
                    'gaze_2d': row['gaze_2d'],
                    'keypoints': row['keypoints']
                }

            for track_id, data in last_known_annotations.items():
                if lifespans[track_id]['min'] <= frame_id <= lifespans[track_id]['max']:
                    color = self._get_color_for_id(track_id)
                    p_bbox = data['p_bbox']
                    
                    # Main BBox
                    cv2.rectangle(frame, (p_bbox[0], p_bbox[1]), (p_bbox[2], p_bbox[3]), color, 2)
                    draw_text(frame, f"ID: {track_id}", (p_bbox[0], p_bbox[1]), bg_color=color)

                    # Face BBox
                    if data['face_bbox'] is not None:
                        face_bbox_abs = data['face_bbox'].copy()
                        face_bbox_abs[0::2] += p_bbox[0]
                        face_bbox_abs[1::2] += p_bbox[1]
                        face_bbox_abs = [int(c) for c in face_bbox_abs]
                        cv2.rectangle(frame, (face_bbox_abs[0], face_bbox_abs[1]), (face_bbox_abs[2], face_bbox_abs[3]), (0, 255, 255), 2)
                        
                    # Gaze Vector
                    if data['gaze_2d'] is not None and len(data['gaze_2d']) > 0:
                        points_2d = data['gaze_2d'][0]
                        nose = (int(points_2d[0][0]), int(points_2d[0][1]))
                        gaze_end = (int(points_2d[1][0]), int(points_2d[1][1]))
                        cv2.arrowedLine(frame, nose, gaze_end, (255, 0, 0), 2)

                    # Keypoints
                    if data['keypoints'] is not None:
                        for x, y, conf in data['keypoints']:
                            if conf > 0.5:
                                cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)
            
            writer.write(frame)
        writer.release()
        print(f"[VIDEO] Saved: {output_path}")

    def generate_step_5_global_reconciliation_video(self, df_before: pd.DataFrame, df_after: pd.DataFrame, merge_map: dict, output_path: str):
        print(f"\n[VIDEO] Generating Step 5: Global Reconciliation video...")
        writer = self._get_video_writer(output_path)

        lifespans = df_after.groupby('track_id')['frame_id'].agg(['min', 'max']).to_dict('index')
        last_known_annotations = {}

        for frame_id, frame in tqdm(enumerate(self.video_reader), total=len(self.video_reader), desc="Video 5/5"):
            draw_text(frame, "Step 5: Global Reconciliation (Merging based on Faces)", (10, 40))
            
            frame_data_after = df_after[df_after['frame_id'] == frame_id]
            for _, row in frame_data_after.iterrows():
                final_id = int(row['track_id'])
                original_row = df_before[(df_before['frame_id'] == frame_id) & (df_before['x1'] == row['x1'])]
                original_id = int(original_row['track_id'].iloc[0]) if not original_row.empty else final_id
                
                label = f"ID: {original_id} -> {final_id} (Face Match)" if original_id in merge_map else f"ID: {final_id}"

                last_known_annotations[original_id] = {
                    'bbox': [int(c) for c in [row['x1'], row['y1'], row['x2'], row['y2']]],
                    'label': label,
                    'final_id': final_id
                }
            
            for data in last_known_annotations.values():
                if lifespans[data['final_id']]['min'] <= frame_id <= lifespans[data['final_id']]['max']:
                    bbox = data['bbox']
                    color = self._get_color_for_id(data['final_id'])
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                    draw_text(frame, data['label'], (bbox[0], bbox[1]), bg_color=color)

            writer.write(frame)
        writer.release()
        print(f"[VIDEO] Saved: {output_path}")
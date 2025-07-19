#!/usr/bin/env python3
"""
step2_visualize_features.py (with improved pose visualization)
"""
import os
import argparse
import pickle
import random

import cv2
import mmcv
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

# =============================================================================
# HELPER FUNCTIONS FOR DRAWING
# =============================================================================


def draw_annotations(frame, frame_data, color_map):
    """Draws all annotations for a single frame with improved pose viz."""
    if not frame_data:
        return frame

    for person in frame_data:
        track_id = person.get('track_id')
        if track_id is None:
            continue

        color = color_map.get(track_id, (255, 255, 255))

        # 1. Draw tracking bounding box and ID
        bbox = person['bbox'].astype(int)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        cv2.putText(
            frame, f"ID: {track_id}", (bbox[0], bbox[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )

        # 2. Draw pose keypoints (with improvements)
        if 'keypoints' in person:
            KEYPOINT_RADIUS = 10       # Increased radius for better visibility
            # Use a bright, consistent color (Cyan)
            KEYPOINT_COLOR = (0, 255, 255)
            for x, y, conf in person['keypoints']:
                cv2.circle(frame, (int(x), int(y)),
                           KEYPOINT_RADIUS, KEYPOINT_COLOR, -1)

        # 3. Draw face bounding box
        if 'face' in person and person.get('face') is not None and person['face'].size > 0:
            face_bbox = person['face'][0].astype(int)
            face_bbox[0::2] += bbox[0]
            face_bbox[1::2] += bbox[1]
            cv2.rectangle(
                frame, (face_bbox[0], face_bbox[1]), (face_bbox[2], face_bbox[3]), (0, 0, 255), 2)

    return frame

# =============================================================================
# The rest of the script remains exactly the same as the previous correct version.
# Just copy and paste the functions below this line from the previous response.
# =============================================================================


def create_annotated_video(features_dir, video_path, output_path):
    print(f"\n[INFO] Creating annotated video...")
    feature_files = sorted([os.path.join(features_dir, f)
                           for f in os.listdir(features_dir) if f.endswith('.pickle')])
    features_by_frame = {}
    all_track_ids = set()
    for f_path in feature_files:
        try:
            with open(f_path, 'rb') as f:
                frame_id, frame_data = pickle.load(f)
            features_by_frame[frame_id] = frame_data
            for person in frame_data:
                if person.get('track_id') is not None:
                    all_track_ids.add(person.get('track_id'))
        except (pickle.UnpicklingError, EOFError):
            print(f"[WARN] Could not load pickle file: {f_path}")

    color_map = {tid: (random.randint(50, 255), random.randint(
        50, 255), random.randint(50, 255)) for tid in all_track_ids}

    video_reader = mmcv.VideoReader(video_path)
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(
        *'mp4v'), video_reader.fps, (video_reader.width, video_reader.height))

    if not video_writer.isOpened():
        print(f"[ERROR] Could not open video writer for path: {output_path}")
        return

    for frame_id, frame in enumerate(video_reader):
        if frame_id % 100 == 0:
            print(f"  > Annotating frame {frame_id}/{len(video_reader)}...")
        frame_data = features_by_frame.get(frame_id, [])
        annotated_frame = draw_annotations(frame.copy(), frame_data, color_map)
        video_writer.write(annotated_frame)

    video_writer.release()
    print(f"[SUCCESS] Annotated video saved successfully.")


def create_embedding_tsne_plot(features_dir, output_path, sample_size=2000):
    print(f"\n[INFO] Creating t-SNE plot of facial embeddings...")
    all_embeddings, all_ids = [], []
    feature_files = [os.path.join(features_dir, f) for f in os.listdir(
        features_dir) if f.endswith('.pickle')]

    for f_path in feature_files:
        try:
            with open(f_path, 'rb') as f:
                _, frame_data = pickle.load(f)
            for person in frame_data:
                if 'face_embedding' in person and person.get('track_id') is not None:
                    all_embeddings.append(person['face_embedding'])
                    all_ids.append(person['track_id'])
        except (pickle.UnpicklingError, EOFError):
            continue

    if len(all_embeddings) < 10:
        print("[WARN] Not enough embeddings to generate a t-SNE plot. Skipping.")
        return

    print(
        f"  > Found {len(all_embeddings)} facial embeddings from {len(set(all_ids))} unique raw IDs.")

    if len(all_embeddings) > sample_size:
        indices = np.random.choice(
            len(all_embeddings), sample_size, replace=False)
        embeddings_to_plot, ids_to_plot = np.array(
            all_embeddings)[indices], np.array(all_ids)[indices]
    else:
        embeddings_to_plot, ids_to_plot = np.array(
            all_embeddings), np.array(all_ids)

    n_samples = embeddings_to_plot.shape[0]
    perplexity_value = min(40, n_samples - 1)

    print("  > Running t-SNE... (this may take a moment)")
    tsne = TSNE(n_components=2, verbose=0,
                perplexity=perplexity_value, max_iter=300, random_state=42)
    tsne_results = tsne.fit_transform(embeddings_to_plot)
    print("  > t-SNE complete.")

    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x=tsne_results[:, 0], y=tsne_results[:, 1],
        hue=ids_to_plot,
        palette=sns.color_palette("hsv", len(set(ids_to_plot))),
        legend="full", alpha=0.7
    )
    plt.title("t-SNE Visualization of Facial Embeddings (by Raw Track ID)")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"[SUCCESS] t-SNE plot saved successfully.")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Step 2: Visualize the raw features from Step 1.")
    parser.add_argument('-f', '--features', type=str, required=True,
                        help="Path to the directory containing raw feature pickle files.")
    parser.add_argument('-v', '--video', type=str, required=True,
                        help="Path to the original input video file.")
    parser.add_argument('-o', '--output', type=str, required=True,
                        help="Directory to save the visualization outputs.")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    annotated_video_path = os.path.join(args.output, "annotated_video.mp4")
    create_annotated_video(args.features, args.video, annotated_video_path)
    tsne_plot_path = os.path.join(args.output, "embeddings_tsne_plot.png")
    create_embedding_tsne_plot(args.features, tsne_plot_path)


if __name__ == '__main__':
    main()

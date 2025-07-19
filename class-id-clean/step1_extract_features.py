#!/usr/bin/env python3
"""
step1_extract_features.py (Final, Aligned Version)

This script is aligned with the core logic of the production pipeline.
It extracts all raw features for a single video and uses the more robust
facial embedding preprocessing from the production code.
"""
import os
import time
import argparse
import pickle
import traceback

import cv2
import mmcv
import torch
import numpy as np
from mmtrack.apis import inference_mot, init_model as init_tracking_model
from mmpose.apis import inference_top_down_pose_model, init_pose_model
from concurrent.futures import ThreadPoolExecutor

from FaceWrapper import RetinaFaceInference
from GazeWrapper import GazeInference
from facenet_pytorch import InceptionResnetV1

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    SOURCE_DIR = './'
    TRACK_CONFIG = os.path.join(SOURCE_DIR, 'configs/mmlab/ocsort_yolox_x_crowdhuman_mot17-private-half.py')
    TRACK_CHECKPOINT = os.path.join(SOURCE_DIR, 'models/mmlab/ocsort_yolox_x_crowdhuman_mot17-private-half_20220813_101618-fe150582.pth')
    POSE_CONFIG = os.path.join(SOURCE_DIR, 'configs/mmlab/hrnet_w32_coco_256x192.py')
    POSE_CHECKPOINT = os.path.join(SOURCE_DIR, 'models/mmlab/hrnet_w32-36af842e.pth')
    
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    TARGET_FPS = 5
    FACE_NUM_POOL_WORKERS = 4
    # Use the target size from the production script
    FACE_EMBEDDING_TARGET_SIZE = (244, 244)

# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def initialize_models(config):
    """Loads and initializes all required models."""
    print("[INFO] Initializing all models...")
    device = torch.device(config.DEVICE)
    models = {
        'tracking': init_tracking_model(config.TRACK_CONFIG, config.TRACK_CHECKPOINT, device=device),
        'pose': init_pose_model(config.POSE_CONFIG, config.POSE_CHECKPOINT, device=device),
        'face': RetinaFaceInference(device=device),
        'gaze': GazeInference(device=device),
        'face_embedding': InceptionResnetV1(pretrained='vggface2', device=device).eval()
    }
    print("[INFO] All models initialized successfully.")
    return models

def preprocess_face_for_embedding(face_frame, target_size):
    """
    Resizes and pads a face image to a target size, matching the production
    script's logic for robust preprocessing.
    Resizes it to fit within the target dimensions while preserving the original aspect ratio (to avoid squashing or stretching the face).
    Pads the remaining space with black pixels to make it perfectly square.
    Why it's important: This standardization is key to reliable re-identification later. Without it, the same person's face at different sizes could produce very different embeddings, making matching impossible.
    """
    # 1. Resize while maintaining aspect ratio
    factor_0 = target_size[0] / face_frame.shape[0]
    factor_1 = target_size[1] / face_frame.shape[1]
    factor = min(factor_0, factor_1)
    dsize = (int(face_frame.shape[1] * factor), int(face_frame.shape[0] * factor))
    face_frame_resized = cv2.resize(face_frame, dsize)

    # 2. Pad to the target size
    diff_0 = target_size[0] - face_frame_resized.shape[0]
    diff_1 = target_size[1] - face_frame_resized.shape[1]
    face_frame_padded = np.pad(
        face_frame_resized,
        (
            (diff_0 // 2, diff_0 - diff_0 // 2),
            (diff_1 // 2, diff_1 - diff_1 // 2),
            (0, 0),
        ),
        "constant",
    )

    # 3. Final check to ensure target size is met
    if face_frame_padded.shape[0:2] != target_size:
        face_frame_padded = cv2.resize(face_frame_padded, target_size)

    return face_frame_padded

def run_feature_extraction(video_path, output_dir, models, config):
    """The main processing loop for feature extraction."""
    print(f"\n[INFO] Starting feature extraction for video: {video_path}")
    os.makedirs(output_dir, exist_ok=True)
    video_reader = mmcv.VideoReader(video_path)
    print(f"[INFO] Video opened. Total frames: {len(video_reader)}, FPS: {video_reader.fps:.2f}")
    frame_skip = int(video_reader.fps / config.TARGET_FPS) if config.TARGET_FPS > 0 else 1
    print(f"[INFO] Processing at ~{config.TARGET_FPS} FPS (skipping every {frame_skip} frames).")
    
    pose_dataset = models['pose'].cfg.data['test']['type']
    pose_dataset_info = models['pose'].cfg.data['test'].get('dataset_info', None)
    
    with ThreadPoolExecutor(max_workers=config.FACE_NUM_POOL_WORKERS) as face_executor:
        for frame_id, frame in enumerate(video_reader):
            if frame_id % frame_skip != 0: continue
            if frame_id % (100 * frame_skip) == 0: print(f"  > Processing frame {frame_id}/{len(video_reader)}...")

            try:
                track_results = inference_mot(models['tracking'], frame, frame_id=frame_id)
                persons_data = [dict(bbox=p[1:], track_id=int(p[0])) for p in track_results.get('track_bboxes', [np.array([])])[0]]

                if persons_data:
                    pose_results, _ = inference_top_down_pose_model(models['pose'], frame, persons_data, format='xyxy', dataset=pose_dataset, dataset_info=pose_dataset_info)
                    for i, person in enumerate(persons_data): person['keypoints'] = pose_results[i]['keypoints']
                    
                    body_frames, body_indexes = [], []
                    for i, person in enumerate(persons_data):
                        person_img = frame[int(person['bbox'][1]):int(person['bbox'][3]), int(person['bbox'][0]):int(person['bbox'][2])]
                        if person_img.size > 0:
                            body_frames.append(person_img)
                            body_indexes.append(i)
                    
                    if body_frames:
                        for i, face_result in zip(body_indexes, face_executor.map(models['face'].run, body_frames)):
                            persons_data[i]['face'] = face_result[0]

                    for person in persons_data:
                        if person.get('face') is not None and person['face'].size > 0:
                            bbox = person['bbox'].astype(int)
                            face_abs = person['face'][0].copy()
                            face_abs[0::2] += bbox[0]; face_abs[1::2] += bbox[1]
                            
                            if isinstance(gaze_result := models['gaze'].run(frame, face_abs.reshape(1, -1)), dict): person.update(gaze_result)
                            
                            face_img = frame[int(face_abs[1]):int(face_abs[3]), int(face_abs[0]):int(face_abs[2])]
                            if face_img.size > 0:
                                # Using the aligned preprocessing function
                                preprocessed_face = preprocess_face_for_embedding(face_img, config.FACE_EMBEDDING_TARGET_SIZE)
                                face_pixels = preprocessed_face.astype(np.float32) / 255.0
                                face_tensor = torch.from_numpy(face_pixels).permute(2, 0, 1).unsqueeze(0).to(config.DEVICE)
                                with torch.no_grad():
                                    embedding = models['face_embedding'](face_tensor)
                                person['face_embedding'] = embedding[0].cpu().numpy()

                with open(os.path.join(output_dir, f'{frame_id:06d}.pickle'), 'wb') as f:
                    pickle.dump((frame_id, persons_data), f)

            except Exception as e:
                print(f"[ERROR] Could not process frame {frame_id}. Skipping. Error: {e}")
                traceback.print_exc()
                with open(os.path.join(output_dir, f'{frame_id:06d}.pickle'), 'wb') as f:
                    pickle.dump((frame_id, []), f)

    print(f"\n[SUCCESS] Feature extraction complete. Raw data saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Step 1: Extract raw features from a video.")
    parser.add_argument('-v', '--video', type=str, required=True, help="Path to the input video file.")
    parser.add_argument('-o', '--output', type=str, required=True, help="Directory to save the raw feature pickle files.")
    args = parser.parse_args()
    config = Config()
    models = initialize_models(config)
    run_feature_extraction(args.video, args.output, models, config)

if __name__ == '__main__':
    main()
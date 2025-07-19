#!/usr/bin/env python3
"""
run_classid_pipeline.py

A single, well-structured script to run the complete ClassID within-session pipeline.
"""

import os
import shutil
import time
import argparse
import pickle
import json
import traceback
from copy import deepcopy

import cv2
import mmcv
import torch
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN

from mmtrack.apis import inference_mot, init_model as init_tracking_model
from mmpose.apis import inference_top_down_pose_model, init_pose_model
from facenet_pytorch import InceptionResnetV1
from torchreid.utils import FeatureExtractor as REIDFeatureExtractor

from FaceWrapper import RetinaFaceInference
from GazeWrapper import GazeInference


# ===================================================================================
# == PART 1: CONFIGURATION & SETUP
# ===================================================================================

class ModelConfig:
    SOURCE_DIR = './'
    TRACK_CONFIG = os.path.join(SOURCE_DIR, 'configs/mmlab/ocsort_yolox_x_crowdhuman_mot17-private-half.py')
    TRACK_CHECKPOINT = os.path.join(SOURCE_DIR, 'models/mmlab/ocsort_yolox_x_crowdhuman_mot17-private-half_20220813_101618-fe150582.pth')
    POSE_CONFIG = os.path.join(SOURCE_DIR, 'configs/mmlab/hrnet_w32_coco_256x192.py')
    POSE_CHECKPOINT = os.path.join(SOURCE_DIR, 'models/mmlab/hrnet_w32-36af842e.pth')

class PipelineConfig:
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    TARGET_FPS = 5
    MIN_ID_FRAMES_FILTER = 30
    LOCAL_MAX_ID_DISTANCE = 75
    LOCAL_MIN_BBOX_OVERLAP = 0.4
    GLOBAL_MAX_EMBEDDING_DISTANCE = 0.4
    GLOBAL_MIN_BBOX_OVERLAP = 0.3
    REP_DBSCAN_EPS = 0.4
    REP_DBSCAN_MIN_SAMPLES = 15
    REP_MAX_GAZE_DEVIATION = 30.0


def initialize_models(device):
    print("[INFO] Initializing models...")
    models = {
        'tracking': init_tracking_model(ModelConfig.TRACK_CONFIG, ModelConfig.TRACK_CHECKPOINT, device=device),
        'pose': init_pose_model(ModelConfig.POSE_CONFIG, ModelConfig.POSE_CHECKPOINT, device=device),
        'face': RetinaFaceInference(device=torch.device(device)),
        'gaze': GazeInference(device=device),
        'reid': REIDFeatureExtractor(model_name='osnet_x1_0', device=device),
        'face_embedding': InceptionResnetV1(pretrained='vggface2', device=device).eval()
    }
    print("[INFO] All models initialized successfully.")
    return models


# ===================================================================================
# == PART 2: FEATURE EXTRACTION PIPELINE
# ===================================================================================

def extract_raw_features(video_path, output_dir, models):
    print(f"\n[STEP 1] Starting feature extraction for: {os.path.basename(video_path)}")
    os.makedirs(output_dir, exist_ok=True)
    pose_dataset = models['pose'].cfg.data['test']['type']
    pose_dataset_info = models['pose'].cfg.data['test'].get('dataset_info', None)
    video_reader = mmcv.VideoReader(video_path)
    print(f"[INFO] Video opened. Total frames: {len(video_reader)}, FPS: {video_reader.fps}")
    frame_skip = int(video_reader.fps / PipelineConfig.TARGET_FPS) if PipelineConfig.TARGET_FPS > 0 else 1

    for frame_id, frame in enumerate(video_reader):
        if frame_id % frame_skip != 0: continue
        if frame_id % (100 * frame_skip) == 0: print(f"[INFO] Processing frame {frame_id}/{len(video_reader)}...")
        
        try:
            track_results = inference_mot(models['tracking'], frame, frame_id=frame_id)
            track_bboxes = track_results.get('track_bboxes', [np.array([])])[0]
            persons_data = [dict(bbox=p[1:], track_id=int(p[0])) for p in track_bboxes]

            if persons_data:
                pose_results, _ = inference_top_down_pose_model(models['pose'], frame, persons_data, format='xyxy', dataset=pose_dataset, dataset_info=pose_dataset_info)
                for i, person in enumerate(persons_data):
                    person['keypoints'] = pose_results[i]['keypoints']
                    bbox = person['bbox'].astype(int)
                    person_img = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    if person_img.size == 0: continue
                    face_bbox_list, _ = models['face'].run(person_img)
                    person['face'] = face_bbox_list
                    
                    # --- THE DEFINITIVE FIX ---
                    # This check is now robust and will not cause a ValueError
                    if 'face' in person and person['face'] is not None and person['face'].size > 0:
                        face_bbox_relative = person['face'][0]
                        face_bbox_absolute = face_bbox_relative.copy()
                        face_bbox_absolute[0::2] += bbox[0]
                        face_bbox_absolute[1::2] += bbox[1]
                        gaze_result = models['gaze'].run(frame, face_bbox_absolute.reshape(1, -1))
                        if isinstance(gaze_result, dict): person.update(gaze_result)
                        face_img = frame[int(face_bbox_absolute[1]):int(face_bbox_absolute[3]), int(face_bbox_absolute[0]):int(face_bbox_absolute[2])]
                        if face_img.size > 0:
                            face_tensor = torch.from_numpy(cv2.resize(face_img, (160, 160))).permute(2, 0, 1).unsqueeze(0).float()
                            face_tensor = (face_tensor - 127.5) / 128.0
                            with torch.no_grad():
                                embedding = models['face_embedding'](face_tensor.to(PipelineConfig.DEVICE))
                            person['face_embedding'] = embedding[0].cpu().numpy()
            
            with open(os.path.join(output_dir, f'{frame_id:06d}.pickle'), 'wb') as f:
                pickle.dump((frame_id, persons_data), f)
        
        except Exception as e:
            print(f"[ERROR] Failed to process frame {frame_id}. Error: {e}")
            traceback.print_exc()

    print(f"[SUCCESS] Finished feature extraction. Raw data saved to: {output_dir}")
    return output_dir

# ===================================================================================
# == PART 3 & 4 (Reconciliation & Representation)
# == These parts remain unchanged from the previous correct version.
# ===================================================================================

def _load_and_organize_data(raw_features_dir):
    """Loads all frame pickles and organizes data by track_id."""
    print("[INFO] Loading and organizing raw feature data...")
    all_files = sorted([os.path.join(raw_features_dir, f) for f in os.listdir(raw_features_dir) if f.endswith('.pickle')])
    organized_data = {}
    for file_path in all_files:
        try:
            with open(file_path, 'rb') as f:
                frame_id, frame_results = pickle.load(f)
            for person in frame_results:
                tid = person.get('track_id')
                if tid is not None:
                    if tid not in organized_data: organized_data[tid] = {}
                    organized_data[tid][frame_id] = person
        except (pickle.UnpicklingError, EOFError):
            print(f"[WARN] Could not read pickle file (possibly empty or corrupt): {file_path}")
    print(f"[INFO] Loaded data for {len(organized_data)} unique raw track IDs.")
    return organized_data

def _get_id_start_stop_df(organized_data):
    id_info = []
    for tid, frames in organized_data.items():
        if not frames: continue
        frame_nums = sorted(frames.keys())
        id_info.append({ 'id': tid, 'min_frame': min(frame_nums), 'max_frame': max(frame_nums), 'total_frames': len(frame_nums)})
    return pd.DataFrame(id_info)

def _calculate_bbox_iou(boxA, boxB):
    interArea = max(0, min(boxA[2], boxB[2]) - max(boxA[0], boxB[0])) * max(0, min(boxA[3], boxB[3]) - max(boxA[1], boxB[1]))
    unionArea = ((boxA[2] - boxA[0]) * (boxA[3] - boxA[1])) + ((boxB[2] - boxB[0]) * (boxB[3] - boxB[1])) - interArea
    return interArea / unionArea if unionArea > 0 else 0

def _resolve_chained_maps(mapping_dict):
    for k in list(mapping_dict.keys()):
        v = mapping_dict[k]
        while v in mapping_dict: v = mapping_dict[v]
        mapping_dict[k] = v
    return mapping_dict

def reconcile_ids(raw_features_dir):
    print("\n[STEP 2] Starting ID Reconciliation...")
    organized_data = _load_and_organize_data(raw_features_dir)
    df_id_info = _get_id_start_stop_df(organized_data)
    if df_id_info.empty:
        print("[WARN] No tracking data found to reconcile.")
        return {}, None
    persistent_ids = set(df_id_info[df_id_info['total_frames'] >= PipelineConfig.MIN_ID_FRAMES_FILTER]['id'])
    ephemeral_ids = set(df_id_info.id) - persistent_ids
    df_persistent_info = df_id_info[df_id_info['id'].isin(persistent_ids)].copy()
    print(f"[INFO] Filtered {len(ephemeral_ids)} ephemeral IDs. Kept {len(persistent_ids)} persistent IDs.")
    print("[INFO] Performing Local (Spatio-Temporal) Reconciliation...")
    local_map = {}
    for _, rowA in df_persistent_info.iterrows():
        idA, endA = rowA['id'], rowA['max_frame']
        candidates = df_persistent_info[(df_persistent_info['min_frame'] > endA) & (df_persistent_info['min_frame'] <= endA + PipelineConfig.LOCAL_MAX_ID_DISTANCE)]
        best_iou, best_match = 0, None
        for _, rowB in candidates.iterrows():
            if idA not in organized_data or idB not in organized_data or endA not in organized_data[idA] or rowB['min_frame'] not in organized_data[idB]: continue
            iou = _calculate_bbox_iou(organized_data[idA][endA]['bbox'], organized_data[idB][rowB['min_frame']]['bbox'])
            if iou > best_iou and iou > PipelineConfig.LOCAL_MIN_BBOX_OVERLAP: best_iou, best_match = iou, idB
        if best_match: local_map[best_match] = idA
    print(f"[INFO] Found {len(local_map)} local spatio-temporal merges.")
    print("[INFO] Performing Global (Behavioral) Reconciliation...")
    signatures = {}
    for tid in persistent_ids:
        embeddings = [fd['face_embedding'] for fd in organized_data.get(tid, {}).values() if 'face_embedding' in fd]
        if len(embeddings) < PipelineConfig.REP_DBSCAN_MIN_SAMPLES: continue
        db = DBSCAN(eps=PipelineConfig.REP_DBSCAN_EPS, min_samples=PipelineConfig.REP_DBSCAN_MIN_SAMPLES).fit(embeddings)
        if -1 in db.labels_:
            unique_labels, counts = np.unique(db.labels_[db.labels_ != -1], return_counts=True)
            if unique_labels.size > 0:
                largest_cluster_label = unique_labels[np.argmax(counts)]
                signatures[tid] = np.median(np.array(embeddings)[db.labels_ == largest_cluster_label], axis=0)
    print(f"[INFO] Created {len(signatures)} robust visual signatures.")
    global_map = {}
    id_list = sorted(list(signatures.keys()))
    df_persistent_info.set_index('id', inplace=True)
    for i in range(len(id_list)):
        for j in range(i + 1, len(id_list)):
            idA, idB = id_list[i], id_list[j]
            if df_persistent_info.loc[idA, 'max_frame'] > df_persistent_info.loc[idB, 'min_frame']: continue
            dist = cdist([signatures[idA]], [signatures[idB]], 'cosine')[0][0]
            if dist < PipelineConfig.GLOBAL_MAX_EMBEDDING_DISTANCE:
                avg_iou = _calculate_bbox_iou(np.mean([fd['bbox'] for fd in organized_data[idA].values()], axis=0), np.mean([fd['bbox'] for fd in organized_data[idB].values()], axis=0))
                if avg_iou > PipelineConfig.GLOBAL_MIN_BBOX_OVERLAP: global_map[idB] = idA
    print(f"[INFO] Found {len(global_map)} global behavioral merges.")
    print("[INFO] Finalizing ID mappings...")
    final_map = {tid: tid for tid in persistent_ids}
    final_map.update(local_map); final_map.update(global_map)
    final_map = _resolve_chained_maps(final_map)
    root_ids = sorted(list(set(final_map.values())))
    sequential_map = {root_id: i for i, root_id in enumerate(root_ids)}
    id_map_final_sequential = {raw_id: sequential_map.get(final_map.get(raw_id)) for raw_id in final_map}
    for tid in ephemeral_ids: id_map_final_sequential[tid] = -1
    print(f"[SUCCESS] ID Reconciliation complete. Consolidated {len(organized_data)} raw IDs into {len(root_ids)} final persistent IDs.")
    return id_map_final_sequential, organized_data

def generate_session_representations(final_id_map, organized_data):
    print("\n[STEP 3] Generating Session-Level Representations...")
    final_ids_to_raw = {}
    for raw_id, final_id in final_id_map.items():
        if final_id != -1 and final_id is not None:
            if final_id not in final_ids_to_raw: final_ids_to_raw[final_id] = []
            final_ids_to_raw[final_id].append(raw_id)
    session_representations = {}
    for final_id, raw_ids in final_ids_to_raw.items():
        all_embeddings, all_face_bboxes = [], []
        for raw_id in raw_ids:
            for frame_data in organized_data.get(raw_id, {}).values():
                if 'rvec' in frame_data and frame_data['rvec'] is not None:
                    pitch, yaw, _ = np.rad2deg(frame_data['rvec'][0])
                    if abs(pitch) < PipelineConfig.REP_MAX_GAZE_DEVIATION and abs(yaw) < PipelineConfig.REP_MAX_GAZE_DEVIATION and 'face_embedding' in frame_data:
                        all_embeddings.append(frame_data['face_embedding'])
                if 'face' in frame_data and frame_data['face'] is not None and frame_data['face'].size > 0:
                    all_face_bboxes.append(frame_data['face'][0])
        if not all_embeddings or not all_face_bboxes:
            print(f"[WARN] No valid data for final_id {final_id}. Skipping.")
            continue
        df_faces = pd.DataFrame(all_face_bboxes, columns=['x1', 'y1', 'x2', 'y2'])
        session_representations[final_id] = {
            'visual_signature': np.median(all_embeddings, axis=0).tolist(),
            'positional_features': {
                'median_face_width': float((df_faces['x2'] - df_faces['x1']).median()),
                'median_face_height': float((df_faces['y2'] - df_faces['y1']).median()),
                'median_face_area': float(((df_faces['x2'] - df_faces['x1']) * (df_faces['y2'] - df_faces['y1'])).median())
            }
        }
    print(f"[SUCCESS] Generated representations for {len(session_representations)} students.")
    return session_representations

def main(args):
    print("="*50 + "\nStarting ClassID Within-Session Pipeline\n" + "="*50)
    video_path = args.video
    if not os.path.exists(video_path):
        print(f"[FATAL] Video file not found at: {video_path}"); return
    session_name = os.path.splitext(os.path.basename(video_path))[0]
    base_output_dir = "classid_output"
    raw_features_dir = os.path.join(base_output_dir, session_name, 'raw_features')
    models = initialize_models(PipelineConfig.DEVICE)
    extract_raw_features(video_path, raw_features_dir, models)
    final_id_map, organized_data = reconcile_ids(raw_features_dir)
    if not final_id_map:
        print("[FATAL] ID Reconciliation failed. Exiting."); return
    session_reps = generate_session_representations(final_id_map, organized_data)
    output_file = os.path.join(base_output_dir, session_name, 'session_representations.json')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump({k: v for k, v in sorted(session_reps.items())}, f, indent=4)
    print(f"\n[FINAL SUCCESS] Pipeline complete. Final representations saved to: {output_file}")
    if not args.keep_raw:
        print(f"[INFO] Cleaning up raw features directory: {raw_features_dir}")
        shutil.rmtree(raw_features_dir)
    print("="*50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the ClassID Within-Session Pipeline.")
    parser.add_argument('-v', '--video', type=str, required=True, help="Path to the input video file.")
    parser.add_argument('--keep-raw', action='store_true', help="Do not delete the raw_features directory after processing.")
    args = parser.parse_args()
    main(args)